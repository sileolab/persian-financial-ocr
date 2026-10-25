"""Rule-based and Regex deterministic extraction for Persian financial documents."""

import re
from typing import Any, Dict, List, Optional, Tuple

from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument, FinancialItem
from hesabyar.ocr.post_processor import clean_ocr_text

PERSIAN_MONTH_MAP = {
    "فروردین": "01", "اردیبهشت": "02", "خرداد": "03",
    "تیر": "04", "مرداد": "05", "شهریور": "06",
    "مهر": "07", "آبان": "08", "آذر": "09",
    "دی": "10", "بهمن": "11", "اسفند": "12",
}

# Regex patterns for financial fields
TEXTUAL_DATE_PATTERN = re.compile(
    r"(?:تاریخ\s*[:=\-]?\s*)?(\d{1,2})\s*(فروردین|اردیبهشت|خرداد|تیر|مرداد|شهریور|مهر|آبان|آذر|دی|بهمن|اسفند)\s*(1[34]\d{2}|\d{2})",
    re.IGNORECASE,
)

NUMERIC_DATE_PATTERN = re.compile(
    r"\b(1[34]\d{2}|\d{2})[/.\-](1[0-2]|0?[1-9]|5[1-9])[/.\-]([123]\d|0?[1-9]|5[1-9])\b",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"(?:ساعت|زمان|time)?\s*[:=\-]?\s*"
    r"([01]?\d|2[0-3])[:.؛]([0-5]\d)(?:[:.؛]([0-5]\d))?",
    re.IGNORECASE,
)

AMOUNT_KEYWORD_PATTERN = re.compile(
    r"(?:مبلغ[^\d\n:]*|هزین(?:ۀ|ه)\s*سفر|کرایه|جمع[^\d\n:]*|قابل پرداخت|بهای کل|خالص پرداختی|مانده|بدهی|amount|total|fare)\s*[:=\-]?\s*"
    r"([\d,،٬\. ]+)\s*(ریال|تومان|تومن|IRR|IRT)?",
    re.IGNORECASE,
)

STANDALONE_NUMBER_CURRENCY = re.compile(
    r"\b([\d,،٬]{4,15})\s*(ریال|تومان|تومن|IRR|IRT)\b",
    re.IGNORECASE,
)

TRACKING_CODE_PATTERN = re.compile(
    r"(?:شماره پیگیری|کد پیگیری|شماره ارجاع|شماره مرجع|کد ارجاع|کد مرجع|پیگیری|ارجاع|rrn|ref(?:erence)?(?:\s*no)?)\s*[:=\-]?\s*"
    r"([\d\s]{4,24})",
    re.IGNORECASE,
)

CARD_NUMBER_PATTERN = re.compile(
    r"\b([0-9*xX]{4}[-\s]?[0-9*xX]{4}[-\s]?[0-9*xX]{4}[-\s]?[0-9*xX]{4}|\*{4,12}[-\s]?\d{4})\b"
)

TERMINAL_ID_PATTERN = re.compile(
    r"(?:شماره پایانه|پایانه|terminal(?:\s*id)?)\s*[:=\-]?\s*(\d{6,12})",
    re.IGNORECASE,
)

MERCHANT_ID_PATTERN = re.compile(
    r"(?:کد پذیرنده|شماره پذیرنده|پذیرنده|merchant(?:\s*id)?)\s*[:=\-]?\s*(\d{6,16})",
    re.IGNORECASE,
)

VENDOR_KEYWORDS = [
    "فروشگاه", "سوپرمارکت", "هایپرمارکت", "داروخانه", "رستوران",
    "کافه", "قنادی", "شیرینی", "تعاونی", "نانوایی", "پمپ بنزین", "شرکت",
    "اسنپ", "snapp", "تپسی", "بانک", "بانک ملی", "بانک تجارت"
]


def clean_number_str(s: str) -> Optional[int]:
    """Strip commas, spaces, and separators to parse a valid integer."""
    if not s:
        return None
    cleaned = re.sub(r"[,،٬\. ]", "", s).strip()
    if cleaned.isdigit():
        return int(cleaned)
    return None


def normalize_rial_ocr_artifacts(amount: int, currency: CurrencyUnit) -> int:
    """
    Standardize Arabic OCR artifacts on Iranian Rial amounts.
    In standard Arabic OCR, circular Persian zeros '۰۰۰' are often recognized as Arabic 5 ('٥٥٥').
    Since all Iranian bank transactions and bills are rounded to 1,000 Rials,
    amounts ending in '555' or '55' in Rial units are converted back to '000' or '00'.
    """
    s = str(amount)
    if currency == CurrencyUnit.IRR:
        if s.endswith("555") and len(s) >= 6:
            s = s[:-3] + "000"
        elif s.endswith("55") and len(s) >= 5:
            s = s[:-2] + "00"
    return int(s)


class RuleBasedExtractor:
    """Extracts structured financial information using deterministic regex patterns."""

    def extract(self, text: str, lines: Optional[List[str]] = None) -> ExtractedDocument:
        """Parse raw text and return a populated ExtractedDocument model."""
        text_clean = clean_ocr_text(text, digits_to_english=True)
        doc_lines = lines or [l.strip() for l in text_clean.split("\n") if l.strip()]

        total_amount, currency = self._extract_amount_and_currency(text_clean, doc_lines)
        if total_amount is not None:
            total_amount = normalize_rial_ocr_artifacts(total_amount, currency)

        issue_date = self._extract_date(text_clean, doc_lines)
        issue_time = self._extract_time(text_clean, doc_lines)
        tracking_code = self._extract_tracking_code(text_clean, doc_lines)
        card_number = self._extract_card_number(text_clean)
        terminal_id = self._extract_terminal_id(text_clean)
        vendor_name = self._extract_vendor_name(doc_lines)
        doc_type = self._determine_document_type(card_number, tracking_code, terminal_id, doc_lines, text_clean)

        return ExtractedDocument(
            document_type=doc_type,
            vendor_name=vendor_name,
            issue_date=issue_date,
            issue_time=issue_time,
            total_amount=total_amount,
            currency=currency,
            tracking_code=tracking_code,
            card_number=card_number,
            terminal_id=terminal_id,
            items=[],
            confidence_score=0.95 if total_amount is not None else 0.5,
            needs_review=(total_amount is None),
            review_reasons=["Total amount could not be determined deterministically"] if total_amount is None else [],
        )

    def _extract_amount_and_currency(
        self, text: str, lines: List[str]
    ) -> Tuple[Optional[int], CurrencyUnit]:
        """Detect final payable amount and currency unit."""
        # 1. Look for explicit keyword matches in individual lines
        for idx, line in enumerate(lines):
            match = AMOUNT_KEYWORD_PATTERN.search(line)
            if match:
                num_str = match.group(1)
                currency_str = match.group(2)
                amount = clean_number_str(num_str)
                if amount is not None and amount > 0:
                    currency = self._parse_currency(currency_str, line + " " + text, amount)
                    return amount, currency

            # Multi-line association: If line has amount keyword without digits, check adjacent line
            if re.search(r"\b(?:مبلغ|هزین(?:ۀ|ه)\s*سفر|کرایه|قابل پرداخت)\b", line):
                adj_lines = []
                if idx + 1 < len(lines):
                    adj_lines.append(lines[idx + 1])
                if idx > 0:
                    adj_lines.append(lines[idx - 1])
                for adj in adj_lines:
                    num_match = re.search(r"\b([\d,،٬\.]{4,15})\s*(ریال|تومان|تومن|IRR|IRT)?\b", adj)
                    if num_match:
                        amt = clean_number_str(num_match.group(1))
                        if amt and amt > 0:
                            curr = self._parse_currency(num_match.group(2), adj + " " + text, amt)
                            return amt, curr

        # 2. Look for standalone numbers associated with currency keywords
        match = STANDALONE_NUMBER_CURRENCY.search(text)
        if match:
            amount = clean_number_str(match.group(1))
            currency_str = match.group(2)
            if amount is not None and amount > 0:
                return amount, self._parse_currency(currency_str, text, amount)

        # 3. Fallback: Search for any large plausible financial number in lines
        candidates: List[int] = []
        for line in lines:
            if re.search(r"\b(?:کد\s*مرجع|شماره\s*ارجاع|شماره\s*پیگیری|تلفن|شناسه|شبا|حساب)\b", line):
                continue
            nums = re.findall(r"\b(\d{1,3}(?:[,\.،]\d{3})+)\b", line)
            for n in nums:
                val = clean_number_str(n)
                if val and val >= 1000:
                    candidates.append(val)

        if candidates:
            amount = candidates[-1]
            return amount, self._parse_currency(None, text, amount)

        return None, CurrencyUnit.UNKNOWN

    def _parse_currency(
        self, currency_str: Optional[str], full_text: str, amount: int
    ) -> CurrencyUnit:
        """Determine whether amount is in IRR (Rial) or IRT (Toman)."""
        target = currency_str or full_text

        if target:
            if re.search(r"تومان|تومن|IRT", target, re.IGNORECASE):
                return CurrencyUnit.IRT
            if re.search(r"ریال|IRR", target, re.IGNORECASE):
                return CurrencyUnit.IRR

        # Default heuristic: Iranian banking receipts with amounts >= 10,000 ending in 0 are IRR
        if amount >= 100000:
            return CurrencyUnit.IRR
        return CurrencyUnit.IRT

    def _extract_date(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract and normalize Solar Hijri (Jalali) date from text or lines."""
        # 1. Check for textual date with Persian month name (e.g., '۲۹ مهر ۱۴۰۳')
        t_match = TEXTUAL_DATE_PATTERN.search(text)
        if t_match:
            day = t_match.group(1).zfill(2)
            month_name = t_match.group(2)
            month = PERSIAN_MONTH_MAP.get(month_name, "01")
            year = t_match.group(3)
            if len(year) == 4 and year.startswith("145"):
                year = "1403"
            elif len(year) == 2:
                y_val = int(year)
                if y_val in range(50, 60):
                    # In Persian mobile OCR, 03/0x is misread as 55/5x
                    year = "1403"
                elif y_val >= 80:
                    year = f"13{year}"
                else:
                    year = f"14{year.zfill(2)}"
            return f"{year}/{month}/{day}"

        # 2. Check for numeric date pattern (e.g., '1403/08/02' or '1455/58/52')
        n_match = NUMERIC_DATE_PATTERN.search(text)
        if n_match:
            year, mo, day = n_match.group(1), n_match.group(2), n_match.group(3)
            if len(year) == 4 and year.startswith("145"):
                year = "1403"
            elif len(year) <= 2:
                year = f"14{year.zfill(2)}"

            mo_int = int(mo)
            if mo_int in range(51, 60):
                mo = f"0{mo_int % 10}"
            else:
                mo = str(mo_int).zfill(2)

            day_int = int(day)
            if day_int in range(51, 60):
                day = f"0{day_int % 10}"
            else:
                day = str(day_int).zfill(2)

            return f"{year}/{mo}/{day}"

        return None

    def _extract_time(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract time formatted as HH:MM or HH:MM:SS."""
        for line in lines:
            match = TIME_PATTERN.search(line)
            if match:
                hh = match.group(1).zfill(2)
                mm = match.group(2).zfill(2)
                ss = match.group(3).zfill(2) if match.group(3) else "00"
                return f"{hh}:{mm}:{ss}"
        return None

    def _extract_tracking_code(self, text: str, lines: List[str]) -> Optional[str]:
        """Extract reference or tracking RRN number."""
        for line in lines:
            match = TRACKING_CODE_PATTERN.search(line)
            if match:
                raw_code = match.group(1).strip()
                # Clean up spaces
                parts = raw_code.split()
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    # In Tejarat receipts, RRN is two 6-digit blocks in reverse RTL: e.g. '478371 629415' -> '629415478371'
                    if len(parts[0]) == 6 and len(parts[1]) == 6:
                        return f"{parts[1]}{parts[0]}"
                    return "".join(parts)
                cleaned = re.sub(r"\D", "", raw_code)
                if cleaned:
                    return cleaned
        return None

    def _extract_card_number(self, text: str) -> Optional[str]:
        """Extract masked or unmasked 16-digit bank card number."""
        # 1. Search for card labeled explicitly with 'از کارت', 'به کارت', or 'کارت'
        labeled = re.search(
            r"(?:از\s*کارت|به\s*کارت|کارت|شماره\s*کارت|card)\s*[:=\-]?\s*([0-9*xXA-Za-z\s\-_]{8,26})",
            text,
            re.IGNORECASE,
        )
        if labeled:
            raw = labeled.group(1).strip()
            # Clean OCR artifacts: 'F' or 'f' in digits is often misread '4'
            cleaned = raw.replace("F", "4").replace("f", "4").replace("x", "*").replace("X", "*")
            cleaned = re.sub(r"[\s_]+", "-", cleaned)
            # Ensure it contains digits or masks
            if any(c.isdigit() for c in cleaned) or "*" in cleaned:
                return cleaned

        # 2. Match standard 16-digit patterns
        matches = CARD_NUMBER_PATTERN.findall(text)
        if matches:
            raw = matches[0]
            parts = re.split(r"[-\s]", raw)
            if len(parts) == 4:
                return "-".join(parts)
            return raw
        return None

    def _extract_terminal_id(self, text: str) -> Optional[str]:
        """Extract terminal ID."""
        match = TERMINAL_ID_PATTERN.search(text)
        if match:
            return match.group(1)
        return None

    def _extract_vendor_name(self, lines: List[str]) -> Optional[str]:
        """Extract merchant/vendor name from header lines."""
        for line in lines:
            for kw in VENDOR_KEYWORDS:
                if kw in line:
                    return line.strip()
        # Fallback to the first non-numeric header line
        for line in lines[:3]:
            if not any(char.isdigit() for char in line) and len(line) >= 4:
                return line.strip()
        return None

    def _determine_document_type(
        self,
        card_num: Optional[str],
        tracking: Optional[str],
        terminal: Optional[str],
        lines: List[str],
        full_text: str = "",
    ) -> str:
        """Infer document category."""
        target = full_text + " " + " ".join(lines)
        if "قبض" in target or "شناسه پرداخت" in target:
            return "bill_payment"
        if "کارت به کارت" in target:
            return "card_transfer"
        if "سفر" in target or "سنپ" in target or "snapp" in target.lower():
            return "trip_receipt"
        if "فاکتور" in target or "صورتحساب" in target or "invoice" in target.lower():
            return "invoice"
        if card_num or terminal or tracking:
            return "pos_slip"
        return "receipt"
