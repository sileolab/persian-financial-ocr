"""Post-processing module for cleaning, normalizing, and structuring Persian/Arabic OCR text."""

import re
from typing import Any, Dict, List, Optional, Tuple

# Mapping tables for character normalization
PERSIAN_TO_ENGLISH_DIGITS = {
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

ARABIC_TO_ENGLISH_DIGITS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

ENGLISH_TO_PERSIAN_DIGITS = {
    "0": "۰", "1": "۱", "2": "۲", "3": "۳", "4": "۴",
    "5": "۵", "6": "۶", "7": "۷", "8": "۸", "9": "۹",
}

# Arabic to Persian character mapping
ARABIC_TO_PERSIAN_CHARS = {
    "ي": "ی",  # Arabic Yeh to Persian Yeh
    "ى": "ی",  # Alef Maksura to Persian Yeh
    "ك": "ک",  # Arabic Kaf to Persian Keheh
    "ة": "ه",  # Teh Marbuta to Heh
    "ؤ": "و",  # Waw with Hamza
    "إ": "ا",  # Alef with Hamza below
    "أ": "ا",  # Alef with Hamza above
    "آ": "آ",
}

# Diacritics and Tashkeel regex pattern
TASHKEEL_REGEX = re.compile(r"[\u064B-\u0652\u0670\u0640]")  # includes Tatweel (ـ)
MULTI_SPACE_REGEX = re.compile(r"[ \t]+")
MULTI_NEWLINE_REGEX = re.compile(r"\n{3,}")
ZWNJ = "\u200c"


def normalize_digits(text: str, to_english: bool = True) -> str:
    """
    Convert Persian and Arabic numerals to standard English digits (or vice-versa).
    Ensures arithmetic and regex extraction works deterministically.
    """
    if not text:
        return ""

    if to_english:
        result = []
        for char in text:
            if char in PERSIAN_TO_ENGLISH_DIGITS:
                result.append(PERSIAN_TO_ENGLISH_DIGITS[char])
            elif char in ARABIC_TO_ENGLISH_DIGITS:
                result.append(ARABIC_TO_ENGLISH_DIGITS[char])
            else:
                result.append(char)
        return "".join(result)
    else:
        result = []
        for char in text:
            if char in ENGLISH_TO_PERSIAN_DIGITS:
                result.append(ENGLISH_TO_PERSIAN_DIGITS[char])
            elif char in ARABIC_TO_ENGLISH_DIGITS:
                eng = ARABIC_TO_ENGLISH_DIGITS[char]
                result.append(ENGLISH_TO_PERSIAN_DIGITS.get(eng, char))
            else:
                result.append(char)
        return "".join(result)


def normalize_persian_letters(text: str) -> str:
    """Unify Arabic variants of Yeh, Kaf, and Heh into standard Persian characters."""
    if not text:
        return ""
    result = []
    for char in text:
        result.append(ARABIC_TO_PERSIAN_CHARS.get(char, char))
    return "".join(result)


def remove_diacritics(text: str) -> str:
    """Strip Arabic/Persian vowels, Tanwin, Sukun, Tashdid, and Tatweel."""
    if not text:
        return ""
    return TASHKEEL_REGEX.sub("", text)


def normalize_spacing(text: str) -> str:
    """
    Standardize spaces and Persian zero-width non-joiners (ZWNJ).
    Fixes fragmented words commonly encountered in OCR output.
    """
    if not text:
        return ""

    # Replace non-breaking spaces with standard space
    text = text.replace("\u00a0", " ")
    text = MULTI_SPACE_REGEX.sub(" ", text)

    # Normalize Persian affixes with ZWNJ (نیم‌فاصله)
    # e.g., 'می ' or 'نمی ' at the beginning of verbs
    text = re.sub(r"(^|\s)(می|نمی)\s+", rf"\1\2{ZWNJ}", text)
    # e.g., ' ها' at the end of plural nouns
    text = re.sub(r"\s+(ها|های|هایی)(\s|$)", rf"{ZWNJ}\1\2", text)
    # e.g., ' تر' / ' ترین'
    text = re.sub(r"\s+(تر|ترین)(\s|$)", rf"{ZWNJ}\1\2", text)

    text = MULTI_NEWLINE_REGEX.sub("\n\n", text)
    return text.strip()


def clean_ocr_text(text: str, digits_to_english: bool = True) -> str:
    """Complete cleaning pipeline for raw OCR output string."""
    if not text:
        return ""
    cleaned = normalize_persian_letters(text)
    cleaned = remove_diacritics(cleaned)
    cleaned = normalize_digits(cleaned, to_english=digits_to_english)
    cleaned = normalize_spacing(cleaned)
    return cleaned


def group_into_rows(
    boxes: List[Dict[str, Any]], y_threshold: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Group detected OCR boxes into horizontal visual lines and sort Right-to-Left.
    Returns:
        (flattened_sorted_boxes, grouped_rows)
    """
    if not boxes:
        return [], []

    # Calculate center-y, center-x, and box height for each box
    enriched = []
    for item in boxes:
        box = item.get("box", [])
        if len(box) == 4 and isinstance(box[0], (list, tuple)):
            # 4 polygon points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0
            h = max(ys) - min(ys)
        elif len(box) == 4:
            # [x, y, w, h]
            x, y, w, h_val = box
            cx = x + w / 2.0
            cy = y + h_val / 2.0
            h = h_val
        else:
            cx, cy, h = 0.0, 0.0, 20.0

        enriched.append({**item, "_cy": cy, "_cx": cx, "_h": max(15.0, float(h))})

    # Sort primarily by vertical Y position
    enriched.sort(key=lambda b: b["_cy"])

    # Group into visual lines based on vertical overlap
    rows: List[List[Dict[str, Any]]] = []
    current_row: List[Dict[str, Any]] = []

    for item in enriched:
        if not current_row:
            current_row.append(item)
        else:
            avg_cy = sum(b["_cy"] for b in current_row) / len(current_row)
            avg_h = sum(b["_h"] for b in current_row) / len(current_row)
            threshold = y_threshold if y_threshold is not None else max(20.0, avg_h * 0.70)
            if abs(item["_cy"] - avg_cy) <= threshold:
                current_row.append(item)
            else:
                rows.append(current_row)
                current_row = [item]

    if current_row:
        rows.append(current_row)

    # Within each horizontal row, sort Right-to-Left (descending X in image coordinates for Persian RTL)
    flattened_boxes: List[Dict[str, Any]] = []
    cleaned_rows: List[List[Dict[str, Any]]] = []

    for row in rows:
        row.sort(key=lambda b: b["_cx"], reverse=True)
        row_cleaned = []
        for item in row:
            clean_item = dict(item)
            clean_item.pop("_cy", None)
            clean_item.pop("_cx", None)
            clean_item.pop("_h", None)
            row_cleaned.append(clean_item)
            flattened_boxes.append(clean_item)
        cleaned_rows.append(row_cleaned)

    return flattened_boxes, cleaned_rows


def sort_reading_order(boxes: List[Dict[str, Any]], y_threshold: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Sort OCR detected bounding boxes in natural Persian reading order:
    Top-to-bottom, and Right-to-Left for items on the same horizontal row.
    """
    flattened, _ = group_into_rows(boxes, y_threshold=y_threshold)
    return flattened


class OCRPostProcessor:
    """Orchestrator for text cleanup, numeral standardization, and reading-order layout."""

    def __init__(self, digits_to_english: bool = True, line_threshold: Optional[int] = None):
        self.digits_to_english = digits_to_english
        self.line_threshold = line_threshold

    def process(self, raw_lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process a list of raw OCR lines:
        1. Clean individual line texts (Persian letters, digits, ZWNJ).
        2. Group and sort into natural Persian reading order (RTL, Top-to-Bottom).
        3. Merge words on the same horizontal line into unified text lines.
        Returns: (processed_lines, full_text)
        """
        cleaned_lines = []
        for line in raw_lines:
            orig_text = line.get("text", "")
            cleaned = clean_ocr_text(orig_text, digits_to_english=self.digits_to_english)
            if cleaned:
                cleaned_lines.append({
                    **line,
                    "text": cleaned,
                    "raw_text": orig_text,
                })

        sorted_boxes, rows = group_into_rows(cleaned_lines, y_threshold=self.line_threshold)

        # Merge boxes on the same horizontal row with space to preserve label-value semantics
        row_strings = []
        for row in rows:
            row_str = " ".join(b["text"] for b in row if b.get("text"))
            if row_str.strip():
                row_strings.append(row_str.strip())

        full_text = "\n".join(row_strings)
        return sorted_boxes, full_text
