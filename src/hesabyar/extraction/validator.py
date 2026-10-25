"""Financial validation, checksums, and discrepancy detection for extracted documents."""

import re
from typing import List, Optional

from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument


def luhn_checksum_valid(card_number_str: str) -> bool:
    """Verify Luhn algorithm checksum on 16-digit card numbers."""
    digits_only = re.sub(r"\D", "", card_number_str)
    if len(digits_only) != 16:
        return False

    total = 0
    reverse_digits = digits_only[::-1]
    for i, char in enumerate(reverse_digits):
        n = int(char)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class DocumentValidator:
    """Validates financial consistency, amounts, dates, and card checksums."""

    def __init__(
        self,
        min_amount: int = 1000,
        max_amount: int = 50_000_000_000,
        confidence_threshold: float = 0.75,
    ):
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.confidence_threshold = confidence_threshold

    def validate(self, doc: ExtractedDocument) -> ExtractedDocument:
        """
        Execute comprehensive validation checks on an ExtractedDocument.
        Flags needs_review=True if any discrepancies or invalid values are found.
        """
        reasons: List[str] = list(doc.review_reasons)

        # 1. Total Amount Check
        if doc.total_amount is None:
            reasons.append("Total payable amount was not detected.")
        elif doc.total_amount < self.min_amount:
            reasons.append(f"Amount {doc.total_amount} is below plausible threshold ({self.min_amount}).")
        elif doc.total_amount > self.max_amount:
            reasons.append(f"Amount {doc.total_amount} exceeds plausible upper bound ({self.max_amount}).")

        # 2. Line Items Cross-Check
        if doc.items:
            item_sum = 0
            has_valid_items = False
            for it in doc.items:
                if it.total_price is not None:
                    item_sum += it.total_price
                    has_valid_items = True

            if has_valid_items and doc.total_amount is not None:
                if item_sum != doc.total_amount:
                    # Check if factor of 10 discrepancy (Toman vs Rial)
                    if item_sum * 10 == doc.total_amount:
                        reasons.append(
                            f"Sum of items ({item_sum:,}) appears to be in Toman while Total ({doc.total_amount:,}) is in Rial."
                        )
                    elif doc.total_amount * 10 == item_sum:
                        reasons.append(
                            f"Sum of items ({item_sum:,}) appears to be in Rial while Total ({doc.total_amount:,}) is in Toman."
                        )
                    else:
                        diff = abs(doc.total_amount - item_sum)
                        reasons.append(
                            f"Sum of line items ({item_sum:,}) does not equal total amount ({doc.total_amount:,}). Diff: {diff:,}."
                        )

        # 3. Jalali Date Sanity Check
        if doc.issue_date:
            parts = doc.issue_date.split("/")
            if len(parts) == 3:
                try:
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    if not (1380 <= y <= 1420):
                        reasons.append(f"Year {y} is outside the expected Solar Hijri range (1380-1420).")
                    if not (1 <= m <= 12):
                        reasons.append(f"Month {m} is out of calendar range (1-12).")
                    if not (1 <= d <= 31):
                        reasons.append(f"Day {d} is out of calendar range (1-31).")
                except ValueError:
                    reasons.append(f"Invalid non-integer date components in {doc.issue_date}.")
            else:
                reasons.append(f"Invalid date format: {doc.issue_date}")

        # 4. Bank Card Luhn Check (only for unmasked cards)
        if doc.card_number and "*" not in doc.card_number and "x" not in doc.card_number.lower():
            digits = re.sub(r"\D", "", doc.card_number)
            if len(digits) == 16 and not luhn_checksum_valid(digits):
                reasons.append(f"Unmasked card number {doc.card_number} failed Luhn checksum.")

        # 5. Confidence Score Threshold
        if doc.confidence_score < self.confidence_threshold:
            reasons.append(f"Confidence score {doc.confidence_score:.2f} is below acceptance threshold {self.confidence_threshold}.")

        # Update review flag
        needs_review = len(reasons) > 0

        # Return updated document model
        return doc.model_copy(
            update={
                "needs_review": needs_review,
                "review_reasons": list(set(reasons)),
            }
        )
