"""Unit tests for rule-based extraction and financial validation."""

import pytest

from hesabyar.extraction.rules import RuleBasedExtractor
from hesabyar.extraction.validator import DocumentValidator, luhn_checksum_valid
from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument, FinancialItem


@pytest.fixture
def sample_pos_receipt_text():
    return """
    فروشگاه هایپرمارکت بهاران
    پذیرنده: 12345678
    شماره پایانه: 87654321
    تاریخ: 1403/08/04  ساعت: 18:32:15
    شماره کارت: 6037-9971-****-1234
    کد پیگیری: 456789123
    مبلغ کل: 250,000 تومان
    خرید با موفقیت انجام شد
    """


@pytest.fixture
def sample_invoice_text():
    return """
    شرکت پخش افق روشن
    فاکتور فروش کالا و خدمات
    تاریخ فاکتور: 1403/07/15
    مبلغ کل فاکتور: 12,500,000 ریال
    کد مرجع: 99887766
    """


def test_rule_based_extractor_pos_receipt(sample_pos_receipt_text):
    extractor = RuleBasedExtractor()
    doc = extractor.extract(sample_pos_receipt_text)

    assert doc.document_type == "pos_slip"
    assert doc.total_amount == 250000
    assert doc.currency == CurrencyUnit.IRT
    assert doc.issue_date == "1403/08/04"
    assert doc.issue_time == "18:32:15"
    assert doc.tracking_code == "456789123"
    assert "6037" in doc.card_number
    assert "هایپرمارکت بهاران" in doc.vendor_name


def test_rule_based_extractor_invoice(sample_invoice_text):
    extractor = RuleBasedExtractor()
    doc = extractor.extract(sample_invoice_text)

    assert doc.total_amount == 12500000
    assert doc.currency == CurrencyUnit.IRR
    assert doc.issue_date == "1403/07/15"
    assert doc.tracking_code == "99887766"


def test_date_2digit_year_normalization():
    extractor = RuleBasedExtractor()
    text = "تاریخ: 03/05/20 مبلغ: 100,000 ریال"
    doc = extractor.extract(text)
    assert doc.issue_date == "1403/05/20"


def test_validator_clean_document(sample_pos_receipt_text):
    extractor = RuleBasedExtractor()
    doc = extractor.extract(sample_pos_receipt_text)

    validator = DocumentValidator()
    validated = validator.validate(doc)

    assert validated.needs_review is False
    assert len(validated.review_reasons) == 0


def test_validator_missing_amount_triggers_review():
    doc = ExtractedDocument(
        document_type="receipt",
        total_amount=None,
        currency=CurrencyUnit.UNKNOWN,
    )
    validator = DocumentValidator()
    validated = validator.validate(doc)

    assert validated.needs_review is True
    assert any("Total payable amount" in r for r in validated.review_reasons)


def test_validator_line_items_sum_mismatch():
    doc = ExtractedDocument(
        document_type="invoice",
        total_amount=100000,
        currency=CurrencyUnit.IRT,
        items=[
            FinancialItem(description="Item 1", total_price=30000),
            FinancialItem(description="Item 2", total_price=40000),
        ],  # Sum = 70,000 != 100,000
    )
    validator = DocumentValidator()
    validated = validator.validate(doc)

    assert validated.needs_review is True
    assert any("does not equal total amount" in r for r in validated.review_reasons)


def test_validator_toman_rial_discrepancy():
    # Items sum to 100,000 Toman, but Total is listed as 1,000,000 Rial
    doc = ExtractedDocument(
        document_type="invoice",
        total_amount=1000000,
        currency=CurrencyUnit.IRR,
        items=[
            FinancialItem(description="Item A", total_price=60000),
            FinancialItem(description="Item B", total_price=40000),
        ],
    )
    validator = DocumentValidator()
    validated = validator.validate(doc)

    assert validated.needs_review is True
    assert any("Toman while Total" in r for r in validated.review_reasons)


def test_validator_invalid_date():
    doc = ExtractedDocument(
        document_type="receipt",
        total_amount=50000,
        currency=CurrencyUnit.IRT,
        issue_date="1403/15/45",  # Invalid month 15 and day 45
    )
    validator = DocumentValidator()
    validated = validator.validate(doc)

    assert validated.needs_review is True
    assert any("calendar range" in r for r in validated.review_reasons)


def test_luhn_card_checksum():
    # Valid test card (Luhn compliant)
    assert luhn_checksum_valid("4532015112830366") is True
    # Invalid card
    assert luhn_checksum_valid("4532015112830367") is False
