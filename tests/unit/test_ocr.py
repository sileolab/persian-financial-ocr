"""Unit tests for OCR post-processing, character/digit normalization, and layout sorting."""

import pytest

from hesabyar.ocr.post_processor import (
    OCRPostProcessor,
    clean_ocr_text,
    normalize_digits,
    normalize_persian_letters,
    normalize_spacing,
    remove_diacritics,
    sort_reading_order,
)


def test_normalize_digits_persian_to_english():
    text = "مبلغ: ۱,۵۰۰,۰۰۰ ریال | کد: ۰۹۱۲۳۴۵۶۷۸"
    normalized = normalize_digits(text, to_english=True)
    assert "1,500,000" in normalized
    assert "0912345678" in normalized


def test_normalize_digits_arabic_to_english():
    text = "المبلغ: ١٥٠٠٠٠٠ ريال"
    normalized = normalize_digits(text, to_english=True)
    assert "1500000" in normalized


def test_normalize_digits_to_persian():
    text = "Amount: 250000 Tomans"
    normalized = normalize_digits(text, to_english=False)
    assert "۲۵۰۰۰۰" in normalized


def test_normalize_persian_letters():
    # Arabic Yeh and Kaf
    arabic_text = "بانك ملي ايران - فايبر"
    persian_text = normalize_persian_letters(arabic_text)
    assert "بانک ملی ایران" in persian_text
    assert "ی" in persian_text and "ک" in persian_text


def test_remove_diacritics():
    text_with_tashkeel = "شَرِکَتِ کالا و خَدَماتِ فـروشـگاهـی"
    cleaned = remove_diacritics(text_with_tashkeel)
    assert "شرکت کالا و خدمات فروشگاهی" in cleaned


def test_normalize_spacing_and_zwnj():
    text = "این فاکتور پرداخت می شود و رسید ها صادر گردید."
    normalized = normalize_spacing(text)
    # Check half-space in می‎شود and ها
    assert "می\u200cشود" in normalized
    assert "رسید\u200cها" in normalized


def test_clean_ocr_text_pipeline():
    raw_ocr = "فـاکـتـور شُـماره: ١٤٠٣/٠٨/٠٤   مَـبـلـغ: ۲۵۰,۰۰۰ ريـال  "
    cleaned = clean_ocr_text(raw_ocr, digits_to_english=True)
    assert "فاکتور شماره: 1403/08/04" in cleaned
    assert "مبلغ: 250,000 ریال" in cleaned


def test_sort_reading_order_rtl_and_vertical():
    # Line 1: Header (Single box)
    # Line 2: Two boxes on the same horizontal band (y ~ 100):
    # Box A: Left side (x=50), Box B: Right side (x=300)
    # In Persian reading order (RTL), Box B should come before Box A!
    boxes = [
        {"box": [[100, 20], [300, 20], [300, 50], [100, 50]], "text": "تیتر اصلی", "confidence": 0.99},
        {"box": [[50, 100], [150, 100], [150, 130], [50, 130]], "text": "مبلغ (سمت چپ)", "confidence": 0.95},
        {"box": [[300, 105], [400, 105], [400, 135], [300, 135]], "text": "عنوان (سمت راست)", "confidence": 0.98},
    ]

    sorted_boxes = sort_reading_order(boxes, y_threshold=15)
    assert len(sorted_boxes) == 3
    assert sorted_boxes[0]["text"] == "تیتر اصلی"
    # In same line, RTL order: Right box first, Left box second
    assert sorted_boxes[1]["text"] == "عنوان (سمت راست)"
    assert sorted_boxes[2]["text"] == "مبلغ (سمت چپ)"


def test_post_processor_full():
    processor = OCRPostProcessor(digits_to_english=True)
    raw_boxes = [
        {"box": [[200, 50], [400, 50], [400, 80], [200, 80]], "text": "رسـيـد پـرداخـت", "confidence": 0.96},
        {"box": [[100, 120], [200, 120], [200, 150], [100, 150]], "text": "۵۰,۰۰۰ تومان", "confidence": 0.92},
    ]
    sorted_lines, full_text = processor.process(raw_boxes)
    assert len(sorted_lines) == 2
    assert "رسید پرداخت" in full_text
    assert "50,000 تومان" in full_text
