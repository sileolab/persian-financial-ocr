"""OCR recognition and post-processing modules."""

from hesabyar.ocr.engine import (
    BaseOCREngine,
    OCRResult,
    OCRTextBox,
    PaddleOCREngine,
    TesseractOCREngine,
    get_ocr_engine,
)
from hesabyar.ocr.post_processor import (
    OCRPostProcessor,
    clean_ocr_text,
    normalize_digits,
    normalize_persian_letters,
    sort_reading_order,
)

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "OCRTextBox",
    "PaddleOCREngine",
    "TesseractOCREngine",
    "get_ocr_engine",
    "OCRPostProcessor",
    "clean_ocr_text",
    "normalize_digits",
    "normalize_persian_letters",
    "sort_reading_order",
]
