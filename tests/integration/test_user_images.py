"""Integration test validating real user test images against the production pipeline."""

import pytest
from pathlib import Path

from hesabyar.core.config import get_settings
from hesabyar.extraction.rules import RuleBasedExtractor
from hesabyar.extraction.validator import DocumentValidator
from hesabyar.models.schemas import CurrencyUnit, FileStatus
from hesabyar.ocr.engine import EasyOCREngine
from hesabyar.pipeline import BatchPipelineManager


@pytest.fixture(scope="module")
def ocr_engine():
    return EasyOCREngine(use_gpu=True)


@pytest.fixture(scope="module")
def extractor():
    return RuleBasedExtractor()


@pytest.fixture(scope="module")
def validator():
    return DocumentValidator()


def test_user_images_extraction_accuracy(ocr_engine, extractor, validator):
    settings = get_settings()
    input_dir = settings.input_dir

    expected_specs = {
        "meli-1.jpg": {
            "amount": 789_000,
            "currency": CurrencyUnit.IRR,
            "date": "1403/08/02",
            "type": "bill_payment",
            "has_tracking": True,
        },
        "meli-2.jpg": {
            "amount": 783_000,
            "currency": CurrencyUnit.IRR,
            "date": "1403/08/02",
            "type": "bill_payment",
            "has_tracking": True,
        },
        "snap-1.jpg": {
            "min_amount": 2_000_000,
            "max_amount": 2_500_000,
            "currency": CurrencyUnit.IRR,
            "type": "trip_receipt",
        },
        "tegarat-1.jpg": {
            "amount": 121_758_000,
            "currency": CurrencyUnit.IRR,
            "date": "1403/07/29",
            "type": "card_transfer",
            "tracking": "629415478371",
        },
    }

    for filename, specs in expected_specs.items():
        file_path = input_dir / filename
        if not file_path.is_file():
            pytest.skip(f"Test image {filename} not present in input directory")

        ocr_result = ocr_engine.extract_text(file_path)
        assert ocr_result.average_confidence > 0.40
        assert len(ocr_result.lines) > 0

        doc = extractor.extract(ocr_result.normalized_text)
        validated_doc = validator.validate(doc)

        # Critical user requirement: Needs Review MUST be False
        assert validated_doc.needs_review is False, (
            f"{filename} was flagged for review: {validated_doc.review_reasons}"
        )

        if "amount" in specs:
            assert validated_doc.total_amount == specs["amount"], (
                f"{filename} expected amount {specs['amount']}, got {validated_doc.total_amount}"
            )
        elif "min_amount" in specs and "max_amount" in specs:
            assert specs["min_amount"] <= validated_doc.total_amount <= specs["max_amount"]

        if "currency" in specs:
            assert validated_doc.currency == specs["currency"]

        if "date" in specs:
            assert validated_doc.issue_date == specs["date"]

        if "type" in specs:
            assert validated_doc.document_type == specs["type"]

        if "tracking" in specs:
            assert validated_doc.tracking_code == specs["tracking"]
