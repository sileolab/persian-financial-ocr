"""End-to-end integration tests for the full Hesabyar pipeline."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from hesabyar.db.state_manager import StateManager
from hesabyar.exporters import export_to_csv, export_to_excel, export_to_json, export_to_word
from hesabyar.extraction import InformationExtractor
from hesabyar.models.schemas import CurrencyUnit, FileStatus
from hesabyar.ocr.engine import BaseOCREngine, OCRResult, OCRTextBox
from hesabyar.processing.preprocessor import DocumentPreprocessor


class MockPipelineOCREngine(BaseOCREngine):
    """Fast deterministic engine for CI integration tests."""

    def extract_text(self, image) -> OCRResult:
        text = "فروشگاه مرکزی حسابیار\nتاریخ: 1403/08/04\nمبلغ کل: 850,000 تومان\nکد پیگیری: 76543210\nشماره کارت: 6037-9971-****-4321"
        lines = [
            OCRTextBox(text=line, confidence=0.97, box=[[10, i * 40], [300, i * 40], [300, i * 40 + 30], [10, i * 40 + 30]])
            for i, line in enumerate(text.split("\n"))
        ]
        return OCRResult(
            raw_text=text,
            normalized_text=text,
            lines=lines,
            average_confidence=0.97,
            engine_name="MockPipelineOCR",
            elapsed_time_ms=10.0,
        )


def test_full_pipeline_e2e(tmp_path: Path):
    # Setup directories
    input_file = tmp_path / "receipt_e2e.jpg"
    cache_file = tmp_path / "cache" / "receipt_e2e_clean.png"
    output_dir = tmp_path / "output"
    db_file = tmp_path / "state.db"

    # Step 1: Create synthetic input image
    img = np.ones((600, 450, 3), dtype=np.uint8) * 240
    cv2.putText(img, "HESABYAR STORE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.putText(img, "Total: 850,000 Tomans", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
    cv2.putText(img, "Date: 1403/08/04", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    cv2.imwrite(str(input_file), img)

    # Step 2: Register in SQLite
    state_mgr = StateManager(db_path=db_file)
    record, is_new = state_mgr.register_file(input_file)
    assert is_new is True
    assert record.status == FileStatus.PENDING

    # Step 3: Preprocessing
    preprocessor = DocumentPreprocessor()
    pre_res = preprocessor.preprocess_file(input_file, cache_file)
    assert pre_res.cached_path.exists()
    state_mgr.update_status(record.file_hash, FileStatus.PREPROCESSING)

    # Step 4: OCR Extraction
    ocr_engine = MockPipelineOCREngine()
    ocr_res = ocr_engine.extract_text(pre_res.cached_path)
    assert ocr_res.average_confidence >= 0.90
    state_mgr.update_status(record.file_hash, FileStatus.OCR_PROCESSING)

    # Step 5: Information Extraction & Financial Validation
    extractor = InformationExtractor()
    extracted_doc = extractor.extract_and_validate(ocr_res.normalized_text)
    assert extracted_doc.total_amount == 850000
    assert extracted_doc.currency == CurrencyUnit.IRT
    assert extracted_doc.issue_date == "1403/08/04"
    assert extracted_doc.tracking_code == "76543210"
    assert extracted_doc.needs_review is False

    # Step 6: Update final DB status to COMPLETED
    state_mgr.update_status(
        record.file_hash,
        FileStatus.COMPLETED,
        ocr_confidence=ocr_res.average_confidence,
        extracted_data_json=extracted_doc.model_dump_json(),
    )
    final_record = state_mgr.get_record(record.file_hash)
    assert final_record.status == FileStatus.COMPLETED

    # Step 7: Export all multi-format outputs
    records = state_mgr.get_all_records()
    assert len(records) == 1

    excel_p = export_to_excel(records, output_dir / "report.xlsx")
    word_p = export_to_word(records, output_dir / "report.docx")
    csv_p = export_to_csv(records, output_dir / "report.csv")
    json_p = export_to_json(records, output_dir / "report.json")

    assert excel_p.exists() and excel_p.stat().st_size > 0
    assert word_p.exists() and word_p.stat().st_size > 0
    assert csv_p.exists() and csv_p.stat().st_size > 0
    assert json_p.exists() and json_p.stat().st_size > 0
