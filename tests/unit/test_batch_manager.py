"""Unit tests for parallel batch pipeline, fault tolerance, and pause/resume functionality."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from hesabyar.db.state_manager import StateManager
from hesabyar.models.schemas import FileStatus, ExtractedDocument, CurrencyUnit
from hesabyar.ocr.engine import BaseOCREngine, OCRResult, OCRTextBox
from hesabyar.pipeline.batch_manager import BatchPipelineManager


class FastMockOCREngine(BaseOCREngine):
    """Fast deterministic OCR engine for batch pipeline unit tests."""

    def extract_text(self, image) -> OCRResult:
        lines = [
            OCRTextBox(
                text="فروشگاه تست حسابیار",
                confidence=0.98,
                box=[[10, 10], [200, 10], [200, 40], [10, 40]],
            ),
            OCRTextBox(
                text="مبلغ کل: 150,000 تومان",
                confidence=0.95,
                box=[[10, 50], [250, 50], [250, 80], [10, 80]],
            ),
            OCRTextBox(
                text="کد پیگیری: 123456",
                confidence=0.99,
                box=[[10, 90], [200, 90], [200, 120], [10, 120]],
            ),
        ]
        text = "\n".join(l.text for l in lines)
        return OCRResult(
            raw_text=text,
            normalized_text=text,
            lines=lines,
            average_confidence=0.97,
            engine_name="MockOCR",
            elapsed_time_ms=5.0,
        )


@pytest.fixture
def batch_setup(tmp_path: Path):
    """Fixture providing a StateManager, Mock OCR Engine, and sample image files."""
    db_file = tmp_path / "test_state.db"
    state_mgr = StateManager(db_path=db_file)
    mock_ocr = FastMockOCREngine()

    manager = BatchPipelineManager(
        state_manager=state_mgr,
        ocr_engine=mock_ocr,
    )

    # Create 4 test image files
    files = []
    for i in range(1, 5):
        img_file = tmp_path / f"receipt_{i}.jpg"
        img = np.ones((300, 400, 3), dtype=np.uint8) * 240
        cv2.putText(img, f"RECEIPT #{i}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(img_file), img)
        files.append(img_file)

    return manager, state_mgr, files, tmp_path


def test_batch_single_file_processing(batch_setup):
    manager, state_mgr, files, _ = batch_setup
    result = manager.process_single_file(files[0])

    assert result.status == FileStatus.COMPLETED
    assert result.extracted_doc is not None
    assert result.extracted_doc.total_amount == 150000
    assert result.extracted_doc.currency == CurrencyUnit.IRT
    assert result.was_skipped is False

    # Check database record
    rec = state_mgr.get_record(result.file_record.file_hash)
    assert rec.status == FileStatus.COMPLETED
    assert rec.ocr_confidence == 0.97


def test_batch_idempotency_skip(batch_setup):
    manager, state_mgr, files, _ = batch_setup

    # First run
    res1 = manager.process_single_file(files[0])
    assert res1.was_skipped is False

    # Second run on same file -> Should be skipped automatically
    res2 = manager.process_single_file(files[0])
    assert res2.was_skipped is True
    assert res2.status == FileStatus.COMPLETED


def test_batch_fault_isolation(batch_setup):
    manager, state_mgr, files, tmp_path = batch_setup

    # Create a corrupted zero-byte image file
    corrupt_file = tmp_path / "corrupt_zero_byte.jpg"
    corrupt_file.write_bytes(b"")

    res_corrupt = manager.process_single_file(corrupt_file)
    assert res_corrupt.status == FileStatus.FAILED
    assert res_corrupt.error is not None

    # Normal file should still process without being affected
    res_normal = manager.process_single_file(files[0])
    assert res_normal.status == FileStatus.COMPLETED


def test_batch_pause_and_resume_simulation(batch_setup):
    """
    Test pausing after N items and resuming remaining items:
    Simulates acceptance criterion: process 2 items, stop, then resume and finish the other 2.
    """
    manager, state_mgr, files, _ = batch_setup

    # Register all 4 files
    for f in files:
        state_mgr.register_file(f)

    stats_initial = state_mgr.get_stats()
    assert stats_initial.pending == 4
    assert stats_initial.completed == 0

    # Phase 1: Process only 2 items (limit=2)
    summary_1 = manager.run_batch(files=files, max_items=2, max_workers=2)
    assert summary_1.processed_count == 2
    assert summary_1.completed_count == 2

    stats_mid = state_mgr.get_stats()
    assert stats_mid.completed == 2
    assert stats_mid.pending == 2

    # Phase 2: Resume remaining items
    summary_2 = manager.run_batch(files=files, max_workers=2)
    # The 2 already completed files are skipped, and the remaining 2 are processed
    assert summary_2.skipped_count == 2
    assert summary_2.completed_count == 2
    assert summary_2.processed_count == 2

    stats_final = state_mgr.get_stats()
    assert stats_final.completed == 4
    assert stats_final.pending == 0
