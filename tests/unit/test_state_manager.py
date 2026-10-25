"""Unit tests for the SQLite State Manager."""

import tempfile
from pathlib import Path
import pytest

from hesabyar.db.state_manager import StateManager
from hesabyar.models.schemas import FileStatus


@pytest.fixture
def temp_db_and_files(tmp_path: Path):
    """Fixture providing a temporary DB and sample files."""
    db_file = tmp_path / "test_state.db"
    mgr = StateManager(db_path=db_file)

    sample_file1 = tmp_path / "invoice1.jpg"
    sample_file1.write_bytes(b"dummy image data 1")

    sample_file2 = tmp_path / "receipt2.png"
    sample_file2.write_bytes(b"dummy image data 2")

    return mgr, sample_file1, sample_file2, tmp_path


def test_db_initialization(temp_db_and_files):
    mgr, _, _, _ = temp_db_and_files
    assert mgr.db_path.exists()
    stats = mgr.get_stats()
    assert stats.total_files == 0


def test_file_registration_and_idempotency(temp_db_and_files):
    mgr, file1, _, _ = temp_db_and_files

    # First registration
    record1, is_new1 = mgr.register_file(file1)
    assert is_new1 is True
    assert record1.file_name == "invoice1.jpg"
    assert record1.status == FileStatus.PENDING

    # Second registration (same file) -> Should be idempotent
    record2, is_new2 = mgr.register_file(file1)
    assert is_new2 is False
    assert record2.file_hash == record1.file_hash
    assert record2.file_name == record1.file_name

    # Stats should only show 1 total file
    stats = mgr.get_stats()
    assert stats.total_files == 1
    assert stats.pending == 1


def test_status_updates(temp_db_and_files):
    mgr, file1, _, _ = temp_db_and_files
    record, _ = mgr.register_file(file1)

    # Move to OCR_PROCESSING
    mgr.update_status(record.file_hash, FileStatus.OCR_PROCESSING)
    updated = mgr.get_record(record.file_hash)
    assert updated.status == FileStatus.OCR_PROCESSING

    # Move to COMPLETED with confidence and extracted data
    mgr.update_status(
        record.file_hash,
        FileStatus.COMPLETED,
        ocr_confidence=0.98,
        extracted_data_json='{"total_amount": 500000}',
    )
    completed_rec = mgr.get_record(record.file_hash)
    assert completed_rec.status == FileStatus.COMPLETED
    assert completed_rec.ocr_confidence == 0.98
    assert '{"total_amount": 500000}' in completed_rec.extracted_data_json


def test_failed_status_and_reset(temp_db_and_files):
    mgr, file1, file2, _ = temp_db_and_files
    rec1, _ = mgr.register_file(file1)
    rec2, _ = mgr.register_file(file2)

    mgr.update_status(rec1.file_hash, FileStatus.FAILED, error_message="Image corrupted")

    stats = mgr.get_stats()
    assert stats.failed == 1
    assert stats.pending == 1

    # Reset failed records back to pending
    affected = mgr.reset_status(target="failed")
    assert affected == 1

    stats_after = mgr.get_stats()
    assert stats_after.failed == 0
    assert stats_after.pending == 2


def test_scan_and_sync_input_dir(temp_db_and_files):
    mgr, _, _, tmp_path = temp_db_and_files
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    # Create dummy images and non-image files
    (input_dir / "doc1.jpg").write_bytes(b"content 1")
    (input_dir / "doc2.png").write_bytes(b"content 2")
    (input_dir / "notes.txt").write_bytes(b"text file should be ignored")

    new_count, total_scanned = mgr.scan_and_sync_input_dir(input_dir)
    assert new_count == 2
    assert total_scanned == 2

    # Run again without changes -> 0 new
    new_count2, total_scanned2 = mgr.scan_and_sync_input_dir(input_dir)
    assert new_count2 == 0
    assert total_scanned2 == 2
