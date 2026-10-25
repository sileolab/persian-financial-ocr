"""Batch processing pipeline with multi-threading concurrency, fault tolerance, and pause/resume."""

import gc
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from hesabyar.core.config import get_settings
from hesabyar.core.exceptions import HesabyarError
from hesabyar.core.logger import get_logger
from hesabyar.db.state_manager import StateManager, get_state_manager
from hesabyar.extraction import InformationExtractor
from hesabyar.models.schemas import ExtractedDocument, FileRecord, FileStatus
from hesabyar.ocr.engine import BaseOCREngine, get_ocr_engine
from hesabyar.processing.preprocessor import DocumentPreprocessor, PreprocessingConfig

logger = get_logger("pipeline.batch_manager")


@dataclass
class PipelineResult:
    """Outcome of processing an individual document through the full pipeline."""
    file_record: FileRecord
    extracted_doc: Optional[ExtractedDocument] = None
    status: FileStatus = FileStatus.PENDING
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    was_skipped: bool = False


@dataclass
class BatchSummary:
    """Aggregated outcome of a batch execution run."""
    total_files: int = 0
    processed_count: int = 0
    skipped_count: int = 0
    completed_count: int = 0
    needs_review_count: int = 0
    failed_count: int = 0
    elapsed_seconds: float = 0.0
    is_paused: bool = False
    results: List[PipelineResult] = field(default_factory=list)


class BatchPipelineManager:
    """
    Industrial pipeline orchestrator for large-scale financial document processing.

    Key Features:
    - Fault Isolation: Crashes in individual documents are isolated and marked FAILED.
    - Idempotency & SHA-256 Hashing: Files already COMPLETED/NEEDS_REVIEW are skipped.
    - Pause / Resume: State is persisted in SQLite WAL after each document;
      halting the process and re-executing resumes exactly from the next pending item.
    - Thread-safe resource reuse.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        ocr_engine: Optional[BaseOCREngine] = None,
        preprocessor: Optional[DocumentPreprocessor] = None,
        extractor: Optional[InformationExtractor] = None,
    ):
        self.settings = get_settings()
        self.state_mgr = state_manager or get_state_manager()
        self.preprocessor = preprocessor or DocumentPreprocessor()
        self.extractor = extractor or InformationExtractor()
        self._ocr_engine = ocr_engine
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()

    @property
    def ocr_engine(self) -> BaseOCREngine:
        """Lazy access to OCR engine."""
        if self._ocr_engine is None:
            self._ocr_engine = get_ocr_engine()
        return self._ocr_engine

    def request_stop(self) -> None:
        """Signal workers to stop gracefully after finishing active tasks."""
        self._stop_requested.set()
        logger.info("Graceful pipeline pause/stop requested.")

    def is_stop_requested(self) -> bool:
        """Check if pipeline pause has been signaled."""
        return self._stop_requested.is_set()

    def process_single_file(
        self, file_path: Union[str, Path], force_reprocess: bool = False
    ) -> PipelineResult:
        """
        Execute end-to-end pipeline on one file:
        1. Hash & Register
        2. Preprocessing & Clean Cache
        3. OCR & Confidence
        4. Information Extraction & Financial Validation
        5. SQLite WAL Status Update
        """
        start_time = time.perf_counter()
        path = Path(file_path).resolve()

        if not path.is_file():
            err_msg = f"File does not exist: {path}"
            logger.error(err_msg)
            return PipelineResult(
                file_record=FileRecord(
                    file_hash="",
                    file_path=str(path),
                    file_name=path.name,
                    file_size=0,
                    status=FileStatus.FAILED,
                    error_message=err_msg,
                ),
                status=FileStatus.FAILED,
                error=err_msg,
            )

        # 1. Register file in database
        record, is_new = self.state_mgr.register_file(path)

        # 2. Check if already processed (Idempotency)
        if not force_reprocess and not is_new:
            if record.status in (FileStatus.COMPLETED, FileStatus.NEEDS_REVIEW):
                logger.debug(f"Skipping already processed file: {record.file_name}")
                return PipelineResult(
                    file_record=record,
                    status=record.status,
                    was_skipped=True,
                )

        # 3. Fault-Isolated Processing
        try:
            # Step A: Preprocessing
            self.state_mgr.update_status(record.file_hash, FileStatus.PREPROCESSING)
            cache_file = self.settings.cache_dir / f"{path.stem}_{record.file_hash[:8]}_clean.png"
            pre_res = self.preprocessor.preprocess_file(
                input_path=path,
                output_path=cache_file,
                file_hash=record.file_hash,
            )

            # Step B: OCR Text Recognition
            self.state_mgr.update_status(record.file_hash, FileStatus.OCR_PROCESSING)
            ocr_res = self.ocr_engine.extract_text(pre_res.cached_path)

            # Step C: Information Extraction & Financial Validation
            self.state_mgr.update_status(record.file_hash, FileStatus.EXTRACTING)
            extracted_doc = self.extractor.extract_and_validate(ocr_res.normalized_text)

            # Step D: Final Lifecycle State
            final_status = FileStatus.NEEDS_REVIEW if extracted_doc.needs_review else FileStatus.COMPLETED
            reasons_str = "; ".join(extracted_doc.review_reasons) if extracted_doc.review_reasons else None

            self.state_mgr.update_status(
                file_hash=record.file_hash,
                status=final_status,
                ocr_confidence=ocr_res.average_confidence,
                extracted_data_json=extracted_doc.model_dump_json(),
                error_message=reasons_str,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            updated_record = self.state_mgr.get_record(record.file_hash) or record

            logger.info(
                f"Successfully processed {path.name} in {elapsed_ms:.1f}ms -> {final_status.value}"
            )

            return PipelineResult(
                file_record=updated_record,
                extracted_doc=extracted_doc,
                status=final_status,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e)
            logger.error(f"Failed processing {path.name}: {err_msg}", exc_info=True)

            self.state_mgr.update_status(
                file_hash=record.file_hash,
                status=FileStatus.FAILED,
                error_message=err_msg,
            )
            failed_record = self.state_mgr.get_record(record.file_hash) or record

            return PipelineResult(
                file_record=failed_record,
                status=FileStatus.FAILED,
                processing_time_ms=elapsed_ms,
                error=err_msg,
            )

    def run_batch(
        self,
        files: Optional[List[Path]] = None,
        max_workers: Optional[int] = None,
        max_items: Optional[int] = None,
        progress_callback: Optional[Callable[[PipelineResult, int, int], None]] = None,
    ) -> BatchSummary:
        """
        Execute parallel batch processing across files with fault tolerance and pause support.
        """
        start_time = time.perf_counter()
        self._stop_requested.clear()
        workers = max_workers or self.settings.max_workers

        # Discover or load target files
        if files is None:
            self.state_mgr.scan_and_sync_input_dir()
            # Fetch pending or failed files
            pending_records = self.state_mgr.get_pending_records()
            file_paths = [Path(r.file_path) for r in pending_records if Path(r.file_path).is_file()]
        else:
            file_paths = [Path(f).resolve() for f in files if Path(f).is_file()]

        if max_items is not None and max_items > 0:
            file_paths = file_paths[:max_items]

        summary = BatchSummary(total_files=len(file_paths))
        if not file_paths:
            summary.elapsed_seconds = time.perf_counter() - start_time
            return summary

        logger.info(
            f"Starting batch run of {len(file_paths)} files with {workers} parallel workers."
        )

        completed_counter = 0

        # Execute processing pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {}
            for p in file_paths:
                if self._stop_requested.is_set():
                    summary.is_paused = True
                    break
                fut = executor.submit(self.process_single_file, p)
                future_to_path[fut] = p

            for future in as_completed(future_to_path):
                completed_counter += 1
                try:
                    res = future.result()
                    summary.results.append(res)

                    if res.was_skipped:
                        summary.skipped_count += 1
                    else:
                        summary.processed_count += 1
                        if res.status == FileStatus.COMPLETED:
                            summary.completed_count += 1
                        elif res.status == FileStatus.NEEDS_REVIEW:
                            summary.needs_review_count += 1
                        elif res.status == FileStatus.FAILED:
                            summary.failed_count += 1

                    if progress_callback:
                        progress_callback(res, completed_counter, len(future_to_path))

                except Exception as exc:
                    logger.error(f"Unexpected worker exception: {exc}")
                    summary.failed_count += 1

                # Periodically trigger memory cleanup in large batches
                if completed_counter % 25 == 0:
                    gc.collect()

                if self._stop_requested.is_set():
                    summary.is_paused = True
                    break

        summary.elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            f"Batch run finished in {summary.elapsed_seconds:.2f}s: "
            f"{summary.completed_count} completed, {summary.needs_review_count} needs review, "
            f"{summary.failed_count} failed, {summary.skipped_count} skipped."
        )
        return summary
