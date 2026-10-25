"""SQLite-based persistent state manager for tracking document processing lifecycle."""

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from hesabyar.core.config import get_settings
from hesabyar.core.exceptions import DatabaseError
from hesabyar.core.logger import get_logger
from hesabyar.models.schemas import FileRecord, FileStatus, ProcessingStats

logger = get_logger("db.state_manager")


class StateManager:
    """Manages document state lifecycle with SQLite in Write-Ahead-Logging (WAL) mode."""

    def __init__(self, db_path: Optional[Path] = None):
        settings = get_settings()
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a contextual SQLite connection with WAL mode and error handling."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=15.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode for explicit transactions
        )
        conn.row_factory = sqlite3.Row
        try:
            # Enable WAL mode and optimization pragmas
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            yield conn
        except sqlite3.Error as e:
            logger.error(f"SQLite error occurred: {e}")
            raise DatabaseError(f"Database error: {e}") from e
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    ocr_confidence REAL,
                    extracted_data_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_status ON files (status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_files_name ON files (file_name);
                """
            )
        logger.debug(f"State database initialized at {self.db_path}")

    @staticmethod
    def compute_file_hash(file_path: Path, block_size: int = 65536) -> str:
        """Compute SHA-256 hash of a file efficiently using chunked reads."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def register_file(self, file_path: Path) -> Tuple[FileRecord, bool]:
        """
        Register a file into the database if not present.
        Returns a tuple of (FileRecord, is_new).
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = self.compute_file_hash(path)
        file_size = path.stat().st_size
        file_name = path.name

        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE file_hash = ?", (file_hash,)
            ).fetchone()

            if row:
                return self._row_to_record(row), False

            now = datetime.utcnow().isoformat()
            conn.execute(
                """
                INSERT INTO files (
                    file_hash, file_path, file_name, file_size,
                    status, error_message, ocr_confidence, extracted_data_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    file_hash,
                    str(path),
                    file_name,
                    file_size,
                    FileStatus.PENDING.value,
                    now,
                    now,
                ),
            )
            logger.info(f"Registered new file: {file_name} (hash: {file_hash[:8]}...)")

            new_row = conn.execute(
                "SELECT * FROM files WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return self._row_to_record(new_row), True

    def update_status(
        self,
        file_hash: str,
        status: FileStatus,
        error_message: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        extracted_data_json: Optional[str] = None,
    ) -> None:
        """Update the processing status and results of a file."""
        now = datetime.utcnow().isoformat()
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE files
                SET status = ?,
                    error_message = COALESCE(?, error_message),
                    ocr_confidence = COALESCE(?, ocr_confidence),
                    extracted_data_json = COALESCE(?, extracted_data_json),
                    updated_at = ?
                WHERE file_hash = ?
                """,
                (
                    status.value,
                    error_message,
                    ocr_confidence,
                    extracted_data_json,
                    now,
                    file_hash,
                ),
            )
            logger.debug(f"Updated status for {file_hash[:8]} -> {status.value}")

    def get_record(self, file_hash: str) -> Optional[FileRecord]:
        """Fetch a single record by file hash."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM files WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            if row:
                return self._row_to_record(row)
            return None

    def get_records_by_status(
        self, status: FileStatus, limit: Optional[int] = None
    ) -> List[FileRecord]:
        """Fetch records matching a specific status."""
        query = "SELECT * FROM files WHERE status = ? ORDER BY created_at ASC"
        params: list = [status.value]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_pending_records(self, limit: Optional[int] = None) -> List[FileRecord]:
        """Fetch pending records waiting for processing."""
        return self.get_records_by_status(FileStatus.PENDING, limit=limit)

    def get_all_records(self) -> List[FileRecord]:
        """Retrieve all tracked file records."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM files ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def get_stats(self) -> ProcessingStats:
        """Calculate counts of records grouped by status."""
        stats = ProcessingStats()
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM files GROUP BY status"
            ).fetchall()
            total = 0
            for r in rows:
                status_str = r["status"]
                count = r["count"]
                total += count
                if status_str == FileStatus.PENDING.value:
                    stats.pending = count
                elif status_str == FileStatus.PREPROCESSING.value:
                    stats.preprocessing = count
                elif status_str == FileStatus.OCR_PROCESSING.value:
                    stats.ocr_processing = count
                elif status_str == FileStatus.EXTRACTING.value:
                    stats.extracting = count
                elif status_str == FileStatus.COMPLETED.value:
                    stats.completed = count
                elif status_str == FileStatus.NEEDS_REVIEW.value:
                    stats.needs_review = count
                elif status_str == FileStatus.FAILED.value:
                    stats.failed = count
            stats.total_files = total
        return stats

    def reset_status(self, target: str = "failed") -> int:
        """
        Reset records back to PENDING.
        target='failed': resets only FAILED records.
        target='all': resets everything except COMPLETED.
        """
        target = target.lower()
        now = datetime.utcnow().isoformat()
        with self._lock, self._get_connection() as conn:
            if target == "failed":
                cursor = conn.execute(
                    """
                    UPDATE files
                    SET status = ?, error_message = NULL, updated_at = ?
                    WHERE status = ?
                    """,
                    (FileStatus.PENDING.value, now, FileStatus.FAILED.value),
                )
            elif target == "all":
                cursor = conn.execute(
                    """
                    UPDATE files
                    SET status = ?, error_message = NULL, updated_at = ?
                    """,
                    (FileStatus.PENDING.value, now),
                )
            else:
                raise ValueError(f"Unknown reset target: {target}. Use 'failed' or 'all'.")
            affected = cursor.rowcount
            logger.info(f"Reset {affected} records ({target}) back to PENDING")
            return affected

    def clean_missing_files(self) -> int:
        """Remove database records whose original files have been deleted from disk."""
        with self._lock, self._get_connection() as conn:
            rows = conn.execute("SELECT file_hash, file_path FROM files").fetchall()
            removed = 0
            for r in rows:
                if not Path(r["file_path"]).exists():
                    conn.execute("DELETE FROM files WHERE file_hash = ?", (r["file_hash"],))
                    removed += 1
            if removed > 0:
                logger.info(f"Cleaned {removed} deleted files from database")
            return removed

    def scan_and_sync_input_dir(
        self,
        input_dir: Optional[Path] = None,
        valid_extensions: Tuple[str, ...] = (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".pdf",
        ),
        purge_missing: bool = True,
    ) -> Tuple[int, int]:
        """
        Scan input directory for documents, registering new files into the database.
        Returns: (new_files_count, total_scanned_count)
        """
        if purge_missing:
            self.clean_missing_files()

        settings = get_settings()
        target_dir = Path(input_dir or settings.input_dir)
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            return 0, 0

        scanned = 0
        new_registered = 0
        for item in target_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in valid_extensions:
                scanned += 1
                _, is_new = self.register_file(item)
                if is_new:
                    new_registered += 1

        logger.info(
            f"Input directory sync completed: {new_registered} new / {scanned} scanned"
        )
        return new_registered, scanned

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> FileRecord:
        """Helper to convert SQLite Row to FileRecord model."""
        created_val = row["created_at"]
        updated_val = row["updated_at"]

        created_dt = None
        updated_dt = None
        if created_val:
            try:
                created_dt = datetime.fromisoformat(created_val)
            except ValueError:
                pass
        if updated_val:
            try:
                updated_dt = datetime.fromisoformat(updated_val)
            except ValueError:
                pass

        return FileRecord(
            file_hash=row["file_hash"],
            file_path=row["file_path"],
            file_name=row["file_name"],
            file_size=row["file_size"],
            status=FileStatus(row["status"]),
            error_message=row["error_message"],
            ocr_confidence=row["ocr_confidence"],
            extracted_data_json=row["extracted_data_json"],
            created_at=created_dt,
            updated_at=updated_dt,
        )


_state_manager_instance: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Singleton getter for StateManager."""
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = StateManager()
    return _state_manager_instance
