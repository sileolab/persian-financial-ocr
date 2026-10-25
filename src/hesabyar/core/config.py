"""Configuration management for Hesabyar using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_project_root() -> Path:
    """Locate the project root directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = find_project_root()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Environment
    hesabyar_env: str = Field(default="development", description="Environment mode (development/production)")

    # Base Paths
    project_root: Path = Field(default_factory=find_project_root)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    input_dir: Path = Field(default=PROJECT_ROOT / "data" / "input")
    output_dir: Path = Field(default=PROJECT_ROOT / "data" / "output")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    ground_truth_dir: Path = Field(default=PROJECT_ROOT / "data" / "ground_truth")

    # Logging
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    log_file_path: Path = Field(default=PROJECT_ROOT / "data" / "hesabyar.log")
    log_max_bytes: int = Field(default=10 * 1024 * 1024, description="Max size of a log file in bytes (10MB)")
    log_backup_count: int = Field(default=5, description="Number of rotated backup log files")

    # Database
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "hesabyar_state.db")

    # Performance & Concurrency
    max_workers: int = Field(default=4, ge=1, le=64, description="Number of parallel worker processes")
    batch_size: int = Field(default=20, ge=1, description="Batch size for pipeline processing")

    # OCR Settings
    ocr_engine: str = Field(default="easyocr", description="Default OCR engine: easyocr, paddleocr, or tesseract")
    ocr_lang: str = Field(default="fa", description="Language code for OCR (fa covers Persian and Arabic script)")
    use_gpu: bool = Field(default=True, description="Enable CUDA GPU acceleration if available")

    # LLM Settings (Phase 4)
    openai_api_key: Optional[str] = Field(default=None)
    openai_base_url: Optional[str] = Field(default=None)
    gemini_api_key: Optional[str] = Field(default=None)
    llm_model: str = Field(default="gemini-2.5-flash")

    @field_validator(
        "data_dir",
        "input_dir",
        "output_dir",
        "cache_dir",
        "ground_truth_dir",
        "log_file_path",
        "db_path",
        mode="before",
    )
    @classmethod
    def resolve_paths(cls, value: str | Path) -> Path:
        """Convert string paths to absolute Paths relative to project root if not absolute."""
        p = Path(value)
        if not p.is_absolute():
            return PROJECT_ROOT / p
        return p

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
