"""Custom domain exceptions for the Hesabyar application."""

from typing import Optional


class HesabyarError(Exception):
    """Base exception for all Hesabyar-related errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(HesabyarError):
    """Raised when configuration or environment variables are invalid."""
    pass


class DatabaseError(HesabyarError):
    """Raised when a database operation fails."""
    pass


class StateTransitionError(DatabaseError):
    """Raised when an invalid state transition is attempted on a file record."""
    pass


class ProcessingError(HesabyarError):
    """Raised when processing a document file fails."""

    def __init__(self, message: str, file_path: Optional[str] = None, details: Optional[str] = None):
        super().__init__(message, details)
        self.file_path = file_path

    def __str__(self) -> str:
        base = super().__str__()
        if self.file_path:
            return f"[{self.file_path}] {base}"
        return base


FileProcessingError = ProcessingError


class ImagePreprocessingError(FileProcessingError):
    """Raised when an image preprocessing step (deskewing, binarization) fails."""
    pass


class OCREngineError(FileProcessingError):
    """Raised when OCR extraction fails."""
    pass


class ExtractionError(FileProcessingError):
    """Raised when rule-based or LLM field extraction fails."""
    pass


class FinancialValidationError(HesabyarError):
    """Raised when financial reconciliation or cross-checks fail."""
    pass


class ExportError(HesabyarError):
    """Raised when exporting processed data to Excel, Word, or JSON fails."""
    pass
