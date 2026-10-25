"""Pydantic schemas and data models for document processing and financial verification."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileStatus(str, Enum):
    """Processing lifecycle states of a document."""
    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    OCR_PROCESSING = "OCR_PROCESSING"
    EXTRACTING = "EXTRACTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class CurrencyUnit(str, Enum):
    """Supported currency representations."""
    IRR = "IRR"      # ریال
    IRT = "IRT"      # تومان
    UNKNOWN = "UNKNOWN"


class FinancialItem(BaseModel):
    """Individual line item in an invoice or receipt."""
    model_config = ConfigDict(extra="ignore")

    description: str = Field(description="Title or description of the product/service")
    quantity: Optional[float] = Field(default=1.0, description="Quantity or count")
    unit_price: Optional[int] = Field(default=None, description="Unit price in detected currency")
    total_price: Optional[int] = Field(default=None, description="Total price for the item")


class ExtractedDocument(BaseModel):
    """Structured financial data extracted from a document."""
    model_config = ConfigDict(extra="ignore")

    document_type: str = Field(default="receipt", description="receipt, pos_slip, invoice, bank_transfer")
    vendor_name: Optional[str] = Field(default=None, description="Store, merchant, or company name")
    issue_date: Optional[str] = Field(default=None, description="Jalali (Solar) date string e.g. 1403/08/04")
    issue_time: Optional[str] = Field(default=None, description="Time string e.g. 14:32:00")
    total_amount: Optional[int] = Field(default=None, description="Final payable amount")
    currency: CurrencyUnit = Field(default=CurrencyUnit.IRT, description="Currency unit (Toman/Rial)")
    tracking_code: Optional[str] = Field(default=None, description="Reference, tracking, or RRN code")
    card_number: Optional[str] = Field(default=None, description="Masked or full bank card number")
    terminal_id: Optional[str] = Field(default=None, description="POS terminal ID or merchant ID")
    items: List[FinancialItem] = Field(default_factory=list, description="Extracted line items")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    needs_review: bool = Field(default=False, description="Flagged for human review due to ambiguity")
    review_reasons: List[str] = Field(default_factory=list, description="Reasons triggering manual review")


class FileRecord(BaseModel):
    """Database record representation of a tracked document file."""
    model_config = ConfigDict(from_attributes=True)

    file_hash: str = Field(description="SHA-256 hash of the original input file")
    file_path: str = Field(description="Absolute or relative path to the source file")
    file_name: str = Field(description="File name with extension")
    file_size: int = Field(description="File size in bytes")
    status: FileStatus = Field(default=FileStatus.PENDING, description="Current lifecycle state")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    ocr_confidence: Optional[float] = Field(default=None, description="Average OCR confidence score")
    extracted_data_json: Optional[str] = Field(default=None, description="Serialized JSON of extracted document")
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)


class ProcessingStats(BaseModel):
    """Summary of database records by processing state."""
    total_files: int = 0
    pending: int = 0
    preprocessing: int = 0
    ocr_processing: int = 0
    extracting: int = 0
    completed: int = 0
    needs_review: int = 0
    failed: int = 0
