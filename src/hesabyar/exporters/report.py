"""JSON summary reporting and financial aggregation metrics."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

from hesabyar.core.exceptions import ExportError
from hesabyar.core.logger import get_logger
from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument, FileRecord, FileStatus

logger = get_logger("exporters.report")


def generate_summary_report(records: List[FileRecord]) -> Dict[str, Any]:
    """Calculate aggregate financial statistics, counts, and breakdown metrics."""
    total_files = len(records)
    completed_count = sum(1 for r in records if r.status == FileStatus.COMPLETED)
    needs_review_count = sum(1 for r in records if r.status == FileStatus.NEEDS_REVIEW)
    failed_count = sum(1 for r in records if r.status == FileStatus.FAILED)

    total_amount_irt = 0
    total_amount_irr = 0
    vendor_counts: Dict[str, int] = {}
    doc_type_counts: Dict[str, int] = {}

    confidence_scores: List[float] = []

    parsed_docs: List[Dict[str, Any]] = []

    for r in records:
        if r.ocr_confidence is not None:
            confidence_scores.append(r.ocr_confidence)

        doc_dict = None
        if r.extracted_data_json:
            try:
                edoc = ExtractedDocument.model_validate_json(r.extracted_data_json)
                doc_dict = edoc.model_dump()

                # Aggregate amounts only from verified completed receipts
                if r.status == FileStatus.COMPLETED and edoc.total_amount:
                    if edoc.currency == CurrencyUnit.IRT:
                        total_amount_irt += edoc.total_amount
                    elif edoc.currency == CurrencyUnit.IRR:
                        total_amount_irr += edoc.total_amount

                v_name = edoc.vendor_name or "نامشخص (Unknown)"
                vendor_counts[v_name] = vendor_counts.get(v_name, 0) + 1

                dtype = edoc.document_type or "receipt"
                doc_type_counts[dtype] = doc_type_counts.get(dtype, 0) + 1

            except Exception:
                pass

        parsed_docs.append({
            "file_name": r.file_name,
            "file_hash": r.file_hash,
            "status": r.status.value,
            "ocr_confidence": r.ocr_confidence,
            "error_message": r.error_message,
            "extracted_data": doc_dict,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        })

    avg_conf = (sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 0.0
    success_rate = (completed_count / total_files * 100.0) if total_files > 0 else 0.0

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "Hesabyar Production Pipeline",
            "version": "0.1.0",
        },
        "kpis": {
            "total_files": total_files,
            "completed_verified": completed_count,
            "needs_review": needs_review_count,
            "failed": failed_count,
            "success_rate_percent": round(success_rate, 2),
            "average_ocr_confidence": round(avg_conf, 4),
        },
        "financial_totals": {
            "total_toman_irt": total_amount_irt,
            "total_rial_irr": total_amount_irr,
            "equivalent_total_toman": total_amount_irt + (total_amount_irr // 10),
        },
        "breakdown": {
            "by_document_type": doc_type_counts,
            "by_vendor": vendor_counts,
        },
        "documents": parsed_docs,
    }


def export_to_json(
    records: List[FileRecord], output_path: Union[str, Path], indent: int = 2
) -> Path:
    """
    Export comprehensive financial report and parsed document data to a formatted JSON file.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    report_data = generate_summary_report(records)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=indent)

        logger.info(f"JSON summary report saved: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed exporting to JSON: {e}")
        raise ExportError(f"JSON export error: {e}") from e
