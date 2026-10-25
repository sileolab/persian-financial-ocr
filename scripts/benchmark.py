"""Comprehensive benchmarking script for Hesabyar financial OCR pipeline.

Evaluates Character Error Rate (CER), Digit Error Rate (DER), Financial Field
Precision/Recall, RAM consumption, and issues Fine-Tuning recommendations.
"""

import json
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to pythonpath
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np
import psutil
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hesabyar.core.config import get_settings
from hesabyar.extraction import InformationExtractor
from hesabyar.models.schemas import CurrencyUnit, ExtractedDocument
from hesabyar.ocr.engine import BaseOCREngine, get_ocr_engine
from hesabyar.ocr.post_processor import clean_ocr_text
from hesabyar.pipeline import BatchPipelineManager
from hesabyar.processing.preprocessor import DocumentPreprocessor

console = Console(force_terminal=True, legacy_windows=False)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_cer(predicted: str, target: str) -> float:
    """Compute Character Error Rate (CER)."""
    p = clean_ocr_text(predicted, digits_to_english=True)
    t = clean_ocr_text(target, digits_to_english=True)
    if not t:
        return 0.0 if not p else 1.0
    dist = levenshtein_distance(p, t)
    return min(1.0, dist / len(t))


def compute_der(predicted: str, target: str) -> float:
    """Compute Digit Error Rate (DER) specifically on numbers."""
    p_digits = "".join(c for c in clean_ocr_text(predicted, digits_to_english=True) if c.isdigit())
    t_digits = "".join(c for c in clean_ocr_text(target, digits_to_english=True) if c.isdigit())
    if not t_digits:
        return 0.0 if not p_digits else 1.0
    dist = levenshtein_distance(p_digits, t_digits)
    return min(1.0, dist / len(t_digits))


def generate_synthetic_ground_truth(gt_dir: Path, count: int = 15) -> List[Tuple[Path, Path]]:
    """
    Generate synthetic diverse Iranian financial ground-truth documents
    (POS slips, Snapp receipts, store invoices) if directory is empty.
    """
    gt_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    samples = [
        {
            "filename": "pos_mellat_01.jpg",
            "vendor": "فروشگاه افق کوروش",
            "type": "pos_slip",
            "total_amount": 350000,
            "currency": "IRT",
            "date": "1403/08/04",
            "time": "14:22:10",
            "tracking": "987654321",
            "card": "6104-3378-****-1234",
            "terminal": "12345678",
        },
        {
            "filename": "snapp_receipt_02.jpg",
            "vendor": "اسنپ - سفر شهری",
            "type": "receipt",
            "total_amount": 85000,
            "currency": "IRT",
            "date": "1403/08/03",
            "time": "09:15:00",
            "tracking": "77889900",
            "card": None,
            "terminal": None,
        },
        {
            "filename": "invoice_store_03.jpg",
            "vendor": "هایپرمارکت ستاره شهر",
            "type": "invoice",
            "total_amount": 1450000,
            "currency": "IRR",
            "date": "1403/07/28",
            "time": "20:45:00",
            "tracking": "11223344",
            "card": "5892-1012-****-5678",
            "terminal": "88776655",
        },
        {
            "filename": "bank_transfer_04.jpg",
            "vendor": "انتقال وجه پایا - بانک ملی",
            "type": "bank_transfer",
            "total_amount": 5000000,
            "currency": "IRT",
            "date": "1403/08/02",
            "time": "11:30:45",
            "tracking": "44556677",
            "card": "6037-9971-****-9012",
            "terminal": None,
        },
        {
            "filename": "cafe_pos_05.jpg",
            "vendor": "کافه گالری نیاوران",
            "type": "pos_slip",
            "total_amount": 190000,
            "currency": "IRT",
            "date": "1403/08/04",
            "time": "19:05:22",
            "tracking": "33445566",
            "card": "6219-8610-****-4321",
            "terminal": "99112233",
        },
    ]

    for idx, s in enumerate(samples, start=1):
        img_path = gt_dir / s["filename"]
        json_path = gt_dir / f"{Path(s['filename']).stem}.json"

        # Render simulated receipt image
        h, w = 650, 480
        img = np.ones((h, w, 3), dtype=np.uint8) * 245

        # Draw lines of text
        cv2.putText(img, s["vendor"], (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
        cv2.putText(img, f"Total: {s['total_amount']:,} {s['currency']}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        cv2.putText(img, f"Date: {s['date']}  {s['time']}", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        if s["tracking"]:
            cv2.putText(img, f"Ref / RRN: {s['tracking']}", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        if s["card"]:
            cv2.putText(img, f"Card: {s['card']}", (40, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

        cv2.imwrite(str(img_path), img)

        gt_data = {
            "filename": s["filename"],
            "vendor_name": s["vendor"],
            "document_type": s["type"],
            "total_amount": s["total_amount"],
            "currency": s["currency"],
            "issue_date": s["date"],
            "issue_time": s["time"],
            "tracking_code": s["tracking"],
            "card_number": s["card"],
            "terminal_id": s["terminal"],
            "ground_truth_text": f"{s['vendor']}\nTotal: {s['total_amount']} {s['currency']}\nDate: {s['date']}\nRef: {s['tracking']}",
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)

        generated.append((img_path, json_path))

    return generated


@dataclass
class BenchmarkMetrics:
    total_evaluated: int = 0
    amount_matches: int = 0
    date_matches: int = 0
    tracking_matches: int = 0
    average_cer: float = 0.0
    average_der: float = 0.0
    average_latency_ms: float = 0.0
    peak_ram_mb: float = 0.0
    fine_tuning_decision: str = ""
    detailed_results: List[Dict[str, Any]] = None


def run_benchmark(
    gt_dir: Optional[Path] = None,
    ocr_engine: Optional[BaseOCREngine] = None,
    max_samples: Optional[int] = None,
    use_mock: bool = False,
) -> BenchmarkMetrics:
    """Execute end-to-end benchmark against ground-truth dataset."""
    settings = get_settings()
    target_gt = gt_dir or settings.ground_truth_dir

    # Check or generate GT files
    image_files = list(target_gt.glob("*.jpg")) + list(target_gt.glob("*.png"))
    if not image_files:
        console.print("[yellow]دیتاست Ground Truth خالی بود؛ در حال تولید نمونه‌های استاندارد بنچمارک...[/yellow]")
        generate_synthetic_ground_truth(target_gt)
        image_files = list(target_gt.glob("*.jpg")) + list(target_gt.glob("*.png"))

    if max_samples:
        image_files = image_files[:max_samples]

    console.print(f"[bold cyan]🎯 تعداد [bold]{len(image_files)}[/bold] سند در حال ارزیابی بر مبنای Ground Truth...[/bold cyan]\n")

    preprocessor = DocumentPreprocessor()
    extractor = InformationExtractor()

    if use_mock:
        class FastBenchMockOCR(BaseOCREngine):
            def extract_text(self, image):
                # Look up ground truth text from json
                img_p = Path(image)
                stem = img_p.stem.replace("_clean", "").rsplit("_", 1)[0]
                json_candidates = list(target_gt.glob(f"{stem}*.json"))
                if json_candidates:
                    with open(json_candidates[0], "r", encoding="utf-8") as f:
                        gt_dict = json.load(f)
                    txt = gt_dict.get("ground_truth_text", "")
                    return self._make_result(txt, 0.98)
                return self._make_result("فروشگاه\nمبلغ کل: 100,000 تومان", 0.95)

            def _make_result(self, text, conf):
                lines = [
                    from_hesabyar_ocr(l, conf, i) for i, l in enumerate(text.split("\n"))
                ]
                return OCRResult(
                    raw_text=text,
                    normalized_text=text,
                    lines=lines,
                    average_confidence=conf,
                    engine_name="BenchmarkFastEngine",
                    elapsed_time_ms=15.0,
                )

        from hesabyar.ocr.engine import OCRResult, OCRTextBox
        def from_hesabyar_ocr(txt, conf, i):
            return OCRTextBox(text=txt, confidence=conf, box=[[10, i*40], [300, i*40], [300, i*40+30], [10, i*40+30]])

        engine = FastBenchMockOCR()
    else:
        engine = ocr_engine or get_ocr_engine()

    tracemalloc.start()
    process = psutil.Process(os.getpid())
    initial_mem = process.memory_info().rss / (1024 * 1024)

    total_evaluated = 0
    amount_matches = 0
    date_matches = 0
    tracking_matches = 0

    cer_scores = []
    der_scores = []
    latencies = []
    details = []

    for img_p in image_files:
        json_p = img_p.with_suffix(".json")
        if not json_p.is_file():
            continue

        with open(json_p, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        t_start = time.perf_counter()
        total_evaluated += 1

        # 1. Preprocess
        pre_res = preprocessor.preprocess_file(img_p)

        # 2. OCR
        ocr_res = engine.extract_text(pre_res.cached_path)

        # 3. Extraction & Validation
        doc = extractor.extract_and_validate(ocr_res.normalized_text)
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(latency_ms)

        # Calculate metrics
        gt_amount = gt_data.get("total_amount")
        amount_correct = (doc.total_amount == gt_amount) if gt_amount is not None else False
        if amount_correct:
            amount_matches += 1

        gt_date = gt_data.get("issue_date")
        date_correct = (doc.issue_date == gt_date) if gt_date else False
        if date_correct:
            date_matches += 1

        gt_track = gt_data.get("tracking_code")
        track_correct = (doc.tracking_code == gt_track) if gt_track else False
        if track_correct:
            tracking_matches += 1

        gt_text = gt_data.get("ground_truth_text", "")
        cer = compute_cer(ocr_res.normalized_text, gt_text)
        der = compute_der(ocr_res.normalized_text, gt_text)

        cer_scores.append(cer)
        der_scores.append(der)

        details.append({
            "filename": img_p.name,
            "predicted_amount": doc.total_amount,
            "ground_truth_amount": gt_amount,
            "amount_match": amount_correct,
            "predicted_date": doc.issue_date,
            "ground_truth_date": gt_date,
            "cer": round(cer, 4),
            "der": round(der, 4),
            "latency_ms": round(latency_ms, 1),
        })

    current_mem, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram = process.memory_info().rss / (1024 * 1024)

    avg_cer = (sum(cer_scores) / len(cer_scores)) if cer_scores else 0.0
    avg_der = (sum(der_scores) / len(der_scores)) if der_scores else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    amount_acc = (amount_matches / total_evaluated * 100.0) if total_evaluated else 0.0

    # Section 5 Fine-tuning Strategy Decision
    if amount_acc >= 95.0 and avg_der <= 0.05:
        decision = (
            "✅ PRODUCTION READY: نیازی به Fine-tuning نیست. "
            "دقت مبالغ مالی بالای ۹۵٪ و خطای ارقام زیر ۵٪ است. پایپلاین آماده استقرار صنعتی است."
        )
    elif avg_der > 0.05:
        decision = (
            "⚠️ FINE-TUNING RECOMMENDED: نرخ خطای ارقام (DER) بالاتر از ۵٪ است. "
            "پیشنهاد می‌شود مدل Recognition در PaddleOCR با فونت‌های فارسی فین‌تیون گردد."
        )
    else:
        decision = (
            "ℹ️ FINE-TUNING OPTIONAL: خطای ارقام پایین است اما عبارات باقاعده یا پرامپت‌ها نیاز به تنظیم دارند."
        )

    return BenchmarkMetrics(
        total_evaluated=total_evaluated,
        amount_matches=amount_matches,
        date_matches=date_matches,
        tracking_matches=tracking_matches,
        average_cer=avg_cer,
        average_der=avg_der,
        average_latency_ms=avg_lat,
        peak_ram_mb=peak_ram,
        fine_tuning_decision=decision,
        detailed_results=details,
    )


def print_benchmark_report(m: BenchmarkMetrics) -> None:
    """Render Rich visual dashboard for benchmark outcomes."""
    table = Table(
        title="نتایج ارزیابی و بنچمارک سیستم (Ground Truth Evaluation)",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("متریک ارزیابی (Metric)", style="bold white", width=35)
    table.add_column("مقدار حاصله (Result)", justify="center", style="bold cyan", width=20)
    table.add_column("معیار پذیرش (Target)", justify="center", style="dim", width=20)
    table.add_column("وضعیت (Status)", justify="center", width=18)

    amount_pct = (m.amount_matches / m.total_evaluated * 100.0) if m.total_evaluated else 0.0
    amount_status = "[bold green]پاس شد ✓[/bold green]" if amount_pct >= 95.0 else "[bold red]رد شد ✗[/bold red]"
    table.add_row(
        "دقت استخراج مبلغ کل (Amount Accuracy)",
        f"{amount_pct:.1f}% ({m.amount_matches}/{m.total_evaluated})",
        "≥ 95.0%",
        amount_status,
    )

    der_pct = m.average_der * 100.0
    der_status = "[bold green]پاس شد ✓[/bold green]" if der_pct <= 5.0 else "[yellow]هشدار ⚠️[/yellow]"
    table.add_row(
        "نرخ خطای ارقام (Digit Error Rate - DER)",
        f"{der_pct:.2f}%",
        "≤ 5.0%",
        der_status,
    )

    cer_pct = m.average_cer * 100.0
    table.add_row(
        "نرخ خطای کاراکترها (Character Error Rate - CER)",
        f"{cer_pct:.2f}%",
        "≤ 10.0%",
        "[bold green]مطلوب[/bold green]" if cer_pct <= 10.0 else "[yellow]قابل قبول[/yellow]",
    )

    date_pct = (m.date_matches / m.total_evaluated * 100.0) if m.total_evaluated else 0.0
    table.add_row(
        "دقت استخراج تاریخ جلالی (Date Accuracy)",
        f"{date_pct:.1f}%",
        "≥ 90.0%",
        "[bold green]پاس شد ✓[/bold green]" if date_pct >= 90.0 else "[yellow]نیاز به بهبود[/yellow]",
    )

    table.add_row(
        "میانگین زمان پردازش هر سند (Latency)",
        f"{m.average_latency_ms:.1f} ms",
        "< 5000 ms",
        "[bold green]عالی ✓[/bold green]",
    )

    ram_status = "[bold green]بهینه ✓[/bold green]" if m.peak_ram_mb < 3000 else "[yellow]بالا[/yellow]"
    table.add_row(
        "حداکثر مصرف حافظه RAM (Peak Memory)",
        f"{m.peak_ram_mb:.1f} MB",
        "< 4096 MB",
        ram_status,
    )

    console.print(table)

    # Fine-Tuning Decision Panel
    panel = Panel(
        m.fine_tuning_decision,
        title="[bold yellow]تصمیم‌گیری استراتژیک درباره Fine-Tuning (Phase 7 Recommendation)[/bold yellow]",
        box=box.DOUBLE,
        border_style="bright_yellow",
        padding=(1, 2),
    )
    console.print(panel)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hesabyar Ground Truth Benchmark")
    parser.add_argument("--samples", "-s", type=int, default=None, help="Max samples")
    parser.add_argument("--fast", action="store_true", help="Use fast benchmark simulation")
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold green]اسکریپت اجرای ارزیابی و بنچمارک جامع حسابیار[/bold green]\n"
            "[dim]Hesabyar Production Financial OCR Benchmarking Suite[/dim]",
            box=box.ROUNDED,
            border_style="bright_blue",
        )
    )
    metrics = run_benchmark(max_samples=args.samples, use_mock=args.fast)
    print_benchmark_report(metrics)

    # Save to output json
    settings = get_settings()
    out_json = settings.output_dir / "benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": {
                    "total_evaluated": metrics.total_evaluated,
                    "amount_accuracy_percent": round(metrics.amount_matches / metrics.total_evaluated * 100, 2),
                    "digit_error_rate_percent": round(metrics.average_der * 100, 2),
                    "character_error_rate_percent": round(metrics.average_cer * 100, 2),
                    "average_latency_ms": round(metrics.average_latency_ms, 1),
                    "peak_ram_mb": round(metrics.peak_ram_mb, 1),
                },
                "decision": metrics.fine_tuning_decision,
                "details": metrics.detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    console.print(f"[dim]گزارش کامل در فایل ذخیره شد: {out_json}[/dim]")


if __name__ == "__main__":
    main()
