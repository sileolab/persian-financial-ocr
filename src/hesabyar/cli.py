"""Command Line Interface (CLI) for Hesabyar built with Typer and Rich."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows consoles to prevent charmap UnicodeEncodeErrors
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hesabyar import __app_name__, __version__
from hesabyar.core.config import get_settings
from hesabyar.core.logger import get_logger, setup_logger
from hesabyar.db.state_manager import get_state_manager
from hesabyar.extraction import InformationExtractor
from hesabyar.exporters import (
    export_to_csv,
    export_to_excel,
    export_to_json,
    export_to_word,
    generate_summary_report,
)
from hesabyar.models.schemas import FileStatus
from hesabyar.ocr.engine import get_ocr_engine
from hesabyar.pipeline import BatchPipelineManager, BatchSummary, PipelineResult
from hesabyar.processing.preprocessor import (
    DocumentPreprocessor,
    PreprocessingConfig,
)

app = typer.Typer(
    name="hesabyar",
    help="Hesabyar - Production-Grade Financial Document OCR and Extraction System",
    no_args_is_help=True,
)
console = Console(force_terminal=True, legacy_windows=False)
logger = get_logger("cli")


def render_banner() -> None:
    """Render the stylish Rich application banner."""
    title = Text()
    title.append(" 🧾 ", style="bold green")
    title.append(f"{__app_name__} ", style="bold cyan")
    title.append(f"v{__version__}", style="dim white")
    title.append(" - Production Financial Document OCR System", style="bold yellow")

    desc = Text("Production-Grade Financial Document OCR & Extraction Pipeline\n", style="italic")
    desc.append("High-precision GPU-accelerated OCR, Rule-based & LLM Financial Parsing", style="dim")

    panel = Panel(
        Align.center(f"{title}\n{desc}"),
        box=box.ROUNDED,
        border_style="bright_blue",
        padding=(0, 2),
    )
    console.print(panel)


def render_stats_table(stats) -> Table:
    """Render a clean Rich summary table for processing statistics."""
    table = Table(
        title="Document Processing Overview",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        title_style="bold cyan",
    )
    table.add_column("Metric", style="bold white", width=32)
    table.add_column("Count", justify="right", style="bold yellow", width=14)
    table.add_column("Status", justify="center", width=22)

    table.add_row("Total Registered Files", str(stats.total_files), "[blue]Registered in DB[/blue]")
    table.add_row("Pending Queue", str(stats.pending), "[yellow]Ready for Processing[/yellow]")
    table.add_row("Image Preprocessing", str(stats.preprocessing), "[cyan]Preprocessing[/cyan]")
    table.add_row("OCR Text Recognition", str(stats.ocr_processing), "[magenta]OCR Processing[/magenta]")
    table.add_row("Structured Extraction", str(stats.extracting), "[magenta]Rule/LLM Parsing[/magenta]")
    table.add_row("Completed (Verified)", str(stats.completed), "[green]Completed ✓[/green]")
    table.add_row("Needs Manual Review", str(stats.needs_review), "[bold orange3]Discrepancy ⚠️[/bold orange3]")
    table.add_row("Failed", str(stats.failed), "[red]Failed ✗[/red]")

    return table


@app.callback()
def main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging"),
):
    """Hesabyar CLI Root Callback."""
    level = "DEBUG" if verbose else None
    setup_logger(log_level=level)


@app.command(name="start")
def start(
    input_folder: Optional[str] = typer.Option(
        None, "--input", "-i", help="Custom input directory containing documents"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="Number of files to process per batch"
    ),
):
    """
    Start processing lifecycle and scan input documents into the database.
    """
    render_banner()
    settings = get_settings()
    settings.ensure_directories()
    state_mgr = get_state_manager()

    target_input = Path(input_folder) if input_folder else settings.input_dir

    console.print(f"[bold cyan]📁 Input Directory:[/bold cyan] [underline]{target_input}[/underline]")
    console.print(f"[bold cyan]🗄️ State Database:[/bold cyan] [underline]{settings.db_path}[/underline]")
    console.print(f"[bold cyan]📝 Log File:[/bold cyan] [underline]{settings.log_file_path}[/underline]\n")

    # Scan and register files
    with console.status("[bold green]Scanning and syncing documents in database...[/bold green]"):
        new_registered, total_scanned = state_mgr.scan_and_sync_input_dir(target_input)

    if new_registered > 0:
        console.print(f"[green]✓ Discovered and registered [bold]{new_registered}[/bold] new documents.[/green]")
    elif total_scanned > 0:
        console.print(f"[dim]All {total_scanned} files are already registered in the database.[/dim]")
    else:
        console.print(f"[yellow]⚠️ No files found in input directory: {target_input}[/yellow]")
        console.print("[dim]Copy invoice and receipt images into the folder above.[/dim]\n")

    # Display current stats
    stats = state_mgr.get_stats()
    console.print(render_stats_table(stats))

    console.print(
        "\n[bold green]✓ Infrastructure loaded successfully. Database in stable WAL mode.[/bold green]"
    )


@app.command(name="preprocess")
def preprocess_cmd(
    input_path: Optional[str] = typer.Option(
        None, "--input", "-i", help="Input directory or single image file"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for cached clean images"
    ),
    method: str = typer.Option(
        "none", "--method", "-m", help="Binarization method: 'none', 'adaptive', or 'otsu'"
    ),
    no_deskew: bool = typer.Option(
        False, "--no-deskew", help="Disable automatic deskewing"
    ),
):
    """
    Batch image preprocessing and resolution normalization.
    """
    render_banner()
    settings = get_settings()
    settings.ensure_directories()
    state_mgr = get_state_manager()

    target_input = Path(input_path) if input_path else settings.input_dir
    target_cache = Path(output_dir) if output_dir else settings.cache_dir
    target_cache.mkdir(parents=True, exist_ok=True)

    config = PreprocessingConfig(
        enable_deskew=not no_deskew,
        binarization_method=method,
    )
    preprocessor = DocumentPreprocessor(config=config)

    # Gather files to process
    files_to_process = []
    if target_input.is_file():
        record, _ = state_mgr.register_file(target_input)
        files_to_process.append(record)
    elif target_input.is_dir():
        state_mgr.scan_and_sync_input_dir(target_input)
        files_to_process = state_mgr.get_records_by_status(FileStatus.PENDING)
    else:
        console.print(f"[red]Invalid input path: {target_input}[/red]")
        raise typer.Exit(1)

    if not files_to_process:
        console.print("[yellow]No files in PENDING status found for preprocessing.[/yellow]")
        return

    console.print(f"[bold cyan]🚀 Starting preprocessing for [bold]{len(files_to_process)}[/bold] documents...[/bold cyan]")
    console.print(f"[dim]Cache Directory: {target_cache}[/dim]\n")

    results_table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta")
    results_table.add_column("File", style="bold white", width=25)
    results_table.add_column("Skew Angle", justify="right", width=15)
    results_table.add_column("Processed Size", justify="center", width=15)
    results_table.add_column("Elapsed (ms)", justify="right", style="cyan", width=14)
    results_table.add_column("Status", justify="center", width=14)

    success_count = 0
    fail_count = 0

    for rec in files_to_process:
        src_path = Path(rec.file_path)
        if not src_path.exists():
            console.print(f"[red]Source file not found: {src_path}[/red]")
            state_mgr.update_status(rec.file_hash, FileStatus.FAILED, error_message="Source file missing")
            fail_count += 1
            continue

        try:
            state_mgr.update_status(rec.file_hash, FileStatus.PREPROCESSING)
            out_file = target_cache / f"{src_path.stem}_{rec.file_hash[:8]}_clean.png"
            res = preprocessor.preprocess_file(
                input_path=src_path,
                output_path=out_file,
                file_hash=rec.file_hash,
            )
            state_mgr.update_status(rec.file_hash, FileStatus.PREPROCESSING)
            results_table.add_row(
                rec.file_name,
                f"{res.skew_angle:+.2f}°",
                f"{res.processed_shape[1]}x{res.processed_shape[0]}",
                f"{res.elapsed_time_ms:.1f}",
                "[green]✓ Ready[/green]",
            )
            success_count += 1
        except Exception as e:
            logger.error(f"Failed preprocessing {rec.file_name}: {e}")
            state_mgr.update_status(rec.file_hash, FileStatus.FAILED, error_message=str(e))
            results_table.add_row(
                rec.file_name,
                "-",
                "-",
                "-",
                "[red]Failed ✗[/red]",
            )
            fail_count += 1

    console.print(results_table)
    console.print(
        f"\n[bold green]✓ Preprocessing complete: [bold]{success_count}[/bold] succeeded, "
        f"[bold]{fail_count}[/bold] failed.[/bold green]"
    )


@app.command(name="ocr")
def ocr_cmd(
    input_path: Optional[str] = typer.Option(
        None, "--input", "-i", help="Specific image file or directory to perform OCR on"
    ),
    engine_name: Optional[str] = typer.Option(
        None, "--engine", "-e", help="OCR engine: 'easyocr', 'paddleocr', or 'tesseract'"
    ),
    show_boxes: bool = typer.Option(
        False, "--boxes", help="Print bounding coordinates of each detected line"
    ),
):
    """
    Extract raw text from documents using high-performance GPU-accelerated OCR.
    """
    render_banner()
    settings = get_settings()
    settings.ensure_directories()
    state_mgr = get_state_manager()

    try:
        ocr_engine = get_ocr_engine(engine_name)
    except Exception as e:
        console.print(f"[red]Error initializing OCR engine: {e}[/red]")
        raise typer.Exit(1)

    # Determine files to process
    targets: list = []
    if input_path:
        p = Path(input_path).resolve()
        if p.is_file():
            rec, _ = state_mgr.register_file(p)
            targets.append(rec)
        elif p.is_dir():
            state_mgr.scan_and_sync_input_dir(p)
            targets = state_mgr.get_records_by_status(FileStatus.PREPROCESSING) or state_mgr.get_pending_records()
    else:
        # Pick files that are either in PREPROCESSING or PENDING
        targets = state_mgr.get_records_by_status(FileStatus.PREPROCESSING)
        if not targets:
            targets = state_mgr.get_pending_records()

    if not targets:
        console.print("[yellow]No files found for OCR text extraction.[/yellow]")
        console.print("[dim]Place images in data/input or run hesabyar start.[/dim]")
        return

    console.print(f"[bold cyan]🔍 Starting OCR text extraction for [bold]{len(targets)}[/bold] documents...[/bold cyan]")
    engine_label = engine_name or settings.ocr_engine
    console.print(f"[dim]Selected Engine: {engine_label}[/dim]\n")

    preprocessor = DocumentPreprocessor()

    for rec in targets:
        src_path = Path(rec.file_path)
        if not src_path.exists():
            console.print(f"[red]Source file not found: {src_path}[/red]")
            continue

        cache_candidate = settings.cache_dir / f"{src_path.stem}_{rec.file_hash[:8]}_clean.png"
        if cache_candidate.exists():
            image_to_ocr = cache_candidate
            source_tag = "[cyan](Cached Image)[/cyan]"
        else:
            pre_res = preprocessor.preprocess_file(src_path, cache_candidate, file_hash=rec.file_hash)
            image_to_ocr = pre_res.cached_path
            source_tag = "[yellow](Direct Image)[/yellow]"

        console.print(f"[bold magenta]▶ Processing:[/bold magenta] [bold white]{rec.file_name}[/bold white] {source_tag}")

        state_mgr.update_status(rec.file_hash, FileStatus.OCR_PROCESSING)
        try:
            ocr_res = ocr_engine.extract_text(image_to_ocr)

            # Store result and confidence in DB
            state_mgr.update_status(
                rec.file_hash,
                FileStatus.EXTRACTING,
                ocr_confidence=ocr_res.average_confidence,
                extracted_data_json=ocr_res.normalized_text,
            )

            conf_color = "green" if ocr_res.average_confidence >= 0.85 else "yellow"
            console.print(
                f"  [dim]Elapsed: {ocr_res.elapsed_time_ms:.1f}ms | "
                f"Confidence: [/dim][bold {conf_color}]{ocr_res.average_confidence:.2%}[/bold {conf_color}]"
            )

            preview_text = ocr_res.normalized_text if ocr_res.normalized_text else "[dim](No text detected)[/dim]"
            console.print(
                Panel(
                    preview_text,
                    title=f"[bold green]Extracted Text ({len(ocr_res.lines)} lines)[/bold green]",
                    box=box.ROUNDED,
                    border_style="green",
                )
            )

            if show_boxes and ocr_res.lines:
                box_table = Table(box=box.SIMPLE, header_style="bold cyan")
                box_table.add_column("#", justify="right", width=6)
                box_table.add_column("Line Text", style="bold white", width=35)
                box_table.add_column("Confidence", justify="right", width=12)
                box_table.add_column("Bounding Box (BBox)", style="dim", width=30)
                for idx, line in enumerate(ocr_res.lines, start=1):
                    box_table.add_row(
                        str(idx),
                        line.text,
                        f"{line.confidence:.2%}",
                        str(line.box),
                    )
                console.print(box_table)

        except Exception as e:
            logger.error(f"OCR failed for {rec.file_name}: {e}")
            state_mgr.update_status(rec.file_hash, FileStatus.FAILED, error_message=str(e))
            console.print(f"[red]OCR Error: {e}[/red]")

    console.print("\n[bold green]✓ OCR text extraction completed successfully.[/bold green]")


@app.command(name="extract")
def extract_cmd(
    target_status: str = typer.Option(
        "EXTRACTING", "--status", "-s", help="Process files with this DB status"
    ),
):
    """
    Structured extraction and financial validation of amounts, dates, and card fields.
    """
    render_banner()
    state_mgr = get_state_manager()
    extractor = InformationExtractor()

    status_enum = getattr(FileStatus, target_status.upper(), FileStatus.EXTRACTING)
    records = state_mgr.get_records_by_status(status_enum)

    if not records:
        console.print(f"[yellow]No documents found with status '{status_enum.value}' for extraction.[/yellow]")
        console.print("[dim]Run hesabyar ocr first to extract text.[/dim]")
        return

    console.print(f"[bold cyan]📊 Starting structured extraction for [bold]{len(records)}[/bold] documents...[/bold cyan]\n")

    summary_table = Table(box=box.ROUNDED, header_style="bold magenta")
    summary_table.add_column("Document", style="bold white", width=22)
    summary_table.add_column("Type", width=14)
    summary_table.add_column("Total Amount", justify="right", style="bold green", width=16)
    summary_table.add_column("Currency", justify="center", width=8)
    summary_table.add_column("Date", justify="center", width=12)
    summary_table.add_column("Tracking Code", justify="center", width=16)
    summary_table.add_column("Validation Status", justify="center", width=20)

    for rec in records:
        text_content = rec.extracted_data_json or ""
        doc = extractor.extract_and_validate(text_content)

        new_status = FileStatus.NEEDS_REVIEW if doc.needs_review else FileStatus.COMPLETED
        state_mgr.update_status(
            rec.file_hash,
            new_status,
            extracted_data_json=doc.model_dump_json(),
            error_message="; ".join(doc.review_reasons) if doc.review_reasons else None,
        )

        amount_str = f"{doc.total_amount:,}" if doc.total_amount is not None else "-"
        currency_str = doc.currency.value if doc.currency else "-"
        date_str = doc.issue_date or "-"
        tracking_str = doc.tracking_code or "-"

        if doc.needs_review:
            status_display = "[bold orange3]Needs Review ⚠️[/bold orange3]"
        else:
            status_display = "[bold green]Verified ✓[/bold green]"

        summary_table.add_row(
            rec.file_name,
            doc.document_type,
            amount_str,
            currency_str,
            date_str,
            tracking_str,
            status_display,
        )

    console.print(summary_table)
    console.print("\n[bold green]✓ Structured extraction and financial validation complete.[/bold green]")


@app.command(name="run")
def run_pipeline(
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of parallel worker threads"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of documents to process in this run"
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Retry previously failed documents"
    ),
):
    """
    Run end-to-end batch processing pipeline with pause/resume support.
    """
    render_banner()
    settings = get_settings()
    settings.ensure_directories()
    state_mgr = get_state_manager()

    if retry_failed:
        reopened = state_mgr.reset_status(target="failed")
        if reopened > 0:
            console.print(f"[yellow]{reopened} previously failed documents reset to PENDING queue.[/yellow]")

    # Sync input directory to discover any newly added files
    new_reg, total_scanned = state_mgr.scan_and_sync_input_dir()

    pending_records = state_mgr.get_pending_records()
    if not pending_records:
        console.print("[green]All documents have already been processed (no documents in PENDING status).[/green]")
        console.print(render_stats_table(state_mgr.get_stats()))
        return

    num_to_process = len(pending_records) if limit is None else min(len(pending_records), limit)
    effective_workers = workers or settings.max_workers

    console.print(f"[bold cyan]🚀 Starting batch pipeline for [bold]{num_to_process}[/bold] documents with [bold]{effective_workers}[/bold] parallel workers...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to pause processing gracefully (state is securely preserved).[/dim]\n")

    batch_mgr = BatchPipelineManager(state_manager=state_mgr)

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Processing documents...", total=num_to_process)

        def on_progress(res: PipelineResult, completed: int, total: int):
            status_text = f"[green]✓ {res.file_record.file_name}[/green]" if res.status == FileStatus.COMPLETED else (
                f"[yellow]⚠️ {res.file_record.file_name}[/yellow]" if res.status == FileStatus.NEEDS_REVIEW else
                f"[red]✗ {res.file_record.file_name}[/red]"
            )
            progress.update(task_id, advance=1, description=f"Processing: {status_text}")

        try:
            summary = batch_mgr.run_batch(
                max_workers=effective_workers,
                max_items=limit,
                progress_callback=on_progress,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Stop signal received. Gracefully pausing pipeline workers...[/yellow]")
            batch_mgr.request_stop()
            summary = BatchSummary(is_paused=True)

    console.print("\n")
    if summary.is_paused:
        console.print("[bold yellow]⏸️ Processing paused. Document statuses safely saved to SQLite database.[/bold yellow]")
        console.print("[dim]Re-run 'hesabyar run' anytime to resume directly from the next pending file.[/dim]\n")
    else:
        console.print("[bold green]✓ Batch pipeline processing completed successfully.[/bold green]\n")

    # Display final overview
    stats = state_mgr.get_stats()
    console.print(render_stats_table(stats))


@app.command(name="export")
def export_cmd(
    format_type: str = typer.Option(
        "all", "--format", "-f", help="Export format: 'all', 'excel', 'word', 'csv', 'json'"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Target output directory"
    ),
):
    """
    Generate Excel, Word, CSV, and JSON export reports.
    """
    render_banner()
    settings = get_settings()
    settings.ensure_directories()
    state_mgr = get_state_manager()

    target_dir = Path(output_dir) if output_dir else settings.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    records = state_mgr.get_all_records()
    if not records:
        console.print("[yellow]No document records found in database for report export.[/yellow]")
        return

    fmt = format_type.lower()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    exported_files = []

    with console.status("[bold green]Generating export reports...[/bold green]"):
        # Excel
        if fmt in ["all", "excel", "xlsx"]:
            excel_path = target_dir / f"hesabyar_report_{timestamp_str}.xlsx"
            export_to_excel(records, excel_path)
            exported_files.append(("Microsoft Excel", excel_path))

        # Word
        if fmt in ["all", "word", "docx"]:
            word_path = target_dir / f"hesabyar_executive_report_{timestamp_str}.docx"
            export_to_word(records, word_path)
            exported_files.append(("Microsoft Word", word_path))

        # CSV
        if fmt in ["all", "csv"]:
            csv_path = target_dir / f"hesabyar_data_{timestamp_str}.csv"
            export_to_csv(records, csv_path)
            exported_files.append(("CSV (UTF-8 BOM)", csv_path))

        # JSON
        if fmt in ["all", "json"]:
            json_path = target_dir / f"hesabyar_summary_{timestamp_str}.json"
            export_to_json(records, json_path)
            exported_files.append(("Structured JSON", json_path))

    # Display results
    export_table = Table(
        title="Generated Export Reports",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    export_table.add_column("Format", style="bold white", width=22)
    export_table.add_column("File Path", style="cyan", width=50)
    export_table.add_column("File Size", justify="right", style="green", width=14)

    for f_type, p in exported_files:
        size_bytes = p.stat().st_size
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes > 1024 else f"{size_bytes} B"
        export_table.add_row(f_type, str(p), size_str)

    console.print(export_table)

    summary = generate_summary_report(records)
    kpis = summary["kpis"]
    fin = summary["financial_totals"]

    console.print(
        f"\n[bold green]✓ Summary reporting complete:[/bold green] "
        f"[bold]{kpis['completed_verified']}[/bold] verified | "
        f"[bold]{kpis['needs_review']}[/bold] needs review | "
        f"Total valid amount: [bold green]{fin['equivalent_total_toman']:,} Toman[/bold green]"
    )


@app.command(name="benchmark")
def benchmark_cmd(
    samples: Optional[int] = typer.Option(
        None, "--samples", "-s", help="Max number of ground truth samples to evaluate"
    ),
    fast: bool = typer.Option(
        False, "--fast", help="Run benchmark with fast engine"
    ),
):
    """
    Run Ground Truth benchmark and compute financial precision metrics.
    """
    render_banner()
    import importlib.util
    bench_script = get_settings().project_root / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark", str(bench_script))
    if spec and spec.loader:
        bench_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench_mod)
        metrics = bench_mod.run_benchmark(max_samples=samples, use_mock=fast)
        bench_mod.print_benchmark_report(metrics)
    else:
        console.print("[red]Benchmark script not found.[/red]")


@app.command(name="status")
def show_status(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of recent records to display"),
):
    """
    Display database statistics and recent document records.
    """
    render_banner()
    state_mgr = get_state_manager()
    stats = state_mgr.get_stats()

    console.print(render_stats_table(stats))

    records = state_mgr.get_all_records()
    if records:
        console.print(f"\n[bold cyan]Recent Documents (up to {limit} records):[/bold cyan]")
        rec_table = Table(box=box.SIMPLE, header_style="bold cyan")
        rec_table.add_column("Hash", style="dim", width=12)
        rec_table.add_column("Filename", style="bold white", width=30)
        rec_table.add_column("Status", justify="center", width=18)
        rec_table.add_column("OCR Conf", justify="right", width=10)
        rec_table.add_column("Registered At", style="dim", width=20)

        for rec in records[:limit]:
            status_style = {
                FileStatus.PENDING: "[yellow]PENDING[/yellow]",
                FileStatus.PREPROCESSING: "[cyan]PREPROCESSING[/cyan]",
                FileStatus.OCR_PROCESSING: "[magenta]OCR[/magenta]",
                FileStatus.EXTRACTING: "[magenta]EXTRACTING[/magenta]",
                FileStatus.COMPLETED: "[green]COMPLETED[/green]",
                FileStatus.NEEDS_REVIEW: "[orange3]NEEDS_REVIEW[/orange3]",
                FileStatus.FAILED: "[red]FAILED[/red]",
            }.get(rec.status, str(rec.status))

            ocr_conf = f"{rec.ocr_confidence:.2f}" if rec.ocr_confidence is not None else "-"
            created_str = rec.created_at.strftime("%Y-%m-%d %H:%M") if rec.created_at else "-"
            rec_table.add_row(
                rec.file_hash[:10] + "...",
                rec.file_name,
                status_style,
                ocr_conf,
                created_str,
            )
        console.print(rec_table)
    else:
        console.print("\n[dim]No documents have been registered in the database yet.[/dim]")


@app.command(name="scan")
def scan_directory(
    path: Optional[str] = typer.Argument(None, help="Directory to scan. Defaults to configured input_dir"),
):
    """
    Manually scan input directory and register documents in the database.
    """
    settings = get_settings()
    state_mgr = get_state_manager()
    target_dir = Path(path) if path else settings.input_dir

    console.print(f"[bold cyan]🔍 Scanning directory:[/bold cyan] {target_dir}")
    new_count, total = state_mgr.scan_and_sync_input_dir(target_dir)
    console.print(f"[green]✓ Directory scan completed: [bold]{new_count}[/bold] new documents added ({total} total files).[/green]")


@app.command(name="reset")
def reset_status(
    target: str = typer.Option("failed", "--target", "-t", help="'failed' to reset only errors, 'all' to reset all records"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """
    Reset document statuses back to PENDING for re-processing.
    """
    if not force:
        confirm = typer.confirm(f"Are you sure you want to reset records matching '{target}' to PENDING?")
        if not confirm:
            console.print("[yellow]Operation cancelled.[/yellow]")
            raise typer.Abort()

    state_mgr = get_state_manager()
    affected = state_mgr.reset_status(target=target)
    console.print(f"[green]✓ Successfully reset [bold]{affected}[/bold] records back to PENDING status.[/green]")


@app.command(name="info")
def system_info():
    """
    Display system information, hardware, GPU, and active configuration.
    """
    render_banner()
    settings = get_settings()

    has_cuda = False
    gpu_name = "None (CPU only)"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    info_table = Table(box=box.ROUNDED, header_style="bold blue")
    info_table.add_column("Configuration", style="bold white", width=28)
    info_table.add_column("Value", style="cyan", width=48)

    info_table.add_row("Application Version", __version__)
    info_table.add_row("Environment Mode", settings.hesabyar_env)
    info_table.add_row("Project Root", str(settings.project_root))
    info_table.add_row("Data Directory", str(settings.data_dir))
    info_table.add_row("Database Path", str(settings.db_path))
    info_table.add_row("Parallel Workers", str(settings.max_workers))
    info_table.add_row("Default OCR Engine", settings.ocr_engine)
    info_table.add_row("OCR Language", settings.ocr_lang)
    info_table.add_row("GPU Acceleration", f"Enabled: {has_cuda} ({gpu_name})")
    info_table.add_row("Default LLM Model", settings.llm_model)

    console.print(info_table)


if __name__ == "__main__":
    app()
