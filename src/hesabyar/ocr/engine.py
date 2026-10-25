"""OCR Engine abstractions and implementations (PaddleOCR and Tesseract fallback)."""

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Crucial environment flags for PaddleX/PaddlePaddle on Windows:
# 1. Disable oneDNN/MKLDNN by default to bypass ConvertPirAttribute2RuntimeAttribute bug
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"
# 2. Skip network latency source checks on each initialization
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from hesabyar.core.config import get_settings
from hesabyar.core.exceptions import OCREngineError
from hesabyar.core.logger import get_logger
from hesabyar.ocr.post_processor import OCRPostProcessor, clean_ocr_text
from hesabyar.processing.image_utils import load_image

logger = get_logger("ocr.engine")


@dataclass
class OCRTextBox:
    """Detected text box with bounding coordinates and recognition confidence."""
    text: str
    confidence: float
    box: List[List[int]]  # 4 corner coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    raw_text: Optional[str] = None


@dataclass
class OCRResult:
    """Complete aggregated outcome of OCR processing on a document."""
    raw_text: str
    normalized_text: str
    lines: List[OCRTextBox]
    average_confidence: float
    engine_name: str
    elapsed_time_ms: float
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


class BaseOCREngine(ABC):
    """Abstract Base Class for OCR engines."""

    def __init__(self):
        self.settings = get_settings()
        self.post_processor = OCRPostProcessor(digits_to_english=True)

    @abstractmethod
    def extract_text(self, image: Union[np.ndarray, Path, str]) -> OCRResult:
        """Perform text recognition on the input image."""
        pass


class EasyOCREngine(BaseOCREngine):
    """
    High-performance GPU-accelerated OCR engine using EasyOCR with Persian/Arabic models.
    Supports CUDA GPU acceleration on PyTorch with automatic CPU fallback.
    """

    def __init__(self, lang: Optional[str] = None, use_gpu: Optional[bool] = None):
        super().__init__()
        self.lang = lang or self.settings.ocr_lang

        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            pass

        self.use_gpu = (use_gpu if use_gpu is not None else self.settings.use_gpu) and has_cuda
        self._reader = None

    def _ensure_initialized(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            try:
                import easyocr
            except ImportError as e:
                logger.error("EasyOCR is not installed.")
                raise OCREngineError(
                    "EasyOCR is not installed. Please install with: pip install easyocr"
                ) from e

            lang_list = ["fa", "en"] if self.lang in ("fa", "ar", "persian") else [self.lang, "en"]
            device_str = "CUDA GPU" if self.use_gpu else "CPU / RAM"
            logger.info(f"Initializing EasyOCR (languages={lang_list}, device={device_str})...")
            try:
                self._reader = easyocr.Reader(lang_list, gpu=self.use_gpu)
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                raise OCREngineError(f"Failed to initialize EasyOCR: {e}") from e

    def extract_text(self, image: Union[np.ndarray, Path, str]) -> OCRResult:
        """Run EasyOCR recognition on image array or file path."""
        self._ensure_initialized()
        start_time = time.perf_counter()

        if isinstance(image, (str, Path)):
            img_array = load_image(image)
        else:
            img_array = image

        # Resolution normalization for optimal OCR font detection
        h, w = img_array.shape[:2]
        scale_factor = 1.0
        if w < 1000:
            scale_factor = 1000.0 / w
            import cv2
            img_array = cv2.resize(
                img_array, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
            )

        try:
            results = self._reader.readtext(img_array)
        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {e}")
            raise OCREngineError(f"EasyOCR recognition error: {e}") from e

        raw_boxes: List[Dict[str, Any]] = []
        for bbox, text_str, conf in results:
            if scale_factor != 1.0:
                poly_list = [[int(pt[0] / scale_factor), int(pt[1] / scale_factor)] for pt in bbox]
            else:
                poly_list = [[int(pt[0]), int(pt[1])] for pt in bbox]

            raw_boxes.append({
                "box": poly_list,
                "text": str(text_str).strip(),
                "confidence": float(conf),
            })

        sorted_boxes, normalized_text = self.post_processor.process(raw_boxes)

        lines: List[OCRTextBox] = []
        total_conf = 0.0
        for b in sorted_boxes:
            conf = b["confidence"]
            total_conf += conf
            lines.append(
                OCRTextBox(
                    text=b["text"],
                    confidence=conf,
                    box=b["box"],
                    raw_text=b.get("raw_text"),
                )
            )

        avg_conf = (total_conf / len(lines)) if lines else 0.0
        raw_text = "\n".join(b.get("raw_text", b["text"]) for b in sorted_boxes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            lines=lines,
            average_confidence=avg_conf,
            engine_name="EasyOCR (CUDA GPU)" if self.use_gpu else "EasyOCR (CPU)",
            elapsed_time_ms=elapsed_ms,
            extra_metadata={"gpu": self.use_gpu, "scale_factor": scale_factor},
        )


class PaddleOCREngine(BaseOCREngine):
    """
    Primary OCR engine using PaddleOCR with Arabic/Persian language models.
    Supports high-speed detection and recognition optimized for receipts and invoices.
    """

    def __init__(self, lang: Optional[str] = None, use_gpu: Optional[bool] = None):
        super().__init__()
        self.lang = lang or self.settings.ocr_lang
        self.use_gpu = use_gpu if use_gpu is not None else self.settings.use_gpu
        self._ocr_instance = None

    def _ensure_initialized(self):
        """Lazy initialization of PaddleOCR model."""
        if self._ocr_instance is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as e:
                logger.error("PaddleOCR is not installed.")
                raise OCREngineError(
                    "PaddleOCR is not installed. Please install with: pip install paddlepaddle paddleocr"
                ) from e

            logger.info(f"Initializing PaddleOCR (lang='{self.lang}', use_gpu={self.use_gpu})...")
            try:
                self._ocr_instance = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                raise OCREngineError(f"Failed to initialize PaddleOCR: {e}") from e

    def extract_text(self, image: Union[np.ndarray, Path, str]) -> OCRResult:
        """Run PaddleOCR on image array or file path."""
        self._ensure_initialized()
        start_time = time.perf_counter()

        if isinstance(image, (str, Path)):
            img_array = load_image(image)
        else:
            img_array = image

        try:
            # In PaddleOCR 3.x / PaddleX, predict returns generator of OCRResult dict-like objects
            predictions = list(self._ocr_instance.predict(img_array))
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            raise OCREngineError(f"PaddleOCR recognition error: {e}") from e

        raw_boxes: List[Dict[str, Any]] = []
        for pred in predictions:
            texts = pred.get("rec_texts", [])
            scores = pred.get("rec_scores", [])
            polys = pred.get("dt_polys", [])

            for text_str, conf, poly in zip(texts, scores, polys):
                if hasattr(poly, "tolist"):
                    poly_list = poly.tolist()
                else:
                    poly_list = [[int(pt[0]), int(pt[1])] for pt in poly]

                raw_boxes.append({
                    "box": poly_list,
                    "text": str(text_str),
                    "confidence": float(conf),
                })

        # Process through Persian post-processor: character cleanup, digit normalization, RTL layout
        sorted_boxes, normalized_text = self.post_processor.process(raw_boxes)

        lines: List[OCRTextBox] = []
        total_conf = 0.0
        for b in sorted_boxes:
            conf = b["confidence"]
            total_conf += conf
            lines.append(
                OCRTextBox(
                    text=b["text"],
                    confidence=conf,
                    box=b["box"],
                    raw_text=b.get("raw_text"),
                )
            )

        avg_conf = (total_conf / len(lines)) if lines else 0.0
        raw_text = "\n".join(b.get("raw_text", b["text"]) for b in sorted_boxes)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            lines=lines,
            average_confidence=avg_conf,
            engine_name="PaddleOCR",
            elapsed_time_ms=elapsed_ms,
            extra_metadata={"lang": self.lang},
        )


class TesseractOCREngine(BaseOCREngine):
    """Secondary/Fallback OCR engine using Tesseract via pytesseract."""

    def __init__(self, lang: str = "fas+ara"):
        super().__init__()
        self.lang = lang

    def extract_text(self, image: Union[np.ndarray, Path, str]) -> OCRResult:
        start_time = time.perf_counter()
        try:
            import pytesseract
        except ImportError as e:
            raise OCREngineError("pytesseract is not installed.") from e

        if isinstance(image, (str, Path)):
            img_array = load_image(image)
        else:
            img_array = image

        try:
            data = pytesseract.image_to_data(
                img_array, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            raise OCREngineError(f"Tesseract OCR failed: {e}") from e

        raw_boxes = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 0:
                x, y, w, h = (
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                raw_boxes.append({
                    "box": box,
                    "text": text,
                    "confidence": conf / 100.0,
                })

        sorted_boxes, normalized_text = self.post_processor.process(raw_boxes)
        lines = [
            OCRTextBox(
                text=b["text"],
                confidence=b["confidence"],
                box=b["box"],
                raw_text=b.get("raw_text"),
            )
            for b in sorted_boxes
        ]
        avg_conf = (sum(l.confidence for l in lines) / len(lines)) if lines else 0.0
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return OCRResult(
            raw_text="\n".join(l.text for l in lines),
            normalized_text=normalized_text,
            lines=lines,
            average_confidence=avg_conf,
            engine_name="Tesseract",
            elapsed_time_ms=elapsed_ms,
        )


def get_ocr_engine(engine_name: Optional[str] = None) -> BaseOCREngine:
    """Factory to retrieve configured OCR engine instance."""
    settings = get_settings()
    target_engine = (engine_name or settings.ocr_engine).lower()

    if target_engine in ("easyocr", "easy"):
        return EasyOCREngine(lang=settings.ocr_lang, use_gpu=settings.use_gpu)
    elif target_engine in ("paddleocr", "paddle"):
        return PaddleOCREngine(lang=settings.ocr_lang, use_gpu=settings.use_gpu)
    elif target_engine in ("tesseract", "tess"):
        return TesseractOCREngine()
    else:
        raise ValueError(
            f"Unsupported OCR engine: {target_engine}. Choose 'easyocr', 'paddleocr', or 'tesseract'."
        )
