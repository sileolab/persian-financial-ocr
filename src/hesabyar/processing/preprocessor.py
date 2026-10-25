"""High-performance image preprocessing pipeline for financial documents and receipts."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from hesabyar.core.config import get_settings
from hesabyar.core.exceptions import ImagePreprocessingError
from hesabyar.core.logger import get_logger
from hesabyar.processing.image_utils import (
    convert_to_grayscale,
    load_image,
    resize_with_aspect_ratio,
    save_image,
)

logger = get_logger("processing.preprocessor")


@dataclass
class PreprocessingConfig:
    """Tunable parameters for the image preprocessing pipeline."""
    enable_deskew: bool = True
    enable_shadow_removal: bool = False
    enable_denoise: bool = False
    binarization_method: str = "none"  # 'none' (recommended for deep learning OCR), 'adaptive', 'otsu'
    min_dimension: int = 1000
    max_dimension: int = 2600
    adaptive_block_size: int = 25
    adaptive_c: int = 11
    clahe_clip_limit: float = 2.0
    deskew_min_angle: float = 0.5


@dataclass
class PreprocessingResult:
    """Metadata and outcomes of document preprocessing."""
    input_path: Path
    cached_path: Path
    file_hash: Optional[str]
    skew_angle: float
    original_shape: Tuple[int, int]
    processed_shape: Tuple[int, int]
    elapsed_time_ms: float


class DocumentPreprocessor:
    """
    Industrial-grade preprocessing engine for financial invoices and thermal receipts.

    Implements:
    - Shadow removal & illumination correction
    - Skew angle detection and deskewing
    - Edge-preserving denoising
    - Adaptive and Otsu binarization
    - Resolution normalization
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.settings = get_settings()

    def estimate_skew_angle(self, gray_image: np.ndarray) -> float:
        """
        Estimate the skew angle of text lines using morphological dilation and minAreaRect.
        Returns angle in degrees (negative = counter-clockwise, positive = clockwise).
        """
        h, w = gray_image.shape[:2]

        # Otsu thresholding with inverted polarity (text = white, background = black)
        _, thresh = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Merge horizontal text characters into continuous word/line blobs
        kernel_width = max(15, int(w * 0.03))
        kernel_height = max(3, int(h * 0.005))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours of dilated text lines
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        angles: List[float] = []
        min_area = (w * h) * 0.0005  # Filter tiny noise blobs

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(cnt)
            rect_w, rect_h = rect[1]
            angle = rect[2]

            # minAreaRect returns angle in [-90, 0] or [0, 90] depending on OpenCV version
            if rect_w < rect_h:
                angle = angle - 90 if angle > 0 else angle + 90

            # Only accept sensible skew angles for scanned/captured documents (-45 to 45 deg)
            if -45.0 <= angle <= 45.0 and abs(angle) > 0.1:
                angles.append(angle)

        if not angles:
            return 0.0

        # Use median to resist outliers from vertical borders or graphics
        median_angle = float(np.median(angles))
        return median_angle

    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect skew and rotate image back to 0 degrees with white border padding.
        Returns: (deskewed_image, detected_angle)
        """
        gray = convert_to_grayscale(image)
        angle = self.estimate_skew_angle(gray)

        if abs(angle) < self.config.deskew_min_angle:
            return image, 0.0

        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotation matrix around image center
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Compute new bounding dimensions to avoid clipping
        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        border_val = (255, 255, 255) if len(image.shape) == 3 else 255
        deskewed = cv2.warpAffine(
            image,
            rot_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )

        logger.debug(f"Deskew applied: rotated by {angle:.2f} degrees")
        return deskewed, angle

    def remove_shadows_and_enhance(self, gray: np.ndarray) -> np.ndarray:
        """
        Correct uneven lighting, remove paper shadows, and apply CLAHE contrast enhancement.
        """
        # Morphological opening to model background illumination
        kernel_size = 25
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Subtract background to normalize contrast
        diff = cv2.absdiff(gray, background)
        normalized = 255 - diff

        # Normalize intensities
        norm_img = cv2.normalize(
            normalized,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )

        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(norm_img)
        return enhanced

    def denoise(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving bilateral filtering to eliminate sensor/thermal noise.
        """
        # Bilateral filter preserves sharp text boundaries while smoothing background
        return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Binarize document image into crisp black text on pure white background.
        """
        method = self.config.binarization_method.lower()

        if method == "adaptive":
            block_size = self.config.adaptive_block_size
            if block_size % 2 == 0:
                block_size += 1
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                self.config.adaptive_c,
            )
        elif method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary
        else:
            # None or grayscale
            return gray

    def process_array(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute full preprocessing on an in-memory NumPy image array.
        Returns: (processed_image, detected_skew_angle)
        """
        # 1. Aspect ratio and dimension normalization
        resized, _ = resize_with_aspect_ratio(
            image,
            min_dimension=self.config.min_dimension,
            max_dimension=self.config.max_dimension,
        )

        # 2. Deskewing
        detected_angle = 0.0
        if self.config.enable_deskew:
            resized, detected_angle = self.deskew(resized)

        # If binarization is disabled, return clean RGB/grayscale resized image directly
        # Deep learning OCR engines (EasyOCR, PaddleOCR) perform best on natural anti-aliased images
        if self.config.binarization_method.lower() == "none":
            return resized, detected_angle

        # 3. Convert to Grayscale
        gray = convert_to_grayscale(resized)

        # 4. Shadow removal and contrast enhancement
        if self.config.enable_shadow_removal:
            enhanced = self.remove_shadows_and_enhance(gray)
        else:
            enhanced = gray

        # 5. Denoise
        if self.config.enable_denoise:
            denoised = self.denoise(enhanced)
        else:
            denoised = enhanced

        # 6. Binarize
        final_image = self.binarize(denoised)

        return final_image, detected_angle

    def preprocess_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        file_hash: Optional[str] = None,
    ) -> PreprocessingResult:
        """
        Load an image file, execute preprocessing, and persist clean result to cache.
        """
        start_time = time.perf_counter()
        in_path = Path(input_path).resolve()
        if not in_path.is_file():
            raise ImagePreprocessingError(
                f"File not found: {in_path}", file_path=str(in_path)
            )

        original_image = load_image(in_path)
        orig_shape = (original_image.shape[0], original_image.shape[1])

        processed_img, angle = self.process_array(original_image)
        proc_shape = (processed_img.shape[0], processed_img.shape[1])

        # Determine target cached file path
        if output_path:
            out_file = Path(output_path).resolve()
        else:
            stem = in_path.stem
            hash_suffix = f"_{file_hash[:8]}" if file_hash else ""
            out_file = self.settings.cache_dir / f"{stem}{hash_suffix}_clean.png"

        save_image(processed_img, out_file)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Preprocessed {in_path.name} in {elapsed_ms:.1f}ms "
            f"(angle: {angle:.2f}°, size: {proc_shape[1]}x{proc_shape[0]})"
        )

        return PreprocessingResult(
            input_path=in_path,
            cached_path=out_file,
            file_hash=file_hash,
            skew_angle=angle,
            original_shape=orig_shape,
            processed_shape=proc_shape,
            elapsed_time_ms=elapsed_ms,
        )
