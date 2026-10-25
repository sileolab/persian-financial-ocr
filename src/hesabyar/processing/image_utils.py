"""Utility functions for loading, saving, and manipulating images with Unicode/Windows path support."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from hesabyar.core.exceptions import ImagePreprocessingError
from hesabyar.core.logger import get_logger

logger = get_logger("processing.image_utils")


def load_image(
    file_path: Union[str, Path], as_gray: bool = False
) -> np.ndarray:
    """
    Safely load an image from disk with full Unicode and Persian/Windows path support.

    Standard cv2.imread fails on non-ASCII paths in Windows; this implementation reads
    raw bytes and decodes them with cv2.imdecode or PIL fallback.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise ImagePreprocessingError(
            f"Image file does not exist: {path}", file_path=str(path)
        )

    try:
        # Read raw bytes to avoid Windows path encoding limitations
        with open(path, "rb") as f:
            file_bytes = bytearray(f.read())

        np_array = np.asarray(file_bytes, dtype=np.uint8)
        flags = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        image = cv2.imdecode(np_array, flags)

        if image is None:
            # Fallback using PIL for unusual color profiles / TIFF
            with Image.open(path) as pil_img:
                if as_gray:
                    pil_img = pil_img.convert("L")
                    image = np.array(pil_img)
                else:
                    pil_img = pil_img.convert("RGB")
                    rgb_array = np.array(pil_img)
                    image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        if image is None or image.size == 0:
            raise ImagePreprocessingError(
                f"Failed to decode image: {path}", file_path=str(path)
            )

        return image

    except Exception as e:
        if isinstance(e, ImagePreprocessingError):
            raise
        raise ImagePreprocessingError(
            f"Error loading image: {e}", file_path=str(path)
        ) from e


def save_image(
    image: np.ndarray,
    output_path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Safely save an image to disk with Unicode/Windows path support.
    """
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if not ext:
        ext = ".jpg"
        path = path.with_suffix(".jpg")

    encode_params = []
    if ext in [".jpg", ".jpeg"]:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    elif ext == ".png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 4]

    try:
        success, encoded_buf = cv2.imencode(ext, image, encode_params)
        if not success:
            raise ImagePreprocessingError(
                f"cv2.imencode failed for extension {ext}", file_path=str(path)
            )

        with open(path, "wb") as f:
            f.write(encoded_buf.tobytes())

        return path
    except Exception as e:
        if isinstance(e, ImagePreprocessingError):
            raise
        raise ImagePreprocessingError(
            f"Error saving image: {e}", file_path=str(path)
        ) from e


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an image (BGR or BGRA) to 8-bit single channel grayscale."""
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    min_dimension: int = 800,
    max_dimension: int = 2600,
) -> Tuple[np.ndarray, float]:
    """
    Resize image while preserving aspect ratio.
    Also ensures the dimensions stay between min_dimension and max_dimension for optimal OCR.
    Returns (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    scale = 1.0

    if target_width is not None and target_height is not None:
        scale = min(target_width / w, target_height / h)
    elif target_width is not None:
        scale = target_width / w
    elif target_height is not None:
        scale = target_height / h
    else:
        # Automatic bounds enforcement
        max_side = max(h, w)
        min_side = min(h, w)

        if max_side > max_dimension:
            scale = max_dimension / max_side
        elif min_side < min_dimension and min_side > 0:
            scale = min_dimension / min_side

    if abs(scale - 1.0) < 1e-3:
        return image, 1.0

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Use INTER_CUBIC / INTER_AREA depending on upscale or downscale
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, scale
