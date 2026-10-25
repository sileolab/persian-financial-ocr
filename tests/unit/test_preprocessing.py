"""Unit tests for the image preprocessing modules."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from hesabyar.processing.image_utils import (
    convert_to_grayscale,
    load_image,
    resize_with_aspect_ratio,
    save_image,
)
from hesabyar.processing.preprocessor import (
    DocumentPreprocessor,
    PreprocessingConfig,
)


@pytest.fixture
def sample_synthetic_receipt():
    """Create a synthetic high-contrast receipt image with text-like horizontal stripes."""
    h, w = 600, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # Off-white paper background

    # Draw simulated invoice lines
    cv2.putText(
        img, "HESABYAR STORE INVOICE", (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2
    )
    cv2.putText(
        img, "Date: 1403/08/04", (30, 110),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1
    )
    cv2.putText(
        img, "Total: 1,500,000 Rials", (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2
    )
    cv2.putText(
        img, "Tracking: 987654321", (30, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1
    )

    # Add a horizontal line
    cv2.line(img, (20, 200), (380, 200), (50, 50, 50), 2)
    return img


def test_unicode_path_load_and_save(tmp_path: Path, sample_synthetic_receipt):
    """Test saving and loading an image with Persian/non-ASCII characters in filename."""
    persian_name = tmp_path / "رسید_خرید_مادر_۱۴۰۳.png"

    saved_path = save_image(sample_synthetic_receipt, persian_name)
    assert saved_path.exists()

    loaded = load_image(saved_path)
    assert loaded is not None
    assert loaded.shape[0] == sample_synthetic_receipt.shape[0]
    assert loaded.shape[1] == sample_synthetic_receipt.shape[1]


def test_grayscale_conversion(sample_synthetic_receipt):
    gray = convert_to_grayscale(sample_synthetic_receipt)
    assert len(gray.shape) == 2
    assert gray.dtype == np.uint8


def test_resize_with_aspect_ratio(sample_synthetic_receipt):
    orig_h, orig_w = sample_synthetic_receipt.shape[:2]

    # Test upscale
    resized, scale = resize_with_aspect_ratio(sample_synthetic_receipt, min_dimension=800)
    assert min(resized.shape[:2]) >= 800
    assert abs((resized.shape[1] / resized.shape[0]) - (orig_w / orig_h)) < 0.05

    # Test downscale
    large_img = np.zeros((4000, 3000, 3), dtype=np.uint8)
    downscaled, scale = resize_with_aspect_ratio(large_img, max_dimension=2000)
    assert max(downscaled.shape[:2]) <= 2000


def test_document_preprocessor_end_to_end(tmp_path: Path, sample_synthetic_receipt):
    input_file = tmp_path / "test_invoice.jpg"
    save_image(sample_synthetic_receipt, input_file)

    cache_dir = tmp_path / "cache"
    output_file = cache_dir / "clean_invoice.png"

    config = PreprocessingConfig(
        enable_deskew=True,
        enable_shadow_removal=True,
        enable_denoise=True,
        binarization_method="adaptive",
    )
    preprocessor = DocumentPreprocessor(config=config)
    result = preprocessor.preprocess_file(input_file, output_file)

    assert result.cached_path.exists()
    assert result.elapsed_time_ms > 0
    assert result.processed_shape[0] > 0

    # Load cleaned image and check binarization
    cleaned = cv2.imread(str(result.cached_path), cv2.IMREAD_GRAYSCALE)
    assert cleaned is not None
    # In binarized images, values are mostly near 0 (text) and 255 (background)
    unique_vals = np.unique(cleaned)
    assert len(unique_vals) <= 10 or 255 in unique_vals


def test_deskew_rotated_image(tmp_path: Path, sample_synthetic_receipt):
    # Rotate receipt intentionally by 10 degrees
    h, w = sample_synthetic_receipt.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 10.0, 1.0)
    rotated = cv2.warpAffine(
        sample_synthetic_receipt, rot_mat, (w, h), borderValue=(240, 240, 240)
    )

    rotated_file = tmp_path / "rotated.jpg"
    save_image(rotated, rotated_file)

    preprocessor = DocumentPreprocessor()
    result = preprocessor.preprocess_file(rotated_file)

    # The preprocessor should detect non-zero angle and deskew
    assert result.cached_path.exists()
    assert abs(result.skew_angle) > 0.1
