"""Parametrized tests converting real sample images in test_images/.

Verifies each supported format converts successfully to DNG and basic
properties (dimensions, dtype) are sane. FITS images may have different
dynamic ranges; we only assert output existence and shape > 0.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from universal_dng_converter import DNGImageConverter

SAMPLES = [
    "test_16bit.tiff",
    "test_color.png",
    "test_gradient.fits",
    "test_grayscale.jpg",
    "test_stars.fits",
]


@pytest.mark.parametrize("filename", SAMPLES)
def test_convert_real_sample(filename: str):
    root = Path("test_images")
    src = root / filename
    assert src.exists(), f"Missing sample file: {src}"

    converter = DNGImageConverter()

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        result = converter.convert_to_dng(src, out_dir)
        assert result is not None, "Conversion returned None"
        assert result.exists(), "Output file missing"
        assert result.suffix.lower() == ".dng"

        # Open DNG (TIFF) and check array integrity
        with tifffile.TiffFile(result) as tif:
            try:
                arr = tif.asarray()
            except Exception as e:  # pragma: no cover
                pytest.fail(f"Failed reading TIFF: {e}")

        # Basic structural assertions
        assert arr.size > 0, "Empty image data"
        assert arr.ndim in (2, 3), f"Unexpected ndim {arr.ndim}"
        assert arr.dtype in (np.uint8, np.uint16), f"Unexpected dtype {arr.dtype}"

        # Additional expectations per file (lightweight)
        if filename == "test_16bit.tiff":
            # Should preserve 16-bit depth typically
            assert arr.dtype == np.uint16
        if filename == "test_color.png":
            assert arr.ndim == 3 and arr.shape[2] in (3, 4)
        if filename.endswith(".fits"):
            # FITS often scaled to full 16-bit
            assert arr.dtype == np.uint16
