"""Functional tests for universal_dng_converter.

Creates synthetic images (PNG, grayscale JPG, 16-bit TIFF) and validates
conversion to .dng (TIFF container) including bit depth and shape.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from universal_dng_converter import DNGImageConverter


def _make_rgb_gradient(w: int = 64, h: int = 48) -> Image.Image:
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = (xx * 255).astype(np.uint8)
    g = (yy * 255).astype(np.uint8)
    b = ((1 - xx) * 255).astype(np.uint8)
    arr = np.dstack([r, g, b])
    return Image.fromarray(arr, mode="RGB")


def _make_gray(w: int = 64, h: int = 48) -> Image.Image:
    grad = np.linspace(0, 1, w, dtype=np.float32)
    row = (grad * 255).astype(np.uint8)
    arr = np.repeat(row[None, :], h, axis=0)
    return Image.fromarray(arr, mode="L")


def _make_tiff_16bit(w: int = 32, h: int = 32) -> Image.Image:
    data = (np.random.rand(h, w) * 65535).astype(np.uint16)
    return Image.fromarray(data, mode="I;16")


def test_basic_png_and_jpg_and_tiff_conversion():
    converter = DNGImageConverter()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_dir = tmp_path / "inputs"
        output_dir = tmp_path / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create inputs
        rgb_png = input_dir / "gradient.png"
        _make_rgb_gradient().save(rgb_png)

        gray_jpg = input_dir / "gray.jpg"
        _make_gray().save(gray_jpg, quality=90)

        tiff_16 = input_dir / "rand16.tiff"
        _make_tiff_16bit().save(tiff_16)

        # Batch convert
        results = converter.batch_convert(input_dir, output_dir, recursive=False)

        # results is list[(Path, Path|None)] from wrapper - assert all succeeded
        assert all(out is not None for _, out in results), "Some conversions failed"

        # Validate output files exist and have .dng extension
        produced = {p.name for p in output_dir.glob("*.dng")}
        assert {"gradient.dng", "gray.dng", "rand16.dng"} <= produced

        # Spot check bit depths by reading via PIL / numpy
        for expected in ["gradient.dng", "gray.dng", "rand16.dng"]:
            arr = np.array(Image.open(output_dir / expected))
            assert arr.ndim in (2, 3)
            if expected == "rand16.dng":
                # Source was 16-bit; ensure preserved
                assert arr.dtype == np.uint16
            else:
                # Allow 8-bit or 16-bit depending on converter behavior
                assert arr.dtype in (np.uint8, np.uint16)
