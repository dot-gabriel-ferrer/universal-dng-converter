#!/usr/bin/env python3
"""
Test script for the DNG Image Converter.
Creates sample images and tests the converter functionality.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image


def create_test_images() -> Path:
    """Create sample test images in various formats."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)

    # Create a simple gradient pattern
    height, width = 256, 256
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    # Test image 1: Simple gradient
    gradient = (x + y) / 2

    # Test image 2: Star field simulation
    stars = np.zeros((height, width))
    np.random.seed(42)
    star_positions = np.random.randint(0, height, (50, 2))
    for pos in star_positions:
        y_pos, x_pos = pos
        if 10 < y_pos < height - 10 and 10 < x_pos < width - 10:
            # Create a simple star with Gaussian profile
            yy, xx = np.meshgrid(
                range(y_pos - 5, y_pos + 6), range(x_pos - 5, x_pos + 6), indexing="ij"
            )
            dist = np.sqrt((yy - y_pos) ** 2 + (xx - x_pos) ** 2)
            star_profile = np.exp(-(dist**2) / 2)
            stars[y_pos - 5 : y_pos + 6, x_pos - 5 : x_pos + 6] += star_profile

    stars = np.clip(stars, 0, 1)

    # Create FITS file
    print("Creating test FITS file...")
    fits_data = (gradient * 65535).astype(np.uint16)
    hdu = fits.PrimaryHDU(fits_data)
    hdu.header["OBJECT"] = "Test Gradient"
    hdu.header["TELESCOP"] = "Test Telescope"
    hdu.header["INSTRUME"] = "Test Camera"
    hdu.header["DATE-OBS"] = "2025-01-01T12:00:00"
    hdu.header["EXPTIME"] = 300.0
    hdu.header["FILTER"] = "V"
    hdu.writeto(test_dir / "test_gradient.fits", overwrite=True)

    # Create another FITS file with star field
    fits_stars = (stars * 65535).astype(np.uint16)
    hdu_stars = fits.PrimaryHDU(fits_stars)
    hdu_stars.header["OBJECT"] = "Test Star Field"
    hdu_stars.header["TELESCOP"] = "Test Telescope"
    hdu_stars.header["INSTRUME"] = "Test Camera"
    hdu_stars.header["DATE-OBS"] = "2025-01-01T12:30:00"
    hdu_stars.header["EXPTIME"] = 600.0
    hdu_stars.header["FILTER"] = "R"
    hdu_stars.writeto(test_dir / "test_stars.fits", overwrite=True)

    # Create PNG file
    print("Creating test PNG file...")
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = (gradient * 255).astype(np.uint8)  # Red channel
    rgb_image[:, :, 1] = (stars * 255).astype(np.uint8)  # Green channel
    rgb_image[:, :, 2] = ((1 - gradient) * 255).astype(np.uint8)  # Blue channel

    Image.fromarray(rgb_image).save(test_dir / "test_color.png")

    # Create JPEG file
    print("Creating test JPEG file...")
    jpeg_data = ((gradient + stars) / 2 * 255).astype(np.uint8)
    jpeg_rgb = np.stack([jpeg_data, jpeg_data, jpeg_data], axis=2)
    Image.fromarray(jpeg_rgb).save(test_dir / "test_grayscale.jpg", quality=95)

    # Create TIFF file
    print("Creating test TIFF file...")
    tiff_data = (gradient * 65535).astype(np.uint16)
    Image.fromarray(tiff_data, mode="I;16").save(test_dir / "test_16bit.tiff")

    print(f"Created test images in {test_dir}/")
    return test_dir


def test_converter() -> bool:
    """Test the DNG converter with sample images."""
    test_dir = create_test_images()
    output_dir = Path("test_dng_output")

    converter_script = Path("scripts/convert-to-dng")

    if not converter_script.exists():
        print(f"Error: Converter script not found at {converter_script}")
        return False

    # Test single file conversion
    print("\n" + "=" * 50)
    print("Testing single file conversion...")
    print("=" * 50)

    cmd = [
        sys.executable,
        str(converter_script),
        "--input",
        str(test_dir / "test_gradient.fits"),
        "--output",
        str(output_dir),
        "--verbose",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error in single file conversion: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    # Test batch conversion
    print("\n" + "=" * 50)
    print("Testing batch conversion...")
    print("=" * 50)

    cmd = [
        sys.executable,
        str(converter_script),
        "--input",
        str(test_dir),
        "--output",
        str(output_dir / "batch"),
        "--format",
        "dng",
        "--scaling",
        "auto",
        "--bit-depth",
        "16",
        "--verbose",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error in batch conversion: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

    # Check output files
    print("\n" + "=" * 50)
    print("Checking output files...")
    print("=" * 50)

    expected_files = [
        output_dir / "test_gradient.dng",
        output_dir / "batch" / "test_gradient.dng",
        output_dir / "batch" / "test_stars.dng",
        output_dir / "batch" / "test_color.dng",
        output_dir / "batch" / "test_grayscale.dng",
        output_dir / "batch" / "test_16bit.dng",
    ]

    success = True
    for expected_file in expected_files:
        if expected_file.exists():
            size = expected_file.stat().st_size
            print(f"✓ {expected_file} ({size} bytes)")
        else:
            print(f"✗ Missing: {expected_file}")
            success = False

    return success


if __name__ == "__main__":
    print("DNG Image Converter Test Suite")
    print("=" * 50)

    success = test_converter()

    if success:
        print("\n🎉 All tests passed!\nManual test examples:")
        print("  scripts/convert-to-dng --input test_images/ --output manual_test/")
        print("  scripts/convert-to-dng --help")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
