#!/usr/bin/env python3
"""
Example script demonstrating RAW-compatible DNG conversion.

This script shows how to use the new RAW compatibility features
to create DNG files that can be read by rawpy and other RAW processors.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from universal_dng_converter import ImageConverter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate RAW-compatible DNG conversion."""
    print("Universal DNG Converter - RAW Compatibility Example")
    print("=" * 50)

    # Create a converter with RAW output enabled
    converter = ImageConverter(raw_output=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Example 1: Create a test image and convert to RAW-compatible DNG
        print("\n1. Creating RAW-compatible DNG from synthetic image...")

        # Create a test 16-bit image simulating sensor data
        test_image = create_synthetic_sensor_data()
        input_path = tmp_path / "synthetic_sensor.tiff"
        tifffile.imwrite(input_path, test_image)

        # Convert to RAW-compatible DNG
        output_dir = tmp_path / "raw_output"
        result = converter.convert_to_raw_compatible_dng(
            input_path=input_path,
            output_dir=output_dir,
            bit_depth=16,
            scaling_method="none",  # Preserve original values for RAW
            validate_raw=True,
        )

        if result:
            print(f"✓ Created RAW-compatible DNG: {result}")
            print(f"  File size: {result.stat().st_size / 1024:.1f} KB")

            # Try to validate with rawpy if available
            validate_with_rawpy(result)
        else:
            print("✗ Failed to create RAW-compatible DNG")

        # Example 2: Handle existing DNG with compatibility issues
        print("\n2. Demonstrating error handling for problematic DNG...")

        # Create a standard (non-RAW) DNG file
        standard_dng = tmp_path / "standard.dng"
        converter_standard = ImageConverter(raw_output=False)
        converter_standard.convert_image(input_path, standard_dng)

        # Try to fix it for RAW compatibility
        fixed_output = tmp_path / "fixed_output"
        fixed_result = converter.handle_raw_processing_error(
            dng_path=standard_dng, output_dir=fixed_output, fallback_format="tiff"
        )

        if fixed_result:
            print(f"✓ Fixed DNG compatibility: {fixed_result}")
        else:
            print("✗ Failed to fix DNG compatibility")

        # Example 3: Compare different output formats
        print("\n3. Comparing standard vs RAW-compatible DNG...")

        compare_output_formats(converter, input_path, tmp_path)

        print("\nExample completed!")


def create_synthetic_sensor_data():
    """Create synthetic sensor data similar to what a camera would produce."""
    # Create a 16-bit image with realistic sensor characteristics
    height, width = 200, 300

    # Simulate Bayer pattern data (RGGB)
    image = np.zeros((height, width), dtype=np.uint16)

    # Add some realistic sensor values
    base_level = 512  # Black level
    white_level = 16383  # White level (14-bit sensor)

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Red pixels (top-left of each 2x2 block)
    red_mask = (y_coords % 2 == 0) & (x_coords % 2 == 0)
    image[red_mask] = base_level + (
        x_coords[red_mask] / width * (white_level - base_level) * 0.8
    ).astype(np.uint16)

    # Green pixels (top-right and bottom-left of each 2x2 block)
    green_mask = ((y_coords % 2 == 0) & (x_coords % 2 == 1)) | (
        (y_coords % 2 == 1) & (x_coords % 2 == 0)
    )
    image[green_mask] = base_level + (
        y_coords[green_mask] / height * (white_level - base_level) * 0.9
    ).astype(np.uint16)

    # Blue pixels (bottom-right of each 2x2 block)
    blue_mask = (y_coords % 2 == 1) & (x_coords % 2 == 1)
    image[blue_mask] = base_level + (
        (x_coords[blue_mask] + y_coords[blue_mask])
        / (width + height)
        * (white_level - base_level)
        * 0.7
    ).astype(np.uint16)

    # Add some noise
    noise = np.random.normal(0, 10, (height, width)).astype(np.int16)
    image = np.clip(image.astype(np.int32) + noise, 0, 65535).astype(np.uint16)

    return image


def validate_with_rawpy(dng_path):
    """Validate DNG file with rawpy if available."""
    try:
        import rawpy

        print(f"  Validating with rawpy...")
        with rawpy.imread(str(dng_path)) as raw:
            print(f"    ✓ rawpy can read the file")
            print(f"    Image size: {raw.raw_image.shape}")
            print(f"    Data type: {raw.raw_image.dtype}")
            print(f"    Camera model: {getattr(raw, 'camera_model', 'Unknown')}")
            print(f"    White level: {getattr(raw, 'white_level', 'Unknown')}")

    except ImportError:
        print("  ⚠ rawpy not available for validation")
    except Exception as e:
        print(f"  ⚠ rawpy validation failed: {e}")


def compare_output_formats(converter, input_path, output_dir):
    """Compare different output format configurations."""
    formats_to_test = [
        ("standard_dng", {"raw_output": False, "output_format": "dng"}),
        ("raw_dng", {"raw_output": True, "output_format": "dng"}),
        ("tiff", {"raw_output": False, "output_format": "tiff"}),
    ]

    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    print("  Format comparison results:")

    for name, settings in formats_to_test:
        test_converter = ImageConverter(**settings)
        output_path = comparison_dir / f"test_{name}.{settings['output_format']}"

        success = test_converter.convert_image(input_path, output_path)

        if success and output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"    {name:12}: {size_kb:6.1f} KB")

            # Try to read with tifffile
            try:
                with tifffile.TiffFile(output_path) as tif:
                    tags = tif.pages[0].tags
                    compression = (
                        tags.get("Compression", {}).value
                        if "Compression" in tags
                        else "Unknown"
                    )
                    photometric = (
                        tags.get("PhotometricInterpretation", {}).value
                        if "PhotometricInterpretation" in tags
                        else "Unknown"
                    )
                    print(
                        f"                  Compression: {compression}, Photometric: {photometric}"
                    )
            except Exception as e:
                print(f"                  Error reading: {e}")
        else:
            print(f"    {name:12}: Failed")


if __name__ == "__main__":
    main()
