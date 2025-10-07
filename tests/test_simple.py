#!/usr/bin/env python3
"""
Simple test for Universal DNG Converter to verify the warning is fixed.
"""

from universal_dng_converter import DNGImageConverter


def test_converter() -> None:
    """Simple test that uses assert instead of return."""
    # Test that converter can be initialized
    converter = DNGImageConverter()
    assert converter is not None
    assert hasattr(converter, "convert_image")
    assert hasattr(converter, "convert_batch")

    print("✓ Converter initialization test passed")


if __name__ == "__main__":
    test_converter()
    print("All tests passed!")
