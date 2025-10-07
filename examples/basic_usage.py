"""
Basic usage examples for Universal DNG Converter.
"""

from pathlib import Path

from universal_dng_converter import DNGImageConverter


def example_single_file_conversion() -> None:
    """Example: Convert a single FITS file to DNG."""
    converter = DNGImageConverter()

    input_file = Path("sample_data/astronomy_image.fits")
    output_dir = Path("./output/")

    # Convert with 16-bit depth and percentile scaling
    result = converter.convert_to_dng(
        input_path=input_file,
        output_dir=output_dir,
        bit_depth=16,
        scaling_method="percentile",
        quality=95,
    )

    if result:
        print(f"Successfully converted: {input_file} -> {result}")
    else:
        print(f"Conversion failed for: {input_file}")


def example_batch_conversion() -> None:
    """Example: Batch convert all images in a directory."""
    converter = DNGImageConverter()

    input_dir = Path("sample_data/images/")
    output_dir = Path("./dng_output/")

    # Batch convert with recursive processing
    results = converter.batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=True,
        bit_depth=16,
        scaling_method="auto",
    )

    # Print results summary
    successful = sum(1 for _, output in results if output is not None)
    total = len(results)

    print(f"Conversion completed: {successful}/{total} files successful")

    # Print individual results
    for input_file, output_file in results:
        if output_file:
            print(f"✓ {input_file.name} -> {output_file.name}")
        else:
            print(f"✗ Failed: {input_file.name}")


def example_custom_scaling() -> None:
    """Example: Demonstrate different scaling methods."""
    converter = DNGImageConverter()

    input_file = Path("sample_data/high_dynamic_range.fits")
    output_dir = Path("./scaling_examples/")

    scaling_methods = ["linear", "percentile", "auto", "none"]

    for method in scaling_methods:
        # Create output filename with scaling method
        output_subdir = output_dir / method
        output_subdir.mkdir(parents=True, exist_ok=True)

        result = converter.convert_to_dng(
            input_path=input_file,
            output_dir=output_subdir,
            scaling_method=method,
            bit_depth=16,
        )

        if result:
            print(f"✓ {method} scaling: {result}")
        else:
            print(f"✗ {method} scaling failed")


def example_with_error_handling() -> None:
    """Example: Proper error handling for conversions."""
    converter = DNGImageConverter()

    test_files = [
        Path("sample_data/valid_image.fits"),
        Path("sample_data/corrupted_image.fits"),
        Path("sample_data/nonexistent.fits"),
    ]

    output_dir = Path("./error_handling_test/")

    for input_file in test_files:
        try:
            result = converter.convert_to_dng(
                input_path=input_file, output_dir=output_dir
            )

            if result:
                print(f"✓ Successfully converted: {input_file}")
            else:
                print(f"⚠ Conversion returned None for: {input_file}")

        except FileNotFoundError:
            print(f"✗ File not found: {input_file}")
        except Exception as e:
            print(f"✗ Error converting {input_file}: {e}")


if __name__ == "__main__":
    print("Universal DNG Converter Examples")
    print("=" * 40)

    print("\n1. Single file conversion:")
    example_single_file_conversion()

    print("\n2. Batch conversion:")
    example_batch_conversion()

    print("\n3. Custom scaling methods:")
    example_custom_scaling()

    print("\n4. Error handling:")
    example_with_error_handling()
