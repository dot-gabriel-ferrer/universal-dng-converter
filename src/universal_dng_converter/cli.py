#!/usr/bin/env python3
"""
CLI module for Universal DNG Converter.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .converter import ImageConverter


def main() -> Optional[int]:
    """Main CLI entry point."""
    examples_lines = [
        "Examples:",
        "  universal-dng-converter --input image.fits --output ./",
        "  universal-dng-converter --input images/ --output dng_output/ --recursive",
        (
            "  universal-dng-converter --input data/ --output converted/ "
            "--bit-depth 16 --scaling percentile"
        ),
    ]
    examples = "\n".join(examples_lines)
    parser = argparse.ArgumentParser(
        description="Universal DNG Converter - Convert images to DNG format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input file or directory path"
    )

    parser.add_argument("--output", "-o", required=True, help="Output directory path")

    parser.add_argument(
        "--recursive", "-r", action="store_true", help="Process directories recursively"
    )

    parser.add_argument(
        "--bit-depth",
        choices=["8", "16"],
        default="16",
        help="Output bit depth (default: 16)",
    )

    parser.add_argument(
        "--scaling",
        choices=["auto", "linear", "percentile", "none"],
        default="auto",
        help="Scaling method (default: auto)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="TIFF compression quality 1-100 (default: 95)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--version", action="version", version="Universal DNG Converter 1.0.0"
    )

    args = parser.parse_args()

    # Create converter instance
    converter = ImageConverter()

    # Set up logging level
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    # Convert single file or batch process
    input_path = Path(args.input)
    output_path = Path(args.output)

    try:
        if input_path.is_file():
            # Single file conversion
            result = converter.convert_to_dng(
                input_path=input_path,
                output_dir=output_path,
                bit_depth=int(args.bit_depth),
                scaling_method=args.scaling,
                quality=args.quality,
            )
            if result:
                print(f"✓ Successfully converted: {input_path} -> {result}")
            else:
                print(f"✗ Failed to convert: {input_path}")
                return 1
        else:
            # Batch conversion
            results = converter.batch_convert(
                input_dir=input_path,
                output_dir=output_path,
                recursive=args.recursive,
                bit_depth=int(args.bit_depth),
                scaling_method=args.scaling,
                quality=args.quality,
            )

            success_count = sum(1 for r in results if r[1] is not None)
            total_count = len(results)

            print(
                f"Conversion completed: {success_count}/{total_count} files successful"
            )

            if success_count < total_count:
                print("Failed conversions:")
                for input_file, output_file in results:
                    if output_file is None:
                        print(f"  ✗ {input_file}")
                return 1

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
