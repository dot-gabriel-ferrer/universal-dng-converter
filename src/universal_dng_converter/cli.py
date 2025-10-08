#!/usr/bin/env python3
"""
CLI module for Universal DNG Converter.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

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

    parser.add_argument("--input", "-i", help="Input file or directory path")

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
        "--raw-compatible",
        action="store_true",
        help=(
            "Create RAW-compatible DNG files "
            "(readable by rawpy and similar libraries)"
        ),
    )

    parser.add_argument(
        "--fix-raw-errors",
        metavar="DNG_FILE",
        help="Fix RAW compatibility issues in an existing DNG file",
    )

    parser.add_argument(
        "--fallback-format",
        choices=["tiff", "png", "jpg"],
        default="tiff",
        help="Fallback format when RAW processing fails (default: tiff)",
    )

    parser.add_argument(
        "--validate-raw",
        action="store_true",
        help=("Validate RAW compatibility of generated DNG files " "(requires rawpy)"),
    )

    parser.add_argument(
        "--pseudo-raw",
        action="store_true",
        help="Create pseudo-RAW DNG files that mimic sensor characteristics",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--version", action="version", version="Universal DNG Converter 1.0.0"
    )

    args = parser.parse_args()

    # Validate input requirements
    if not args.fix_raw_errors and not args.input:
        parser.error("--input/-i is required unless using --fix-raw-errors")

    # Handle special case: fixing RAW errors in existing DNG
    if args.fix_raw_errors:
        converter = ImageConverter(raw_output=True)

        if args.verbose:
            import logging

            logging.basicConfig(level=logging.INFO)

        try:
            dng_path = Path(args.fix_raw_errors)
            if not dng_path.exists():
                print(f"✗ DNG file not found: {dng_path}")
                return 1

            output_path = Path(args.output)
            result = converter.handle_raw_processing_error(
                dng_path, output_path, args.fallback_format
            )

            if result:
                print(f"✓ Successfully fixed RAW compatibility: {dng_path} -> {result}")
                return 0
            else:
                print(f"✗ Failed to fix RAW compatibility: {dng_path}")
                return 1

        except Exception as e:
            print(f"Error fixing RAW compatibility: {e}")
            return 1

    # Create converter instance with RAW support
    converter = ImageConverter(
        raw_output=args.raw_compatible if hasattr(args, "raw_compatible") else False
    )

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
            if args.pseudo_raw:
                # Use pseudo-RAW mode
                result = converter.create_pseudo_raw_dng(
                    input_path=input_path,
                    output_dir=output_path,
                    simulate_bayer=True,
                )
            elif args.raw_compatible:
                result = converter.convert_to_raw_compatible_dng(
                    input_path=input_path,
                    output_dir=output_path,
                    bit_depth=int(args.bit_depth),
                    scaling_method=args.scaling,
                    quality=args.quality,
                    validate_raw=args.validate_raw,
                )
            else:
                result = converter.convert_to_dng(
                    input_path=input_path,
                    output_dir=output_path,
                    bit_depth=int(args.bit_depth),
                    scaling_method=args.scaling,
                    quality=args.quality,
                )
            if result:
                print(f"✓ Successfully converted: {input_path} -> {result}")
                if args.raw_compatible:
                    print("  Created RAW-compatible DNG file")
                elif args.pseudo_raw:
                    print(
                        "  Created pseudo-RAW DNG file with sensor-like characteristics"
                    )
            else:
                print(f"✗ Failed to convert: {input_path}")
                return 1
        else:
            # Batch conversion
            if args.pseudo_raw:
                # For pseudo-RAW, process files individually
                input_path = Path(args.input)
                output_path = Path(args.output)

                pattern = "**/*" if args.recursive else "*"
                files: List[Path] = []
                for ext in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tiff",
                    ".tif",
                    ".fits",
                    ".fit",
                    ".fts",
                ]:
                    files.extend(input_path.glob(f"{pattern}{ext}"))
                    files.extend(input_path.glob(f"{pattern}{ext.upper()}"))

                results = []
                for file_path in sorted(files):
                    result = converter.create_pseudo_raw_dng(file_path, output_path)
                    results.append((file_path, result))

                success_count = sum(1 for _, r in results if r is not None)
                total_count = len(results)

                print(
                    f"Conversion completed: {success_count}/{total_count} "
                    f"files successful"
                )
                print("  Used pseudo-RAW DNG format with sensor simulation")

                if success_count < total_count:
                    print("Failed conversions:")
                    for file_path, result in results:
                        if result is None:
                            print(f"  ✗ {file_path}")
                    return 1
            else:
                # Standard batch conversion
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
                    f"Conversion completed: {success_count}/{total_count} "
                    f"files successful"
                )
                if args.raw_compatible:
                    print("  Used RAW-compatible DNG format")

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
