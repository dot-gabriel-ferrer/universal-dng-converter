#!/usr/bin/env python3
"""
Professional Multi-Format Image to DNG Converter

This tool converts various image formats (FITS, PNG, JPEG, TIFF, BMP, GIF, EXR)
to DNG (Digital Negative) format while preserving metadata and providing robust
scaling options.

Features:
- Support for multiple input formats: FITS, PNG, JPEG, TIFF, BMP, GIF (first frame), EXR
- Intelligent bit-depth conversion and scaling
- Metadata extraction and preservation
- Batch processing with recursive directory scanning
- Configurable output options and quality settings
- Comprehensive logging and error handling
- Optional external DNG converter integration

Usage:
    python dng_image_converter.py --input images/ --output dng_output/
    python dng_image_converter.py --input single_image.fits --output ./
    python dng_image_converter.py --input images/ --output dng/ \
        --format tiff --recursive

Requirements:
    pip install astropy pillow tifffile numpy

Optional external tools for enhanced DNG support:
    - Adobe DNG Converter
    - exiftool for advanced metadata embedding

Author: Gabriel Ferrer
License: MIT
Version: 1.0.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
from PIL import ExifTags, Image

# Optional imports for specialized formats
try:
    from astropy.io import fits

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logging.warning("astropy not available - FITS support disabled")

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dng_converter.log")],
)
logger = logging.getLogger(__name__)


class ImageConverter:
    """Professional image converter with DNG output support."""

    SUPPORTED_FORMATS = {
        ".fits": "FITS Astronomical Image",
        ".fit": "FITS Astronomical Image",
        ".fts": "FITS Astronomical Image",
        ".png": "Portable Network Graphics",
        ".jpg": "JPEG Image",
        ".jpeg": "JPEG Image",
        ".tif": "Tagged Image File Format",
        ".tiff": "Tagged Image File Format",
        ".bmp": "Bitmap Image",
        ".gif": "Graphics Interchange Format",
        ".exr": "OpenEXR High Dynamic Range",
    }

    def __init__(
        self,
        output_format: str = "dng",
        quality: int = 100,
        bit_depth: int = 16,
        scaling_method: str = "auto",
    ):
        """
        Initialize the image converter.

        Args:
            output_format: Output format ('dng', 'tiff')
            quality: Output quality (0-100)
            bit_depth: Target bit depth (8, 16)
            scaling_method: Scaling method ('auto', 'linear', 'percentile', 'none')
        """
        self.output_format = output_format
        self.quality = quality
        self.bit_depth = bit_depth
        self.scaling_method = scaling_method
        self.stats = {"converted": 0, "skipped": 0, "errors": 0}

    def _load_fits_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load FITS image and extract metadata."""
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy required for FITS support")

        with fits.open(file_path) as hdul:
            # Use primary HDU or first image HDU
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            data = hdu.data

            # Extract relevant metadata
            metadata = {
                "ORIGINAL_FORMAT": "FITS",
                "BITPIX": hdu.header.get("BITPIX", "Unknown"),
                "NAXIS": hdu.header.get("NAXIS", "Unknown"),
                "NAXIS1": hdu.header.get("NAXIS1", "Unknown"),
                "NAXIS2": hdu.header.get("NAXIS2", "Unknown"),
                "OBJECT": hdu.header.get("OBJECT", ""),
                "TELESCOPE": hdu.header.get("TELESCOP", ""),
                "INSTRUMENT": hdu.header.get("INSTRUME", ""),
                "OBSERVER": hdu.header.get("OBSERVER", ""),
                "DATE-OBS": hdu.header.get("DATE-OBS", ""),
                "EXPTIME": hdu.header.get("EXPTIME", ""),
                "FILTER": hdu.header.get("FILTER", ""),
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

        return data, metadata

    def _load_standard_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load standard image formats and extract EXIF metadata."""
        # Load with PIL for metadata
        with Image.open(file_path) as img:
            metadata = {"ORIGINAL_FORMAT": img.format or "Unknown"}

            # Extract EXIF data
            if hasattr(img, "_getexif") and img._getexif():
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[f"EXIF_{tag}"] = str(value)

            # Convert to numpy array
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    # Convert RGBA to RGB
                    img = img.convert("RGB")
                elif img.mode in ["L", "P"]:
                    # Convert grayscale/palette to RGB
                    img = img.convert("RGB")

            data = np.array(img)

        return data, metadata

    def _load_exr_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load EXR image using OpenCV if available."""
        if not OPENCV_AVAILABLE:
            raise ImportError("opencv-python required for EXR support")

        data = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if data is None:
            raise ValueError(f"Could not load EXR file: {file_path}")

        # Convert BGR to RGB if color image
        if len(data.shape) == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        metadata = {"ORIGINAL_FORMAT": "EXR", "HDR": "True"}
        return data, metadata

    def _scale_image_data(self, data: np.ndarray) -> np.ndarray:
        """Scale image data to target bit depth with robust methods."""
        if self.scaling_method == "none":
            return data

        # Handle different input data types
        if data.dtype == np.uint8:
            data = data.astype(np.float64)
        elif data.dtype == np.uint16:
            data = data.astype(np.float64) / 65535.0
        elif data.dtype in [np.float32, np.float64]:
            data = data.astype(np.float64)
        else:
            # Convert other types to float
            data = data.astype(np.float64)

            # Apply scaling method
            if self.scaling_method == "auto":
                # Intelligent auto-scaling
                if np.all(data >= 0) and np.all(data <= 1):
                    pass  # Already normalized
                elif np.min(data) >= 0 and np.max(data) <= 255:
                    data = data / 255.0
                elif np.min(data) >= 0 and np.max(data) <= 65535:
                    data = data / 65535.0
                else:
                    from typing import Tuple as _TupleFloat

                    p_bounds_auto: _TupleFloat[float, float] = np.percentile(
                        data, [0.1, 99.9]
                    )
                    p_low, p_high = float(p_bounds_auto[0]), float(p_bounds_auto[1])
                    if p_high > p_low:
                        data = np.clip((data - p_low) / (p_high - p_low), 0, 1)
                    else:
                        data = np.zeros_like(data)
            elif self.scaling_method == "linear":
                data_min, data_max = np.min(data), np.max(data)
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)
                else:
                    data = np.zeros_like(data)
            elif self.scaling_method == "percentile":
                from typing import Tuple as _TupleFloat2

                p_bounds: _TupleFloat2[float, float] = np.percentile(data, [1, 99])
                p_low, p_high = float(p_bounds[0]), float(p_bounds[1])
                if p_high > p_low:
                    data = np.clip((data - p_low) / (p_high - p_low), 0, 1)
                else:
                    data = np.zeros_like(data)

        # Scale to target bit depth
        if self.bit_depth == 8:
            return (data * 255).astype(np.uint8)
        elif self.bit_depth == 16:
            return (data * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")

    def _write_dng_tiff(
        self, data: np.ndarray, output_path: str, metadata: Dict[str, Any]
    ) -> None:
        """Write DNG-compatible TIFF with metadata."""
        # Prepare TIFF tags for DNG compatibility
        tiff_tags = {
            "Software": "DNG Image Converter v1.0.0",
            "Artist": "Gabriel Ferrer",
            "ImageDescription": (
                f"Converted from {metadata.get('ORIGINAL_FORMAT', 'Unknown')}"
            ),
        }

        # Add custom metadata as TIFF tags
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)) and len(str(value)) < 100:
                tiff_tags[f"CustomTag_{key}"] = str(value)

        # Write TIFF file
        if self.output_format == "dng":
            # Use .dng extension for DNG files
            output_path = output_path.replace(".tiff", ".dng").replace(".tif", ".dng")

        tifffile.imwrite(
            output_path,
            data,
            photometric="rgb" if len(data.shape) == 3 else "minisblack",
            compression="lzw",
            metadata=tiff_tags,
            resolution=(300, 300),  # 300 DPI
            resolutionunit="inch",
        )

        logger.info(f"Written {self.output_format.upper()} file: {output_path}")

    def convert_image(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> bool:
        """
        Convert a single image file to DNG format.

        Args:
            input_path: Path to input image
            output_path: Path for output DNG file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)

            # Determine file type and load accordingly
            extension = input_path.suffix.lower()

            if extension not in self.SUPPORTED_FORMATS:
                logger.warning(f"Unsupported format: {extension}")
                self.stats["skipped"] += 1
                return False

            logger.info(f"Converting {input_path} -> {output_path}")

            # Load image data and metadata
            if extension in [".fits", ".fit", ".fts"]:
                data, metadata = self._load_fits_image(str(input_path))
            elif extension == ".exr":
                data, metadata = self._load_exr_image(str(input_path))
            else:
                data, metadata = self._load_standard_image(str(input_path))

            # Handle different array dimensions
            if data.ndim == 2:
                # Grayscale - keep as is
                pass
            elif data.ndim == 3:
                # Color image - ensure RGB order
                if data.shape[2] > 3:
                    # Remove alpha channel if present
                    data = data[:, :, :3]
            else:
                raise ValueError(f"Unsupported data dimensions: {data.shape}")

            # Scale image data
            scaled_data = self._scale_image_data(data)

            # Write DNG/TIFF output
            self._write_dng_tiff(scaled_data, str(output_path), metadata)

            self.stats["converted"] += 1
            return True

        except Exception as e:
            logger.error(f"Error converting {input_path}: {str(e)}")
            self.stats["errors"] += 1
            return False

    def convert_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
    ) -> Dict[str, int]:
        """
        Convert multiple images in batch mode.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            recursive: Search subdirectories recursively

        Returns:
            Dict[str,int]: Conversion statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all supported image files
        pattern = "**/*" if recursive else "*"
        all_files: List[Path] = []

        for ext in self.SUPPORTED_FORMATS.keys():
            all_files.extend(input_dir.glob(f"{pattern}{ext}"))
            all_files.extend(input_dir.glob(f"{pattern}{ext.upper()}"))

        if not all_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return self.stats

        logger.info(f"Found {len(all_files)} image files to convert")

        # Convert each file
        for input_file in sorted(all_files):
            # Generate output filename
            output_file = output_dir / f"{input_file.stem}.{self.output_format}"

            # Maintain directory structure if recursive
            if recursive:
                rel_path = input_file.relative_to(input_dir)
                output_file = (
                    output_dir
                    / rel_path.parent
                    / f"{input_file.stem}.{self.output_format}"
                )
                output_file.parent.mkdir(parents=True, exist_ok=True)

            self.convert_image(str(input_file), str(output_file))

        return self.stats

    # ---------------- Backwards compatibility wrappers -----------------
    def convert_to_dng(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        bit_depth: Optional[int] = None,
        scaling_method: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> Optional[Path]:
        """Legacy wrapper returning output file Path or None."""
        in_path = Path(input_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prev_bit, prev_scaling, prev_quality = (
            self.bit_depth,
            self.scaling_method,
            self.quality,
        )
        try:
            if bit_depth is not None:
                self.bit_depth = bit_depth
            if scaling_method is not None:
                self.scaling_method = scaling_method
            if quality is not None:
                self.quality = quality
            out_file = out_dir / f"{in_path.stem}.{self.output_format}"
            ok = self.convert_image(in_path, out_file)
            return out_file if ok else None
        finally:
            self.bit_depth, self.scaling_method, self.quality = (
                prev_bit,
                prev_scaling,
                prev_quality,
            )

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
        bit_depth: Optional[int] = None,
        scaling_method: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> List[Tuple[Path, Optional[Path]]]:
        """Legacy batch wrapper returning per-file tuples (input, output|None)."""
        prev_bit, prev_scaling, prev_quality = (
            self.bit_depth,
            self.scaling_method,
            self.quality,
        )
        try:
            if bit_depth is not None:
                self.bit_depth = bit_depth
            if scaling_method is not None:
                self.scaling_method = scaling_method
            if quality is not None:
                self.quality = quality
            input_dir_p = Path(input_dir)
            output_dir_p = Path(output_dir)
            pattern = "**/*" if recursive else "*"
            files: List[Path] = []
            for ext in self.SUPPORTED_FORMATS.keys():
                files.extend(input_dir_p.glob(f"{pattern}{ext}"))
                files.extend(input_dir_p.glob(f"{pattern}{ext.upper()}"))
            results: List[Tuple[Path, Optional[Path]]] = []
            for f in sorted(files):
                if recursive:
                    rel = f.relative_to(input_dir_p)
                    out_parent = output_dir_p / rel.parent
                else:
                    out_parent = output_dir_p
                out_parent.mkdir(parents=True, exist_ok=True)
                out_file = out_parent / f"{f.stem}.{self.output_format}"
                ok = self.convert_image(f, out_file)
                results.append((f, out_file if ok else None))
            return results
        finally:
            self.bit_depth, self.scaling_method, self.quality = (
                prev_bit,
                prev_scaling,
                prev_quality,
            )


def main() -> None:
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Professional Multi-Format Image to DNG Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input images/ --output dng_output/
  %(prog)s --input single_image.fits --output ./
  %(prog)s --input images/ --output dng/ --format tiff --recursive
  %(prog)s --input data/ --scaling percentile --bit-depth 8
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file or directory path"
    )

    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory path"
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["dng", "tiff"],
        default="dng",
        help="Output format (default: dng)",
    )

    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Process subdirectories recursively",
    )

    parser.add_argument(
        "--scaling",
        choices=["auto", "linear", "percentile", "none"],
        default="auto",
        help="Data scaling method (default: auto)",
    )

    parser.add_argument(
        "--bit-depth",
        choices=[8, 16],
        type=int,
        default=16,
        help="Target bit depth (default: 16)",
    )

    parser.add_argument(
        "--quality", type=int, default=100, help="Output quality 0-100 (default: 100)"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Create converter instance
    converter = ImageConverter(
        output_format=args.format,
        quality=args.quality,
        bit_depth=args.bit_depth,
        scaling_method=args.scaling,
    )

    # Log available format support
    logger.info("Supported formats:")
    for ext, desc in converter.SUPPORTED_FORMATS.items():
        available = "✓"
        if ext in [".fits", ".fit", ".fts"] and not ASTROPY_AVAILABLE:
            available = "✗ (requires astropy)"
        elif ext == ".exr" and not OPENCV_AVAILABLE:
            available = "✗ (requires opencv-python)"
        logger.info(f"  {ext}: {desc} {available}")

    # Process input
    if input_path.is_file():
        # Single file conversion
        output_file = Path(args.output) / f"{input_path.stem}.{args.format}"
        Path(args.output).mkdir(parents=True, exist_ok=True)
        converter.convert_image(str(input_path), str(output_file))
    else:
        # Batch conversion
        converter.convert_batch(str(input_path), args.output, args.recursive)

    # Print final statistics
    stats = converter.stats
    logger.info("Conversion Summary:")
    logger.info(f"  Converted: {stats['converted']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Errors: {stats['errors']}")

    if stats["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
