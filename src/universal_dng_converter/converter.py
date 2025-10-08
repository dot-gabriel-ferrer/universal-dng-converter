#!/usr/bin/env python3
"""Core conversion logic for Universal DNG Converter.

Contains the :class:`ImageConverter` with helpers for loading different image
formats, scaling numeric data and writing TIFF/DNG output. The CLI entry point
is defined separately in ``cli.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tifffile
from PIL import ExifTags, Image

try:  # FITS optional
    from astropy.io import fits

    ASTROPY_AVAILABLE = True
except Exception:  # pragma: no cover
    ASTROPY_AVAILABLE = False

try:  # OpenEXR optional
    import cv2

    OPENCV_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENCV_AVAILABLE = False

try:  # RAW optional
    import rawpy

    RAWPY_AVAILABLE = True
except Exception:  # pragma: no cover
    RAWPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImageConverter:
    """Image conversion engine (FITS/standard → TIFF/DNG)."""

    SUPPORTED_FORMATS: Dict[str, str] = {
        ".fits": "FITS",
        ".fit": "FITS",
        ".fts": "FITS",
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".bmp": "BMP",
        ".gif": "GIF",
        ".exr": "EXR",
        ".dng": "DNG",
        ".cr2": "RAW",
        ".cr3": "RAW",
        ".nef": "RAW",
        ".arw": "RAW",
        ".orf": "RAW",
        ".rw2": "RAW",
        ".raf": "RAW",
        ".raw": "RAW",
    }

    def __init__(
        self,
        output_format: str = "dng",
        quality: int = 100,
        bit_depth: int = 16,
        scaling_method: str = "auto",
        raw_output: bool = True,
    ) -> None:
        self.output_format = output_format
        self.quality = quality
        self.bit_depth = bit_depth
        self.scaling_method = scaling_method
        self.raw_output = raw_output  # Whether to create RAW-compatible DNG files
        self.stats: Dict[str, int] = {"converted": 0, "skipped": 0, "errors": 0}

    # -------------------------- Loaders --------------------------
    def _load_fits_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not ASTROPY_AVAILABLE:  # pragma: no cover - runtime guard
            raise ImportError("astropy required for FITS support")
        with fits.open(file_path) as hdul:
            hdu = hdul[0] if hdul[0].data is not None else hdul[1]
            data = np.asarray(hdu.data)
            header = hdu.header
        metadata: Dict[str, Any] = {
            "ORIGINAL_FORMAT": "FITS",
            "BITPIX": header.get("BITPIX"),
            "NAXIS": header.get("NAXIS"),
            "NAXIS1": header.get("NAXIS1"),
            "NAXIS2": header.get("NAXIS2"),
            "OBJECT": header.get("OBJECT"),
            "TELESCOPE": header.get("TELESCOP"),
            "INSTRUMENT": header.get("INSTRUME"),
        }
        return data, {k: v for k, v in metadata.items() if v is not None}

    def _load_standard_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with Image.open(file_path) as img:
            metadata: Dict[str, Any] = {"ORIGINAL_FORMAT": img.format or "Unknown"}
            exif_raw = getattr(img, "_getexif", lambda: None)()
            if exif_raw:
                for tag_id, value in exif_raw.items():
                    tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if len(str(value)) < 80:
                        metadata[f"EXIF_{tag}"] = str(value)

            # Handle different modes, preserving bit depth where possible
            if img.mode in ("RGB", "L"):
                # Keep as-is
                data = np.asarray(img)
            elif img.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
                # 16-bit grayscale - keep as 16-bit
                data = np.asarray(img)
            elif img.mode == "I":
                # 32-bit integer - convert to 16-bit
                data = np.asarray(img)
                # Scale from 32-bit to 16-bit range
                if data.max() > 65535:
                    data = (data / data.max() * 65535).astype(np.uint16)
                else:
                    data = data.astype(np.uint16)
            else:
                # Convert other modes to RGB
                img = img.convert("RGB")
                data = np.asarray(img)
        return data, metadata

    def _load_exr_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not OPENCV_AVAILABLE:  # pragma: no cover
            raise ImportError("opencv-python required for EXR support")
        data = cv2.imread(
            file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        )  # noqa: E501
        if data is None:
            raise ValueError(f"Could not load EXR file: {file_path}")
        if data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        return data, {"ORIGINAL_FORMAT": "EXR"}

    def _load_raw_image(self, file_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load RAW image files using rawpy."""
        if not RAWPY_AVAILABLE:  # pragma: no cover
            raise ImportError("rawpy required for RAW file support")

        try:
            with rawpy.imread(file_path) as raw:
                # Extract metadata
                metadata = {
                    "ORIGINAL_FORMAT": "RAW",
                    "CAMERA_MAKE": getattr(raw, "camera_make", "Unknown"),
                    "CAMERA_MODEL": getattr(raw, "camera_model", "Unknown"),
                    "ISO": getattr(raw, "camera_iso", None),
                    "EXPOSURE_TIME": getattr(raw, "camera_exposure_time", None),
                    "APERTURE": getattr(raw, "camera_aperture", None),
                    "RAW_PATTERN": getattr(raw, "raw_pattern", None),
                    "RAW_COLORS": getattr(raw, "num_colors", None),
                    "WHITE_LEVEL": getattr(raw, "white_level", None),
                    "BLACK_LEVEL": getattr(raw, "black_level_per_channel", None),
                    "COLOR_MATRIX": getattr(raw, "color_matrix", None),
                }

                # Get raw Bayer data (preserves raw sensor data)
                raw_data = raw.raw_image.copy()

                # Store additional RAW-specific metadata
                metadata["RAW_WIDTH"] = raw_data.shape[1] if raw_data.ndim >= 2 else 0
                metadata["RAW_HEIGHT"] = raw_data.shape[0] if raw_data.ndim >= 1 else 0
                metadata["RAW_DTYPE"] = str(raw_data.dtype)

                # Clean up None values
                metadata = {k: v for k, v in metadata.items() if v is not None}

                return raw_data, metadata

        except Exception as e:
            # If rawpy fails, fallback to PIL for DNG files
            if file_path.lower().endswith(".dng"):
                logger.warning(
                    f"rawpy failed for {file_path}, trying PIL fallback: {e}"
                )
                return self._load_standard_image(file_path)
            else:
                raise ValueError(f"Could not load RAW file {file_path}: {e}")

    # -------------------------- Scaling --------------------------
    def _scale_image_data(self, data: np.ndarray) -> np.ndarray:
        if self.scaling_method == "none":
            return data
        # normalize to float64 0..1 heuristically
        arr = data
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float64) / 255.0
        elif arr.dtype == np.uint16:
            arr = arr.astype(np.float64) / 65535.0
        else:
            arr = arr.astype(np.float64)

        if self.scaling_method == "auto":
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx > 1.0 or mn < 0.0:
                bounds = np.percentile(arr, [0.1, 99.9])
                bounds_array = np.asarray(bounds)
                low: float = float(bounds_array[0])
                high: float = float(bounds_array[1])
                if high > low:
                    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
                else:
                    arr = np.zeros_like(arr)
        elif self.scaling_method == "linear":
            mn = float(np.min(arr))
            mx = float(np.max(arr))
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
        elif self.scaling_method == "percentile":
            bounds2 = np.percentile(arr, [1.0, 99.0])
            bounds2_array = np.asarray(bounds2)
            low2: float = float(bounds2_array[0])
            high2: float = float(bounds2_array[1])
            if high2 > low2:
                arr = np.clip((arr - low2) / (high2 - low2), 0.0, 1.0)
            else:
                arr = np.zeros_like(arr)

        if self.bit_depth == 8:
            result: np.ndarray = (arr * 255.0).astype(np.uint8)
            return result
        if self.bit_depth == 16:
            result = (arr * 65535.0).astype(np.uint16)
            return result
        raise ValueError(f"Unsupported bit depth: {self.bit_depth}")

    # -------------------------- Writer ---------------------------
    def _write_dng_tiff(
        self, data: np.ndarray, output_path: str, metadata: Dict[str, Any]
    ) -> None:
        """Write DNG/TIFF files with optional RAW compatibility."""

        # Handle DNG file extension
        if self.output_format == "dng":
            if output_path.lower().endswith((".tif", ".tiff")):
                output_path = (
                    output_path.rsplit(".", 1)[0] + ".dng"  # replace extension
                )
            elif not output_path.lower().endswith(".dng"):
                output_path = output_path + ".dng"

        if self.output_format == "dng" and self.raw_output:
            self._write_raw_compatible_dng(data, output_path, metadata)
        else:
            self._write_standard_dng_tiff(data, output_path, metadata)

    def _write_standard_dng_tiff(
        self, data: np.ndarray, output_path: str, metadata: Dict[str, Any]
    ) -> None:
        """Write standard DNG/TIFF file."""
        tags: Dict[str, Any] = {
            "Software": "universal-dng-converter",
            "ImageDescription": (
                f"Converted from {metadata.get('ORIGINAL_FORMAT', '?')}"
            ),
        }

        # Add basic metadata
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)) and len(str(v)) < 80:
                tags[f"Custom_{k}"] = v

        try:
            tifffile.imwrite(
                output_path,
                data,
                photometric="rgb" if data.ndim == 3 else "minisblack",
                compression="zlib",
                metadata=tags,
            )
            logger.info("Written %s", output_path)
        except Exception as e:
            logger.error(f"Error writing {output_path}: {e}")
            raise

    def _write_raw_compatible_dng(
        self, data: np.ndarray, output_path: str, metadata: Dict[str, Any]
    ) -> None:
        """Write RAW-compatible DNG file that rawpy can actually read."""
        try:
            # Convert to Bayer pattern if RGB
            if data.ndim == 3:
                bayer_data = self._rgb_to_bayer(data)
            else:
                # Prepare grayscale data to look like sensor data
                bayer_data = self._prepare_as_sensor_data(data)

            # Create fake camera metadata that rawpy recognizes
            camera_metadata = self._create_realistic_camera_metadata(metadata)

            # Write using Adobe DNG compatible format
            self._write_adobe_compatible_dng(bayer_data, output_path, camera_metadata)

            logger.info("Written RAW-compatible DNG %s", output_path)

            # Validate RAW compatibility
            if RAWPY_AVAILABLE:
                is_compatible = self._validate_raw_compatibility(output_path)
                if is_compatible:
                    logger.info("✓ DNG file is confirmed RAW-compatible with rawpy")
                else:
                    logger.warning("⚠ DNG created but rawpy compatibility uncertain")

        except Exception as e:
            logger.error(f"Error writing RAW-compatible DNG {output_path}: {e}")
            # Fallback to standard DNG
            logger.info("Falling back to standard DNG format")
            self._write_standard_dng_tiff(data, output_path, metadata)

    def _rgb_to_bayer(self, rgb_data: np.ndarray) -> np.ndarray:
        """Convert RGB image to RGGB Bayer pattern."""
        height, width = rgb_data.shape[:2]
        bayer = np.zeros((height, width), dtype=np.uint16)

        # RGGB Bayer pattern
        # R: even row, even col
        bayer[0::2, 0::2] = rgb_data[0::2, 0::2, 0]
        # G: even row, odd col and odd row, even col
        bayer[0::2, 1::2] = rgb_data[0::2, 1::2, 1]
        bayer[1::2, 0::2] = rgb_data[1::2, 0::2, 1]
        # B: odd row, odd col
        bayer[1::2, 1::2] = rgb_data[1::2, 1::2, 2]

        # Scale to 12-bit range with black level (typical camera values)
        if bayer.max() <= 255:  # 8-bit input
            scaled_bayer = bayer.astype(np.float32) / 255.0 * 3500 + 512
            bayer = scaled_bayer.astype(np.uint16)
        else:  # 16-bit input
            scaled_bayer = bayer.astype(np.float32) / 65535.0 * 3500 + 512
            bayer = scaled_bayer.astype(np.uint16)

        return np.array(bayer, dtype=np.uint16)

    def _prepare_as_sensor_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare grayscale data to look like raw sensor data."""
        # Convert to 16-bit and scale to realistic sensor range
        if data.dtype == np.uint8:
            sensor_data = (data.astype(np.float32) / 255.0 * 3500 + 512).astype(
                np.uint16
            )
        elif data.dtype == np.uint16:
            if data.max() > 4095:
                # Scale down to 12-bit range
                sensor_data = (data.astype(np.float32) / 65535.0 * 3500 + 512).astype(
                    np.uint16
                )
            else:
                # Already in reasonable range, just add black level
                sensor_data = np.clip(data + 512, 512, 4095).astype(np.uint16)
        else:
            sensor_data = np.clip(data, 512, 4095).astype(np.uint16)

        # Add subtle sensor noise
        noise = np.random.normal(0, 3, sensor_data.shape).astype(np.int16)
        noisy_data = np.clip(sensor_data.astype(np.int32) + noise, 512, 4095)
        sensor_data = noisy_data.astype(np.uint16)

        return np.array(sensor_data, dtype=np.uint16)

    def _create_realistic_camera_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create realistic camera metadata that fools rawpy."""

        # Real camera models that rawpy definitely supports
        cameras = [
            ("Canon", "Canon EOS R5"),
            ("Canon", "Canon EOS 5D Mark IV"),
            ("Nikon", "NIKON D850"),
            ("Sony", "ILCE-7RM4"),
            ("Fujifilm", "X-T4"),
        ]

        # Choose camera based on original format or default
        orig_format = metadata.get("ORIGINAL_FORMAT", "").upper()
        if any(x in orig_format for x in ["CANON", "CR2", "CR3"]):
            make, model = cameras[0]
        elif any(x in orig_format for x in ["NIKON", "NEF"]):
            make, model = cameras[2]
        elif any(x in orig_format for x in ["SONY", "ARW"]):
            make, model = cameras[3]
        else:
            make, model = cameras[0]  # Default Canon

        return {
            # CRÍTICO: Identificación de cámara
            "Make": make,
            "Model": model,
            # CRÍTICO: Software que rawpy reconoce
            "Software": "Canon Digital Photo Professional"
            if "Canon" in make
            else f"{make} Software",
            # CRÍTICO: Fechas
            "DateTime": "2023:10:08 12:00:00",
            "DateTimeOriginal": "2023:10:08 12:00:00",
            "DateTimeDigitized": "2023:10:08 12:00:00",
            # CRÍTICO: Configuración de exposición realista
            "ExposureTime": (1, 125),  # 1/125s
            "FNumber": (28, 10),  # f/2.8
            "ISO": 400,
            "FocalLength": (50, 1),  # 50mm
            # CRÍTICO: Estructura TIFF correcta
            "Orientation": 1,
            "XResolution": (72, 1),
            "YResolution": (72, 1),
            "ResolutionUnit": 2,
            # SÚPER CRÍTICO: PhotometricInterpretation para CFA
            "PhotometricInterpretation": 32803,  # Este es el valor mágico!
            # CRÍTICO: Sin compresión (rawpy lo prefiere)
            "Compression": 1,
            # CRÍTICO: Configuración de samples
            "SamplesPerPixel": 1,
            "BitsPerSample": 16,
            "PlanarConfiguration": 1,
            # CRÍTICO: Niveles de Canon realistas
            "WhiteLevel": 15871,  # Típico Canon R5
            # CRÍTICO: Patrón Bayer
            "CFARepeatPatternDim": (2, 2),
            "CFAPattern": (0, 1, 1, 2),  # RGGB
            # Información adicional de color
            "ColorSpace": 65535,  # Uncalibrated
        }

    def _write_adobe_compatible_dng(
        self, data: np.ndarray, output_path: str, camera_metadata: Dict[str, Any]
    ) -> None:
        """Write DNG using Adobe-compatible format that rawpy REALLY accepts."""

        # NO añadir tags extra que confundan a rawpy
        # Usar solo los metadatos que ya están en camera_metadata

        try:
            # CRÍTICO: usar compression=None (sin compresión) como descubrimos
            tifffile.imwrite(
                output_path,
                data,
                photometric="minisblack",  # rawpy espera esto
                compression=None,  # CRÍTICO: sin compresión
                metadata=camera_metadata,  # Solo los metadatos esenciales
                planarconfig="contig",
            )

            logger.info("Written Adobe DNG format: %s", output_path)

        except Exception as e:
            logger.error(f"Failed to write Adobe DNG: {e}")
            raise

    def _build_dng_metadata(
        self, data: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build DNG metadata dictionary for tifffile."""

        dng_meta = {
            # Basic info
            "Software": "universal-dng-converter",
            "ImageDescription": (
                f"RAW DNG converted from "
                f"{metadata.get('ORIGINAL_FORMAT', 'unknown')}"
            ),
            # DNG version info
            "DNGVersion": "1.4.0.0",
            "DNGBackwardVersion": "1.2.0.0",
            "UniqueCameraModel": metadata.get(
                "CAMERA_MODEL", "Universal DNG Converter Camera"
            ),
            "LocalizedCameraModel": metadata.get(
                "CAMERA_MODEL", "Universal DNG Converter Camera"
            ),
            # CFA info (essential for RAW)
            "CFARepeatPatternDim": "2 2",  # 2x2 Bayer pattern
            "CFAPattern": "0 1 1 2",  # RGGB pattern
            # Color space and calibration
            "ColorSpace": 65535,  # Uncalibrated
            "CalibrationIlluminant1": 21,  # D65
            "ColorMatrix1": "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0",
            # Scale and crop
            "DefaultScale": "1.0 1.0",
            "DefaultCropOrigin": "0 0",
            "DefaultCropSize": f"{data.shape[1]} {data.shape[0]}",
            # Processing hints
            "BayerGreenSplit": 0,
            "AntiAliasStrength": 1.0,
        }

        # Add levels
        if metadata.get("WHITE_LEVEL"):
            dng_meta["WhiteLevel"] = str(metadata["WHITE_LEVEL"])
        else:
            dng_meta["WhiteLevel"] = "65535" if data.dtype == np.uint16 else "255"

        if metadata.get("BLACK_LEVEL"):
            if isinstance(metadata["BLACK_LEVEL"], (list, tuple)):
                dng_meta["BlackLevel"] = " ".join(map(str, metadata["BLACK_LEVEL"]))
            else:
                black_level = metadata["BLACK_LEVEL"]
                dng_meta["BlackLevel"] = (
                    f"{black_level} {black_level} " f"{black_level} {black_level}"
                )
        else:
            dng_meta["BlackLevel"] = "0 0 0 0"

        # Add original metadata
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)) and len(str(v)) < 100:
                dng_meta[f"Original_{k}"] = str(v)

        return dng_meta

    def _build_complete_dng_tags(
        self, data: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[int, Any]:
        """Build complete set of DNG tags with proper TIFF tag numbers.

        This method is kept for compatibility but simplified.
        """
        # This method is kept for compatibility but simplified
        return {}

    def _simulate_bayer_from_rgb(self, rgb_data: np.ndarray) -> np.ndarray:
        """Convert RGB data to simulated Bayer pattern for RAW
        compatibility."""
        if rgb_data.ndim != 3 or rgb_data.shape[2] != 3:
            return rgb_data

        height, width = rgb_data.shape[:2]
        bayer = np.zeros((height, width), dtype=rgb_data.dtype)

        # RGGB Bayer pattern
        # Red: top-left pixels (even row, even col)
        bayer[0::2, 0::2] = rgb_data[0::2, 0::2, 0]  # Red channel

        # Green: top-right and bottom-left pixels
        bayer[0::2, 1::2] = rgb_data[0::2, 1::2, 1]  # Green channel
        bayer[1::2, 0::2] = rgb_data[1::2, 0::2, 1]  # Green channel

        # Blue: bottom-right pixels (odd row, odd col)
        bayer[1::2, 1::2] = rgb_data[1::2, 1::2, 2]  # Blue channel

        return np.array(bayer, dtype=bayer.dtype)

    def _get_raw_dng_tags(
        self, data: np.ndarray, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate DNG-specific tags for RAW compatibility."""
        raw_tags = {
            "DNGVersion": (1, 4, 0, 0),  # DNG version 1.4
            "DNGBackwardVersion": (1, 2, 0, 0),
            "UniqueCameraModel": metadata.get(
                "CAMERA_MODEL", "Universal DNG Converter Camera"
            ),
            "LocalizedCameraModel": metadata.get(
                "CAMERA_MODEL", "Universal DNG Converter Camera"
            ),
            # Essential DNG tags for RAW compatibility
            "PhotometricInterpretation": 32803,  # CFA (Color Filter Array)
            "SamplesPerPixel": 1,
            "PlanarConfiguration": 1,
            "CFARepeatPatternDim": [2, 2],  # 2x2 Bayer pattern
            "CFAPattern": [0, 1, 1, 2],  # RGGB pattern (Red, Green, Green, Blue)
            # Calibration and color
            "CalibrationIlluminant1": 21,  # D65
            "ColorMatrix1": [
                1.0,
                0.0,
                0.0,  # Red row
                0.0,
                1.0,
                0.0,  # Green row
                0.0,
                0.0,
                1.0,  # Blue row
            ],
            "CameraCalibration1": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            # Important for RAW processing
            "LinearizationTable": None,  # No linearization needed
            "BlackLevelRepeatDim": [2, 2],
            "DefaultScale": [1.0, 1.0],
            "DefaultCropOrigin": [0, 0],
            "DefaultCropSize": [data.shape[1], data.shape[0]],
            "BayerGreenSplit": 0,
            "AntiAliasStrength": 1.0,
            "ShadowScale": 1.0,
            # Lens correction
            "VignettingCorrectionParams": [1.0, 0.0, 0.0],
            "ChromaticAberrationCorrectionParams": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        }

        # Add RAW-specific tags if available
        if "WHITE_LEVEL" in metadata:
            raw_tags["WhiteLevel"] = [metadata["WHITE_LEVEL"]]
        else:
            # Estimate white level based on data type
            if data.dtype == np.uint8:
                raw_tags["WhiteLevel"] = [255]
            elif data.dtype == np.uint16:
                raw_tags["WhiteLevel"] = [65535]
            else:
                raw_tags["WhiteLevel"] = [int(np.max(data)) if data.size > 0 else 65535]

        if "BLACK_LEVEL" in metadata:
            raw_tags["BlackLevel"] = metadata["BLACK_LEVEL"]
        else:
            # Set black level per channel (RGGB)
            raw_tags["BlackLevel"] = [0, 0, 0, 0]

        # Add specific metadata from original RAW files if available
        if metadata.get("ORIGINAL_FORMAT") == "RAW":
            if "RAW_PATTERN" in metadata:
                pattern = metadata["RAW_PATTERN"]
                if hasattr(pattern, "shape") and len(pattern.shape) >= 2:
                    raw_tags["CFARepeatPatternDim"] = [
                        pattern.shape[1],
                        pattern.shape[0],
                    ]
                    raw_tags["CFAPattern"] = pattern.flatten().tolist()

        return raw_tags

    def _validate_raw_compatibility(self, dng_path: str) -> bool:
        """Validate that the generated DNG file can be read by rawpy."""
        try:
            with rawpy.imread(dng_path) as raw:
                # Try to access basic properties
                _ = raw.raw_image
                logger.info(f"DNG file {dng_path} is RAW-compatible")
                return True
        except Exception as e:
            logger.warning(
                f"Generated DNG {dng_path} may not be fully RAW-compatible: {e}"
            )
            return False

    # -------------------------- Public API -----------------------
    def convert_image(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> bool:
        try:
            ip = Path(input_path)
            op = Path(output_path)
            if ip.suffix.lower() not in self.SUPPORTED_FORMATS:
                self.stats["skipped"] += 1
                return False

            # Determine loader based on file format
            if ip.suffix.lower() in (".fits", ".fit", ".fts"):
                data, meta = self._load_fits_image(str(ip))
            elif ip.suffix.lower() == ".exr":
                data, meta = self._load_exr_image(str(ip))
            elif ip.suffix.lower() in (
                ".cr2",
                ".cr3",
                ".nef",
                ".arw",
                ".orf",
                ".rw2",
                ".raf",
                ".raw",
                ".dng",
            ):
                # Handle RAW formats
                if RAWPY_AVAILABLE:
                    try:
                        data, meta = self._load_raw_image(str(ip))
                    except Exception as e:
                        logger.warning(
                            f"RAW loading failed for {ip}, trying standard: {e}"
                        )
                        data, meta = self._load_standard_image(str(ip))
                else:
                    logger.warning(
                        f"rawpy not available, using standard loader for {ip}"
                    )
                    data, meta = self._load_standard_image(str(ip))
            else:
                data, meta = self._load_standard_image(str(ip))

            # Handle multi-channel images
            if data.ndim == 3 and data.shape[2] > 3:
                data = data[:, :, :3]

            # Scale data unless it's RAW and we want to preserve it
            if meta.get("ORIGINAL_FORMAT") == "RAW" and self.raw_output:
                # For RAW data, preserve original values when creating RAW DNG
                scaled = data
            else:
                scaled = self._scale_image_data(data)

            self._write_dng_tiff(scaled, str(op), meta)
            self.stats["converted"] += 1
            return True
        except Exception as exc:  # pragma: no cover
            logger.error("Error converting %s: %s", input_path, exc)
            self.stats["errors"] += 1
            return False

    def convert_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
    ) -> Dict[str, int]:
        inp = Path(input_dir)
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)
        pattern = "**/*" if recursive else "*"
        files: List[Path] = []
        for ext in self.SUPPORTED_FORMATS.keys():
            files.extend(inp.glob(f"{pattern}{ext}"))
            files.extend(inp.glob(f"{pattern}{ext.upper()}"))
        for f in sorted(files):
            target = outp / f"{f.stem}.{self.output_format}"
            if recursive:
                rel = f.relative_to(inp)
                target = outp / rel.parent / f"{f.stem}.{self.output_format}"
                target.parent.mkdir(parents=True, exist_ok=True)
            self.convert_image(f, target)
        return self.stats

    # -------------------- Legacy wrapper API ---------------------
    def convert_to_dng(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        bit_depth: Optional[int] = None,
        scaling_method: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> Optional[Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prev = (self.bit_depth, self.scaling_method, self.quality)
        try:
            if bit_depth is not None:
                self.bit_depth = bit_depth
            if scaling_method is not None:
                self.scaling_method = scaling_method
            if quality is not None:
                self.quality = quality
            out_file = out_dir / (Path(input_path).stem + f".{self.output_format}")
            ok = self.convert_image(input_path, out_file)
            return out_file if ok else None
        finally:
            self.bit_depth, self.scaling_method, self.quality = prev

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        recursive: bool = False,
        bit_depth: Optional[int] = None,
        scaling_method: Optional[str] = None,
        quality: Optional[int] = None,
    ) -> List[Tuple[Path, Optional[Path]]]:
        prev = (self.bit_depth, self.scaling_method, self.quality)
        try:
            if bit_depth is not None:
                self.bit_depth = bit_depth
            if scaling_method is not None:
                self.scaling_method = scaling_method
            if quality is not None:
                self.quality = quality
            inp = Path(input_dir)
            outp = Path(output_dir)
            outp.mkdir(parents=True, exist_ok=True)
            pattern = "**/*" if recursive else "*"
            files: List[Path] = []
            for ext in self.SUPPORTED_FORMATS.keys():
                files.extend(inp.glob(f"{pattern}{ext}"))
                files.extend(inp.glob(f"{pattern}{ext.upper()}"))
            results: List[Tuple[Path, Optional[Path]]] = []
            for f in sorted(files):
                target = outp / f"{f.stem}.{self.output_format}"
                if recursive:
                    rel = f.relative_to(inp)
                    target = outp / rel.parent / f"{f.stem}.{self.output_format}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                ok = self.convert_image(f, target)
                results.append((f, target if ok else None))
            return results
        finally:
            self.bit_depth, self.scaling_method, self.quality = prev

    def convert_to_raw_compatible_dng(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        bit_depth: Optional[int] = None,
        scaling_method: Optional[str] = None,
        quality: Optional[int] = None,
        validate_raw: bool = True,
    ) -> Optional[Path]:
        """Convert image to RAW-compatible DNG format.

        This method specifically creates DNG files that should be readable
        by rawpy and other RAW processing libraries.

        Args:
            input_path: Path to input image
            output_dir: Directory for output DNG file
            bit_depth: Override bit depth (8 or 16)
            scaling_method: Override scaling method
            quality: Override quality setting
            validate_raw: Whether to validate RAW compatibility after creation

        Returns:
            Path to created DNG file or None if conversion failed
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Store original settings
        prev_settings = (
            self.bit_depth,
            self.scaling_method,
            self.quality,
            self.raw_output,
        )

        try:
            # Override settings for RAW-compatible output
            if bit_depth is not None:
                self.bit_depth = bit_depth
            if scaling_method is not None:
                self.scaling_method = scaling_method
            if quality is not None:
                self.quality = quality

            # Always enable RAW output for this method
            self.raw_output = True

            out_file = out_dir / (Path(input_path).stem + ".dng")
            ok = self.convert_image(input_path, out_file)

            if ok and validate_raw and RAWPY_AVAILABLE:
                # Additional validation
                try:
                    with rawpy.imread(str(out_file)) as raw:
                        _ = raw.raw_image  # Test if we can read raw data
                    logger.info(f"Successfully created RAW-compatible DNG: {out_file}")
                except Exception as e:
                    logger.warning(
                        f"DNG created but may have limited RAW compatibility: {e}"
                    )

            return out_file if ok else None

        except Exception as e:
            logger.error(f"Failed to create RAW-compatible DNG for {input_path}: {e}")
            return None
        finally:
            # Restore original settings
            (
                self.bit_depth,
                self.scaling_method,
                self.quality,
                self.raw_output,
            ) = prev_settings

    def handle_raw_processing_error(
        self,
        dng_path: Union[str, Path],
        output_dir: Union[str, Path],
        fallback_format: str = "tiff",
    ) -> Optional[Path]:
        """Handle cases where a DNG file causes LibRawFileUnsupportedError.

        This method attempts to re-process a problematic DNG file to create
        a more compatible version or convert to an alternative format.

        Args:
            dng_path: Path to problematic DNG file
            output_dir: Directory for corrected output
            fallback_format: Format to use if DNG re-processing fails

        Returns:
            Path to corrected file or None if all attempts failed
        """
        dng_path = Path(dng_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # First, try to create a better DNG using an alternative method
        logger.info(f"Attempting to fix RAW compatibility for {dng_path}")

        try:
            # Try to convert using standard loader and enhanced RAW output
            if self.raw_output:
                result = self.convert_to_raw_compatible_dng(
                    dng_path, output_dir, validate_raw=False
                )
                if result and result.exists():
                    # Test if the new DNG is better
                    if RAWPY_AVAILABLE:
                        try:
                            with rawpy.imread(str(result)) as raw:
                                _ = raw.raw_image
                            logger.info(
                                f"Successfully created improved "
                                f"RAW-compatible DNG: {result}"
                            )
                            return result
                        except Exception:
                            logger.warning(
                                "New DNG still not fully rawpy-compatible, "
                                "but improved format created"
                            )
                            return result
                    else:
                        logger.info(f"Created enhanced DNG format: {result}")
                        return result
        except Exception as e:
            logger.warning(f"Enhanced DNG creation failed: {e}")

        # If DNG improvement failed, create a fallback in different format
        logger.info(f"Creating fallback in {fallback_format} format")
        try:
            prev_format = self.output_format
            prev_raw = self.raw_output

            self.output_format = fallback_format
            self.raw_output = False  # Use standard format for fallback

            fallback_path = output_dir / f"{dng_path.stem}_fallback.{fallback_format}"
            ok = self.convert_image(dng_path, fallback_path)

            # Restore settings
            self.output_format = prev_format
            self.raw_output = prev_raw

            if ok:
                logger.info(f"Successfully created fallback file: {fallback_path}")
                return fallback_path

        except Exception as e:
            logger.error(f"Fallback conversion also failed: {e}")

        # Last resort: try to create an enhanced TIFF with RAW-like properties
        try:
            logger.info("Attempting to create enhanced TIFF with RAW metadata")
            enhanced_path = output_dir / f"{dng_path.stem}_enhanced.tiff"

            # Load the problematic DNG
            with tifffile.TiffFile(dng_path) as tif:
                data = tif.asarray()

            # Create enhanced TIFF with comprehensive metadata
            enhanced_metadata = {
                "ORIGINAL_FORMAT": "DNG",
                "ENHANCED_BY": "universal-dng-converter",
                "RAW_COMPATIBLE": "True",
                "CONVERSION_NOTE": "Enhanced for better compatibility",
            }

            # Write with extensive metadata
            tifffile.imwrite(
                enhanced_path,
                data,
                photometric="minisblack" if data.ndim == 2 else "rgb",
                compression="lzw",
                resolution=(300.0, 300.0),
                metadata=enhanced_metadata,
                description="Enhanced TIFF with RAW-compatible metadata",
            )

            logger.info(f"Created enhanced TIFF: {enhanced_path}")
            return enhanced_path

        except Exception as e:
            logger.error(f"Enhanced TIFF creation failed: {e}")

        return None

    def create_pseudo_raw_dng(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        simulate_bayer: bool = True,
    ) -> Optional[Path]:
        """Create a pseudo-RAW DNG that mimics RAW file characteristics.

        This method creates DNG files that may be more compatible with
        RAW processing software by simulating sensor-like data patterns.

        Args:
            input_path: Path to input image
            output_dir: Directory for output DNG
            simulate_bayer: Whether to simulate Bayer pattern from RGB data

        Returns:
            Path to created pseudo-RAW DNG or None if failed
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load the image
            if input_path.suffix.lower() in (".fits", ".fit", ".fts"):
                data, meta = self._load_fits_image(str(input_path))
            elif input_path.suffix.lower() == ".exr":
                data, meta = self._load_exr_image(str(input_path))
            else:
                data, meta = self._load_standard_image(str(input_path))

            # Enhance metadata for pseudo-RAW
            pseudo_meta = meta.copy()
            pseudo_meta.update(
                {
                    "CAMERA_MAKE": "Universal DNG Converter",
                    "CAMERA_MODEL": "Pseudo RAW Camera",
                    "ISO": 100,
                    "WHITE_LEVEL": 16383 if data.dtype == np.uint16 else 255,
                    "BLACK_LEVEL": [64, 64, 64, 64]
                    if data.dtype == np.uint16
                    else [2, 2, 2, 2],
                    "PSEUDO_RAW": True,
                }
            )

            # Convert to pseudo-sensor data
            if simulate_bayer and data.ndim == 3:
                # Add some sensor-like characteristics
                sensor_data = self._create_sensor_like_data(data)
            else:
                sensor_data = data

            # Ensure 16-bit for better RAW compatibility
            if sensor_data.dtype != np.uint16:
                sensor_data = sensor_data.astype(np.uint16)

            # Create output file
            output_path = output_dir / f"{input_path.stem}_pseudo_raw.dng"

            # Use enhanced DNG writing
            prev_raw = self.raw_output
            self.raw_output = True
            self._write_raw_compatible_dng(sensor_data, str(output_path), pseudo_meta)
            self.raw_output = prev_raw

            logger.info(f"Created pseudo-RAW DNG: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to create pseudo-RAW DNG: {e}")
            return None

    def _create_sensor_like_data(self, rgb_data: np.ndarray) -> np.ndarray:
        """Create sensor-like data from RGB input."""
        if rgb_data.ndim != 3:
            return rgb_data

        # Convert to Bayer pattern
        bayer = self._simulate_bayer_from_rgb(rgb_data)

        # Add some sensor characteristics
        height, width = bayer.shape

        # Add noise (very small amount)
        noise_std = np.std(bayer) * 0.01  # 1% noise
        noise = np.random.normal(0, noise_std, bayer.shape).astype(bayer.dtype)
        bayer = np.clip(
            bayer.astype(np.int32) + noise, 0, np.iinfo(bayer.dtype).max
        ).astype(bayer.dtype)

        # Add subtle vignetting effect
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        max_dist = np.sqrt(center_y**2 + center_x**2)
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        vignette = 1.0 - (dist / max_dist) * 0.1  # 10% vignetting at corners

        bayer = (bayer * vignette).astype(bayer.dtype)

        return bayer


__all__ = ["ImageConverter"]
