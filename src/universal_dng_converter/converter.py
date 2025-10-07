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
    }

    def __init__(
        self,
        output_format: str = "dng",
        quality: int = 100,
        bit_depth: int = 16,
        scaling_method: str = "auto",
    ) -> None:
        self.output_format = output_format
        self.quality = quality
        self.bit_depth = bit_depth
        self.scaling_method = scaling_method
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
        tags: Dict[str, Any] = {
            "Software": "universal-dng-converter",
            "ImageDescription": (
                f"Converted from {metadata.get('ORIGINAL_FORMAT', '?')}"
            ),
        }
        for k, v in metadata.items():
            if isinstance(v, (str, int, float)) and len(str(v)) < 80:
                tags[f"Custom_{k}"] = v
        if self.output_format == "dng":
            if output_path.lower().endswith((".tif", ".tiff")):
                output_path = (
                    output_path.rsplit(".", 1)[0] + ".dng"  # replace extension
                )
            elif not output_path.lower().endswith(".dng"):
                output_path = output_path + ".dng"
        tifffile.imwrite(
            output_path,
            data,
            photometric="rgb" if data.ndim == 3 else "minisblack",
            compression="zlib",
            metadata=tags,
        )
        logger.info("Written %s", output_path)

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
            if ip.suffix.lower() in (".fits", ".fit", ".fts"):
                data, meta = self._load_fits_image(str(ip))
            elif ip.suffix.lower() == ".exr":
                data, meta = self._load_exr_image(str(ip))
            else:
                data, meta = self._load_standard_image(str(ip))
            if data.ndim == 3 and data.shape[2] > 3:
                data = data[:, :, :3]
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


__all__ = ["ImageConverter"]
