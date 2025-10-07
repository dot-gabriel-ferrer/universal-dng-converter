# Universal DNG Converter

<!-- Badges (PyPI/CI) temporarily removed until package & workflows published -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<!-- After first release re-enable (examples):
[![PyPI Version](https://img.shields.io/pypi/v/universal-dng-converter.svg)](https://pypi.org/project/universal-dng-converter/)
[![Python Versions](https://img.shields.io/pypi/pyversions/universal-dng-converter.svg)](https://pypi.org/project/universal-dng-converter/)
[![CI](https://github.com/dot-gabriel-ferrer/universal-dng-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/dot-gabriel-ferrer/universal-dng-converter/actions/workflows/ci.yml)
-->

Professional multi-format image converter with DNG output support for astronomical and photographic applications.

## Overview

Universal DNG Converter is a robust, production-ready tool that converts various image formats to Adobe's Digital Negative (DNG) format while preserving metadata and providing intelligent scaling options. Originally designed for astronomical imaging workflows, it excels at handling high-dynamic-range data from FITS files while supporting standard photographic formats.

## Step-by-Step Guide (Local Clone → First DNG)

### 1. Prerequisites

- Python 3.8–3.12
- Recommended: virtual environment
- Optional: `opencv-python` for EXR (install later if needed)

### 2. Clone

```
git clone https://github.com/dot-gabriel-ferrer/universal-dng-converter.git
cd universal-dng-converter
```

### 3. Environment

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
```

### 4. Install

Development (with tests, lint, type checking):

```
pip install -U pip
pip install -e .[dev]
```

Runtime only:

```
pip install .
```

Add EXR later:

```
pip install opencv-python
```

### 5. Sanity Check

```
universal-dng-converter --help
```

You should see CLI options including scaling and batch flags.

### 6. Single Conversion

```
universal-dng-converter --input tests/test_images/test_grayscale.jpg --output output.dng
```

If `--output` is a directory, the name is derived automatically.

### 7. Batch Conversion

Si el input es un directorio se convierte en lote automáticamente:
```
universal-dng-converter --input tests/test_images --output converted_dngs --recursive
```
Para filtrar extensiones actualmente usa todo el directorio (filtro por extensiones: mejora futura).

### 8. Scaling Methods

- auto (default): heuristic min/max
- linear: raw min→0, max→65535 mapping
- percentile: robust (defaults p2/p98) — use for noisy FITS
- none: no scaling (may clip or appear dark)
  Example:

```
universal-dng-converter --input image.fits --output out.dng --scaling percentile --pmin 1 --pmax 99
```

### 9. Python API

```python
from universal_dng_converter import ImageConverter

conv = ImageConverter(scaling="auto")
dst = conv.convert_to_dng("tests/test_images/test_color.png", "color.dng")
print("Created:", dst)

for src, out in conv.batch_convert("tests/test_images", "batch_out", recursive=False):
    print(src, '->', out)
```

### 10. Validate Output

```
exiftool color.dng | grep -E "(Software|Bits Per Sample|DNG Version)"
```

Open in Lightroom / darktable (pseudo-DNG: TIFF + tags, no sensor CFA raw pipeline).

### 11. Tests & Quality

```
pytest -q
pre-commit install
git add . && git commit -m "quality run"
```

Hooks: black, flake8, mypy, etc.

### 12. Optional Dependencies & Troubleshooting

| Issue / Feature          | Fix / Action                                      |
| ------------------------ | ------------------------------------------------- |
| Dark image               | Use `--scaling percentile`                        |
| Washed out               | Adjust `--pmin/--pmax` or use `linear`            |
| EXR not loading          | `pip install opencv-python`                       |
| LZW compression error    | `pip install imagecodecs` or disable compression* |
| FITS header missing      | Ensure primary HDU has image data                 |
| Extremely large FITS     | Crop/rebin before conversion                      |

*Actualmente se fuerza `compression="lzw"`; se añadirá opción para `none`.

### 13. Limitations

- Outputs TIFF with DNG-like tags (not full raw sensor pipeline)
- No compression toggle yet (planned)
- Color profile not embedded (assumed sRGB)

### 14. Roadmap (Planned)

- Configurable compression (none | lzw | deflate)
- Parallel batch mode
- Enhanced metadata (camera model passthrough)
- Extension filtering flag
- EXR automated tests

### 15. Contributing (Quick)

1. Fork & branch
2. `pip install -e .[dev]`
3. `pre-commit install`
4. Add tests, run `pytest -q`
5. Open PR

---

## ✨ Features (Summary)

- **🎯 Multi-format support**: FITS, PNG, JPEG, TIFF, BMP, GIF (first frame), EXR
- **🔧 Intelligent bit-depth conversion**: 8-bit and 16-bit output with smart scaling
- **📊 Advanced scaling methods**: Auto, linear, percentile, or no scaling
- **🏷️ Metadata preservation**: Extracts and embeds FITS headers and EXIF data
- **⚡ Batch processing**: Process single files or entire directories
- **🔄 Recursive directory scanning**: Process subdirectories automatically
- **📝 Comprehensive logging**: Detailed progress and error reporting
- **💻 Professional CLI**: Full command-line interface with extensive options
- **🐍 Python API**: Programmatic access for integration into workflows

## 🚀 Quick Start (Direct Install)

### Installation (from PyPI when published)

```bash
pip install universal-dng-converter  # (pending publication)
```

EXR support:
```bash
pip install universal-dng-converter[exr]
```

If you need LZW compression support install:
```
pip install imagecodecs
```

### Basic Usage

Convert a single image:

```bash
universal-dng-converter --input image.fits --output ./output/
```

Batch convert with advanced options:

```bash
universal-dng-converter \
  --input images/ \
  --output dng_output/ \
  --recursive \
  --bit-depth 16 \
  --scaling percentile \
  --verbose
```

### Python API (Alt example)

```python
from universal_dng_converter import DNGImageConverter
from pathlib import Path

converter = DNGImageConverter()
result = converter.convert_to_dng(
    input_path=Path("image.fits"),
    output_dir=Path("./output/"),
    bit_depth=16,
    scaling_method="auto"
)
```

## 📖 Documentation

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Usage Guide](docs/usage.md)** - Comprehensive usage examples
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Development Guide](docs/development.md)** - Contributing and development setup

## 🎯 Use Cases

### Astronomical Imaging

- Convert FITS files from telescopes and cameras to DNG for Adobe Lightroom/Photoshop
- Preserve astronomical metadata and handle high-dynamic-range data
- Batch process entire observation sessions

### Photography Workflows

- Convert RAW-like formats to standardized DNG
- Maintain EXIF metadata across format conversions
- Integrate into automated processing pipelines
- Archive scientific datasets in standardized format
- Convert microscopy and medical imaging data

## 📋 Requirements

### System Requirements

- Python 3.8 or higher
- 2GB+ RAM (for large astronomical images)
- Cross-platform: Windows, macOS, Linux

### Dependencies

- **astropy** - FITS file handling and astronomical calculations
- **Pillow** - Core image processing capabilities
- **tifffile** - Advanced TIFF/DNG output generation
- **numpy** - High-performance numerical operations

## 🔧 Supported Formats

| Format | Extensions                    | Support Level | Notes                   |
| ------ | ----------------------------- | ------------- | ----------------------- |
| FITS   | `.fits`, `.fit`, `.fts` | 🟢 Full       | Primary target format   |
| TIFF   | `.tif`, `.tiff`           | 🟢 Full       | Preserves all metadata  |
| PNG    | `.png`                      | 🟢 Full       | Supports transparency   |
| JPEG   | `.jpg`, `.jpeg`           | 🟢 Full       | EXIF metadata preserved |
| BMP    | `.bmp`                      | 🟢 Full       | Uncompressed support    |
| GIF    | `.gif`                      | 🟡 Partial    | First frame only        |
| EXR    | `.exr`                      | 🟡 Optional   | Requires opencv-python  |

## 🎛️ Scaling Methods

The converter offers four intelligent scaling methods optimized for different data types:

- **🤖 Auto**: Automatically selects optimal method based on image characteristics
- **📈 Linear**: Full dynamic range mapping (min→0, max→max_value)
- **📊 Percentile**: Robust scaling using 1st-99th percentiles (recommended for noisy data)
- **🔒 None**: Preserves original values without modification

## 🏗️ Project Structure

```
universal-dng-converter/
├── 📁 src/
│   └── universal_dng_converter/     # Main package
│       ├── __init__.py              # Package initialization
│       ├── converter.py             # Core conversion logic
│       └── cli.py                   # Command-line interface
├── 📁 tests/                        # Test suite
│   ├── __init__.py
│   └── test_converter.py           # Unit tests
├── 📁 scripts/                      # Standalone scripts
│   └── convert-to-dng              # Direct execution script
├── 📁 docs/                         # Documentation
│   ├── installation.md
│   ├── usage.md
│   └── development.md
├── 📁 examples/                     # Usage examples
│   └── basic_usage.py
├── pyproject.toml                   # Modern Python packaging
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt            # Development dependencies
└── README.md                        # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Development Guide](docs/development.md) for details on:

- Setting up the development environment
- Code style guidelines (Black + isort)
- Testing procedures (pytest + coverage)
- Submitting pull requests

### Quick Development Setup

```bash
git clone https://github.com/dot-gabriel-ferrer/universal-dng-converter.git
cd universal-dng-converter
pip install -e ".[dev]"
pre-commit install
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Adobe Systems** for the DNG specification
- **AstroPy Project** for excellent FITS file handling
- **Pillow Contributors** for comprehensive image format support
- **Python Scientific Community** for the foundational tools

---

**⭐ Star this repository if you find it useful!**

**Elías Gabriel Ferrer Jorge**

## License

MIT License

## Version

1.0.0 - Professional multi-format DNG converter
