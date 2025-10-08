# Universal DNG Converter

[![PyPI Version](https://img.shields.io/pypi/v/universal-dng-converter.svg)](https://pypi.org/project/universal-dng-converter/)
[![Python Versions](https://img.shields.io/pypi/pyversions/universal-dng-converter.svg)](https://pypi.org/project/universal-dng-converter/)
[![CI](https://github.com/dot-gabriel-ferrer/universal-dng-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/dot-gabriel-ferrer/universal-dng-converter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Professional multi-format image converter with DNG output support for astronomical and photographic applications.

## Overview

Universal DNG Converter is a robust, production-ready tool that converts various image formats to Adobe's Digital Negative (DNG) format while preserving metadata and providing intelligent scaling options. Originally designed for astronomical imaging workflows, it excels at handling high-dynamic-range data from FITS files while supporting standard photographic formats.

## Quick Installation

### From PyPI (Recommended)

```bash
pip install universal-dng-converter
```

### Basic Usage Test

```bash
# Download a test image or use your own
universal-dng-converter --input image.jpg --output converted.dng
```

The package includes all required dependencies and is ready to use immediately.

## Step-by-Step Guide

### Installation Options

#### Standard Installation (Recommended)
```bash
pip install universal-dng-converter
```

#### Development Installation
```bash
git clone https://github.com/dot-gabriel-ferrer/universal-dng-converter.git
cd universal-dng-converter
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
pip install -e .[dev]
```

### System Requirements

- Python 3.8–3.12
- Recommended: virtual environment for development
- Optional: opencv-python for EXR support (installed automatically)

### Installation Verification

```bash
# Check version
pip show universal-dng-converter

# Test CLI
universal-dng-converter --help

# Test Python import
python -c "from universal_dng_converter import DNGImageConverter; print('Installation successful!')"
```

### First Conversion

```bash
# Single image conversion
universal-dng-converter --input image.jpg --output converted.dng

# Using test images (if installed from source)
universal-dng-converter --input test_images/test_color.png --output output.dng
```

### Batch Conversion

```bash
# Convert all images in a directory
universal-dng-converter --input input_directory --output output_directory --recursive
```

### Python API Usage

```python
from universal_dng_converter import DNGImageConverter

# Initialize converter
converter = DNGImageConverter()

# Convert single image
output_path = converter.convert_to_dng("input.jpg", "output.dng")
print("Created:", output_path)

# Batch conversion
results = converter.batch_convert("input_dir/", "output_dir/", recursive=True)
for input_file, output_file in results:
    if output_file:
        print(f"Success: {input_file} → {output_file}")
    else:
        print(f"Failed: {input_file}")
```

### Advanced Configuration: Scaling Methods

The converter supports different scaling methods for optimal results:

- **auto** (default): Intelligent heuristic min/max scaling
- **linear**: Raw min→0, max→65535 mapping
- **percentile**: Robust scaling using percentiles (best for noisy data)
- **none**: No scaling (may clip or appear dark)

```bash
# Use percentile scaling for astronomical images
universal-dng-converter --input image.fits --output out.dng --scaling percentile

# Linear scaling for standard images
universal-dng-converter --input photo.jpg --output out.dng --scaling linear
```

### Supported Formats

**Input formats:**
- **Standard**: PNG, JPEG, TIFF, BMP, GIF
- **Astronomical**: FITS (.fits, .fit, .fts)
- **HDR**: EXR (requires opencv-python)

**Output format:**
- DNG (Digital Negative) - TIFF-based format compatible with Adobe Lightroom, Photoshop, and other RAW processors

### Output Validation

```bash
# Check output file properties with exiftool (if available)
exiftool converted.dng | grep -E "(Software|Bits Per Sample|DNG Version)"

# Open in image viewers
# - Adobe Lightroom (full DNG support)
# - Adobe Photoshop
# - darktable (open-source RAW processor)
# - Any TIFF-compatible viewer
```

### Development Setup

For contributors and advanced users:

```bash
# Run tests
pytest

# Run quality checks
pre-commit run --all-files

# Install development dependencies
pip install -e .[dev]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dark output image | Use `--scaling percentile` for better contrast |
| Washed out image | Try `--scaling linear` or adjust bit depth |
| EXR files not supported | Install with `pip install opencv-python` |
| Import errors | Ensure Python 3.8+ and reinstall: `pip install --upgrade universal-dng-converter` |
| Permission errors | Use `pip install --user universal-dng-converter` |

## Features

**Multiple input formats** (PNG, JPEG, TIFF, FITS, EXR)
**Intelligent scaling** (auto, linear, percentile, none)
**Batch processing** with recursive directory scanning
**Metadata preservation** from original files
**16-bit support** for high dynamic range
**Cross-platform** (Windows, macOS, Linux)
**Python 3.8-3.12** compatibility
**Command-line interface** and **Python API**
**Quality assurance** (full test suite, type checking)

## Links

- **PyPI Package**: https://pypi.org/project/universal-dng-converter/
- **GitHub Repository**: https://github.com/dot-gabriel-ferrer/universal-dng-converter
- **Issues & Support**: https://github.com/dot-gabriel-ferrer/universal-dng-converter/issues
- **Documentation**: [docs/](docs/)

## Contributing

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Adobe Systems** for the DNG specification
- **AstroPy Project** for excellent FITS file handling
- **Pillow Contributors** for comprehensive image format support
- **Python Scientific Community** for the foundational tools
