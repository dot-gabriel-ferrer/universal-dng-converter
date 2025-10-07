# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation Methods

### From PyPI (Recommended)

```bash
pip install universal-dng-converter
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/dot-gabriel-ferrer/universal-dng-converter.git
cd universal-dng-converter
```

2. Install in development mode:
```bash
pip install -e .
```

3. For development with all dependencies:
```bash
pip install -e ".[dev]"
```

### For EXR Support

To enable EXR format support, install the optional dependency:

```bash
pip install "universal-dng-converter[exr]"
```

## Verify Installation

Test the installation by running:

```bash
universal-dng-converter --version
```

## Requirements

### Core Dependencies

- `astropy>=5.0` - For FITS file handling
- `pillow>=8.0.0` - For general image processing
- `tifffile>=2021.1.11` - For TIFF/DNG output
- `numpy>=1.20.0` - For numerical operations

### Optional Dependencies

- `opencv-python>=4.5.0` - For EXR format support

### Development Dependencies

- `pytest>=6.0` - For testing
- `black>=21.0` - For code formatting
- `flake8>=3.8` - For linting
- `mypy>=0.910` - For type checking
- `pre-commit>=2.15` - For pre-commit hooks
