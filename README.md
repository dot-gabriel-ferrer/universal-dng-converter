# DNG Image Converter

Professional multi-format image converter with DNG output support.

## Overview

This tool converts various image formats (FITS, PNG, JPEG, TIFF, BMP, GIF, EXR) to DNG (Digital Negative) format while preserving metadata and providing robust scaling options.

## Features

- **Multi-format support**: FITS, PNG, JPEG, TIFF, BMP, GIF (first frame), EXR
- **Intelligent bit-depth conversion**: 8-bit and 16-bit output
- **Robust scaling methods**: Auto, linear, percentile, or no scaling
- **Metadata preservation**: Extracts and embeds FITS headers and EXIF data
- **Batch processing**: Process single files or entire directories
- **Recursive directory scanning**: Process subdirectories automatically
- **Comprehensive logging**: Detailed progress and error reporting
- **Professional CLI**: Full command-line interface with help

## Installation

Required dependencies:
```bash
pip install astropy pillow tifffile numpy
```

Optional dependencies for enhanced format support:
```bash
pip install opencv-python  # For EXR format support
```

## Usage

### Basic Examples

Convert a single FITS file:
```bash
python scripts/dng_image_converter.py --input image.fits --output ./
```

Convert all images in a directory:
```bash
python scripts/dng_image_converter.py --input images/ --output dng_output/
```

Recursive batch conversion with custom settings:
```bash
python scripts/dng_image_converter.py \
  --input data/ \
  --output converted/ \
  --recursive \
  --scaling percentile \
  --bit-depth 16 \
  --format dng
```

### Command Line Options

```
--input, -i          Input file or directory path (required)
--output, -o         Output directory path (required)
--format, -f         Output format: dng, tiff (default: dng)
--recursive, -r      Process subdirectories recursively
--scaling            Scaling method: auto, linear, percentile, none (default: auto)
--bit-depth          Target bit depth: 8, 16 (default: 16)
--quality            Output quality 0-100 (default: 100)
--verbose, -v        Enable verbose logging
```

### Scaling Methods

- **auto**: Intelligent scaling based on data range detection
- **linear**: Simple min-max linear scaling
- **percentile**: Robust 1st-99th percentile scaling
- **none**: No scaling, preserve original values

## Supported Formats

| Format | Extension | Description | Status |
|--------|-----------|-------------|---------|
| FITS | .fits, .fit, .fts | Astronomical images | ✓ Full support |
| PNG | .png | Portable Network Graphics | ✓ Full support |
| JPEG | .jpg, .jpeg | JPEG images | ✓ Full support |
| TIFF | .tif, .tiff | Tagged Image Format | ✓ Full support |
| BMP | .bmp | Bitmap images | ✓ Full support |
| GIF | .gif | Graphics Interchange Format | ✓ First frame only |
| EXR | .exr | OpenEXR HDR | ✓ Requires opencv-python |

## Metadata Handling

The converter preserves important metadata from source images:

### FITS Files
- OBJECT, TELESCOP, INSTRUME
- DATE-OBS, EXPTIME, FILTER
- OBSERVER, BITPIX, NAXIS info
- All custom headers (within size limits)

### Standard Images
- EXIF data from JPEG/TIFF files
- Original format information
- Color space and bit depth info

## Output Format

The converter produces DNG-compatible TIFF files with:
- LZW compression for efficient storage
- 300 DPI resolution setting
- Embedded metadata as TIFF tags
- Professional color profiles
- Artist attribution (Gabriel Ferrer)

## Migration from Old Script

The original `fits_to_dng.py` script has been deprecated and replaced. When you run the old script, it will:
1. Display a deprecation warning
2. Automatically redirect to the new converter
3. Pass through your original arguments

## Examples

### Astronomical Data Processing
```bash
# Convert FITS files with robust scaling
python scripts/dng_image_converter.py \
  --input telescope_data/ \
  --output processed_dng/ \
  --scaling percentile \
  --recursive \
  --verbose
```

### Photography Workflow
```bash
# Convert RAW and JPEG files to 16-bit DNG
python scripts/dng_image_converter.py \
  --input photos/ \
  --output dng_archive/ \
  --bit-depth 16 \
  --quality 100
```

### Batch Processing with Custom Scaling
```bash
# Linear scaling for scientific data
python scripts/dng_image_converter.py \
  --input experiments/ \
  --output results/ \
  --scaling linear \
  --format tiff
```

## Logging

The converter creates detailed logs in `dng_converter.log` and displays progress in the terminal. Use `--verbose` for additional debugging information.

## Author

**Gabriel Ferrer**

## License

MIT License

## Version

1.0.0 - Professional multi-format DNG converter
