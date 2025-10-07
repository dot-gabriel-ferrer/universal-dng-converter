# Usage Guide

## Command Line Interface

The Universal DNG Converter provides a comprehensive command-line interface for converting various image formats to DNG.

### Basic Usage

Convert a single image:
```bash
universal-dng-converter --input image.fits --output ./output/
```

Batch convert all images in a directory:
```bash
universal-dng-converter --input images/ --output dng_output/
```

### Advanced Options

#### Recursive Processing
Process all subdirectories:
```bash
universal-dng-converter --input data/ --output converted/ --recursive
```

#### Bit Depth Control
Choose output bit depth (8 or 16 bit):
```bash
universal-dng-converter --input image.fits --output ./ --bit-depth 16
```

#### Scaling Methods
Control how pixel values are scaled:

- **auto** (default): Automatically choose best method
- **linear**: Linear scaling from min to max
- **percentile**: Use 1st and 99th percentiles
- **none**: No scaling applied

```bash
universal-dng-converter --input image.fits --output ./ --scaling percentile
```

#### Quality Settings
Set TIFF compression quality (1-100):
```bash
universal-dng-converter --input image.fits --output ./ --quality 95
```

#### Verbose Output
Enable detailed logging:
```bash
universal-dng-converter --input image.fits --output ./ --verbose
```

### Complete Example

```bash
universal-dng-converter \
  --input /path/to/images/ \
  --output /path/to/dng_output/ \
  --recursive \
  --bit-depth 16 \
  --scaling percentile \
  --quality 95 \
  --verbose
```

## Python API

### Basic Usage

```python
from universal_dng_converter import DNGImageConverter
from pathlib import Path

# Create converter instance
converter = DNGImageConverter()

# Convert single file
result = converter.convert_to_dng(
    input_path=Path("image.fits"),
    output_dir=Path("./output/"),
    bit_depth=16,
    scaling_method="auto"
)

if result:
    print(f"Converted to: {result}")
else:
    print("Conversion failed")
```

### Batch Processing

```python
from universal_dng_converter import DNGImageConverter
from pathlib import Path

converter = DNGImageConverter()

# Batch convert directory
results = converter.batch_convert(
    input_dir=Path("images/"),
    output_dir=Path("dng_output/"),
    recursive=True,
    bit_depth=16,
    scaling_method="percentile"
)

# Check results
for input_file, output_file in results:
    if output_file:
        print(f"✓ {input_file} -> {output_file}")
    else:
        print(f"✗ Failed: {input_file}")
```

## Supported Formats

### Input Formats

- **FITS** (.fits, .fit) - Astronomical image format
- **PNG** (.png) - Portable Network Graphics
- **JPEG** (.jpg, .jpeg) - JPEG images
- **TIFF** (.tif, .tiff) - Tagged Image File Format
- **BMP** (.bmp) - Bitmap images
- **GIF** (.gif) - Graphics Interchange Format (first frame only)
- **EXR** (.exr) - OpenEXR high dynamic range (requires opencv-python)

### Output Format

- **DNG** (.dng) - Digital Negative format compatible with Adobe applications

## Scaling Methods Explained

### Auto Scaling
Automatically selects the best scaling method based on image characteristics:
- For FITS files: Uses percentile scaling
- For other formats: Uses linear scaling

### Linear Scaling
Maps the full range of pixel values (min to max) to the output bit depth.
Best for images with good contrast and no extreme outliers.

### Percentile Scaling
Uses the 1st and 99th percentiles as scaling bounds, clipping extreme values.
Ideal for astronomical images with hot pixels or noise.

### No Scaling
Preserves original pixel values without modification.
Use when the input is already properly scaled.
