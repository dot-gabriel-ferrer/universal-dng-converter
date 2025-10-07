# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-07

### Added
- Professional project structure with src/ layout
- Modern Python packaging with pyproject.toml
- Comprehensive test suite with pytest
- Command-line interface with argparse
- Batch processing capabilities
- Support for multiple image formats (FITS, PNG, JPEG, TIFF, BMP, GIF, EXR)
- Intelligent scaling methods (auto, linear, percentile, none)
- Metadata preservation for FITS and EXIF data
- Professional documentation and examples
- Development tools (pre-commit, tox, black, flake8, mypy)
- Type hints throughout codebase

### Changed
- Restructured from single script to professional package
- Improved error handling and logging
- Enhanced CLI with comprehensive options
- Better performance for large file processing

### Deprecated
- Direct script execution (use CLI instead)

### Fixed
- Memory efficiency for large astronomical images
- Proper handling of different bit depths
- Robust error handling for corrupted files

## [1.0.1] - 2025-10-07

### Added
- Functional and real sample image tests (`tests/test_functional_conversion.py`, `tests/test_real_images.py`).
- Step-by-step setup guide in README.

### Changed
- README reorganized; removed broken badges; clarified batch behavior.
- Version bump for initial PyPI release preparation.

### Fixed
- CLI examples referencing non-existent flags (`--batch`).
- LZW compression dependency note (added troubleshooting guidance).

### Pending / Roadmap
- Configurable compression option.
- Extension filtering flag.
- Parallel batch processing.
