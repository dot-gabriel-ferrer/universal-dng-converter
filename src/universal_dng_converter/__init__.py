"""
Universal DNG Converter

Professional multi-format image converter with DNG output support.
"""

__version__ = "1.0.1"
__author__ = "Gabriel Ferrer"
__license__ = "MIT"

from .converter import ImageConverter

# Alias for backwards compatibility and clearer naming
DNGImageConverter = ImageConverter

__all__ = ["DNGImageConverter", "ImageConverter"]
