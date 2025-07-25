"""
Ad Banner Expiry Detection System - Core Application Package

This package contains the core modules for banner detection, OCR processing,
CSV matching, and interactive cropping.
"""

__version__ = '1.0.0'  # Current version of the application

# Enable direct imports within the app package
import sys
import os

# Add this directory to path for intra-package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Export key modules and functions for easier access
try:
    from .utils import (
        parse_gps_coordinates,
        calculate_distance,
        is_expired,
        generate_banner_id,
        resize_image_for_display,
        draw_detection_boxes,
        create_directories,
        detect_text_language,
        normalize_hindi_text,
        extract_hindi_keywords,
        compare_hindi_text_similarity,
        preprocess_text_for_matching,
        ContentCategory
    )
except ImportError:
    # This prevents circular imports during initialization
    pass