import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(env_var, default_path):
    """Get path from environment variable or use default relative to BASE_DIR"""
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    return os.path.join(BASE_DIR, default_path)

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Path configurations
YOLO_MODEL_PATH = get_path('BANNER_MODEL_PATH', "data/models/yolov8s_banner.pt")
CSV_DATA_PATH = get_path('BANNER_CSV_PATH', "data/company_data.csv")
RESULTS_DIR = get_path('BANNER_RESULTS_DIR', "data/results/")
TEST_IMAGES_DIR = get_path('BANNER_TEST_IMAGES_DIR', "data/test_images/")

# Ensure directories exist
for directory in [RESULTS_DIR, TEST_IMAGES_DIR, os.path.dirname(YOLO_MODEL_PATH)]:
    ensure_dir(directory)

# Derived paths
AUTO_CROPS_DIR = os.path.join(RESULTS_DIR, "auto_crops/")
MANUAL_CROPS_DIR = os.path.join(RESULTS_DIR, "manual_crops/")
DETECTIONS_DIR = os.path.join(RESULTS_DIR, "detections/")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs/")

for directory in [AUTO_CROPS_DIR, MANUAL_CROPS_DIR, DETECTIONS_DIR, LOGS_DIR]:
    ensure_dir(directory)

# CPU OPTIMIZATION SETTINGS - NEW SECTION
CPU_OPTIMIZATION_CONFIG = {
    'max_image_dimension': 1024,        # Reduce image size for CPU processing
    'enable_gpu': False,                # Force CPU-only processing
    'max_processing_threads': 2,        # Limit threads to prevent CPU overload
    'memory_optimization': True,        # Enable memory optimizations
    'early_termination': True,          # Stop on first successful match
    'cache_enabled': True,              # Enable caching for repeated patterns
    'minimal_preprocessing': True       # Use minimal image preprocessing
}

# GPS EXTRACTION - ENHANCED FOR RELIABILITY
GPS_EXTRACTION_CONFIG = {
    'enable_region_caching': True,      # Cache successful regions
    'enable_pattern_caching': True,     # Cache successful patterns
    'max_regions_to_try': 7,           # Increased regions to try
    'max_preprocessing_variants': 4,    # Multiple preprocessing methods
    'enable_early_success_return': True, # Return immediately on success
    'timeout_per_region': 3.0,         # 3 second timeout per region
    'enable_fast_mode': True,           # Use fastest OCR configs only
    'comprehensive_patterns': True     # Use all GPS patterns
}

# Language Processing Configuration - Without Polyglot
LANGUAGE_PROCESSING_CONFIG = {
    'enable_polyglot': False,  # Disabled due to installation issues
    'use_character_based_detection': True,  # Use character-based detection instead
    'fallback_to_fuzzy_matching': True,  # Enable fallback matching methods
    'enable_hindi_normalization': True   # Enable Hindi text normalization
}

# HINDI LANGUAGE SUPPORT - NEW SECTION
HINDI_PROCESSING_CONFIG = {
    'enable_hindi_ocr': True,           # Enable Hindi OCR processing
    'enable_devanagari_normalization': True,  # Unicode normalization
    'enable_mixed_script_processing': True,   # Handle Hindi+English
    'hindi_keyword_extraction': True,  # Extract Hindi keywords
    'enable_character_based_detection': True,   # Hindi to Roman transliteration
    'character_similarity_matching': True,    # Character-level matching
    'enable_hindi_stop_words': True    # Use Hindi stop word filtering
}

# ENHANCED LANGUAGE DETECTION - NEW SECTION
LANGUAGE_DETECTION_CONFIG = {
    'hindi_unicode_range': (0x0900, 0x097F),  # Devanagari range
    'english_unicode_range': (0x0041, 0x007A),  # Basic Latin range
    'min_chars_for_detection': 3,      # Minimum characters for detection
    'confidence_threshold': 0.6        # Language detection confidence
}

# OCR ENGINE CONFIGURATION - ENHANCED
OCR_ENGINE_CONFIG = {
    'primary_engines': ['tesseract', 'easyocr', 'paddleocr'],
    'hindi_preferred_engines': ['easyocr', 'tesseract', 'paddleocr'],
    'english_preferred_engines': ['tesseract', 'easyocr'],
    'enable_engine_fallback': True,
    'max_engines_per_text': 2,         # Limit engines for speed
    'enable_confidence_weighting': True
}

# DYNAMIC THRESHOLDS - UPDATED FOR HINDI SUPPORT
DYNAMIC_CONTENT_MATCH_THRESHOLDS = {
    'pure_english': 65.0,      # Pure English banners (reduced for CPU speed)
    'english_dominant': 55.0,  # Mostly English with some Hindi (reduced)
    'hindi_dominant': 45.0,    # Mostly Hindi with some English (reduced)
    'pure_hindi': 35.0,        # Pure Hindi banners (lower due to complexity)
    'mixed_content': 50.0      # Balanced Hindi-English content
}

# HINDI-SPECIFIC MATCHING PARAMETERS - NEW SECTION
HINDI_MATCHING_CONFIG = {
    'exact_match_bonus': 20,           # Bonus for exact Hindi word matches
    'partial_match_threshold': 3,      # Minimum characters for partial matching
    'keyword_weight': 0.7,             # Weight for keyword vs full text matching
    'character_similarity_weight': 0.3, # Weight for character-level similarity
    'transliteration_bonus': 15,       # Bonus for successful transliteration matches
    'stop_words_filtering': True       # Enable Hindi stop words filtering
}

# OCR CONFIDENCE THRESHOLDS - UPDATED
BANNER_OCR_CONFIDENCE_THRESHOLDS = {
    'pure_english': 75.0,      # Reduced from 80.0 for CPU speed
    'english_dominant': 65.0,  # Reduced from 70.0 for CPU speed
    'hindi_dominant': 55.0,    # Reduced from 60.0 for CPU speed
    'pure_hindi': 45.0         # Reduced from 50.0 for Hindi complexity
}

# GPS PROXIMITY THRESHOLD
GPS_PROXIMITY_THRESHOLD = float(os.environ.get('BANNER_GPS_THRESHOLD', "2500"))

# OCR LANGUAGE CONFIGURATION
OCR_LANGS = "hin+eng"  # Support both Hindi and English
GPS_OCR_CONFIDENCE_THRESHOLD = 0.6  # Lower threshold for faster acceptance

# TESSERACT CONFIGURATIONS - OPTIMIZED
TESSERACT_CONFIGS = {
    'gps_extraction': [
        '--psm 6 --oem 3',                    # Fastest general
        '--psm 7 --oem 3',                    # Single line
        '--psm 8 --oem 3',                    # Single word
        '--psm 6 -c tessedit_char_whitelist=0123456789.,:-LatLongGPS '  # GPS specific
    ],
    'hindi_text': [
        '--psm 6 --oem 3',                    # General Hindi
        '--psm 7 --oem 3',                    # Single line Hindi
        '--psm 13 --oem 3'                    # Raw line Hindi
    ],
    'english_text': [
        '--psm 6 --oem 3',                    # General English
        '--psm 7 --oem 3'                     # Single line English
    ]
}

# PERFORMANCE OPTIMIZATIONS - ENHANCED
PERFORMANCE_OPTIMIZATIONS = {
    'enable_smart_engine_selection': True,     # Content-based OCR engine selection
    'enable_early_termination': True,          # Stop on high-confidence matches
    'max_processing_time_per_banner': 15,      # Reduced from 30 for speed
    'enable_preprocessing_optimization': True,  # Category-specific preprocessing
    'enable_result_caching': True,             # Cache OCR results
    'enable_memory_cleanup': True,             # Clean memory between banners
    'enable_priority_processing': True         # Process high-priority patterns first
}

# CACHE CONFIGURATION - NEW SECTION
CACHE_CONFIG = {
    'gps_pattern_cache_size': 50,      # Cache successful GPS patterns
    'ocr_result_cache_size': 100,      # Cache OCR results
    'matching_cache_size': 200,        # Cache matching results
    'enable_pattern_learning': True,   # Learn from successful patterns
    'cache_expiry_hours': 24          # Cache expiry time
}

# YOLO CONFIGURATION - CPU OPTIMIZED
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Slightly higher for fewer false positives
YOLO_IOU_THRESHOLD = 0.5          # Higher to merge overlapping detections
YOLO_CPU_OPTIMIZATIONS = {
    'device': 'cpu',               # Force CPU
    'half': False,                 # Disable half precision on CPU
    'augment': False,              # Disable augmentation for speed
    'agnostic_nms': False,         # Disable for speed
    'max_det': 10                  # Limit detections for speed
}

# BANNER PROCESSING CONFIGURATION
BANNER_PROCESSING_CONFIG = {
    'max_banners_per_image': 5,     # Limit banners to process
    'skip_small_banners': True,     # Skip banners smaller than threshold
    'min_banner_area': 2500,        # Minimum banner area (pixels)
    'parallel_processing': False,   # Disable parallel for CPU
    'batch_processing': False       # Process one by one
}

# MEMORY MANAGEMENT - NEW SECTION
MEMORY_CONFIG = {
    'clear_cache_frequency': 3,     # Clear cache every 3 images
    'max_cache_size_mb': 100,      # Maximum cache size
    'enable_garbage_collection': True,  # Force garbage collection
    'optimize_image_loading': True  # Optimize image loading
}

# FUZZY MATCHING CONFIGURATION
FUZZY_MATCHING_CONFIG = {
    'partial_threshold': 70.0,     # Reduced for CPU speed
    'token_threshold': 75.0,       # Reduced for CPU speed
    'ratio_threshold': 65.0        # Reduced for CPU speed
}

# WEB APP CONFIGURATIONS
UPLOAD_FOLDER = get_path('BANNER_UPLOAD_FOLDER', "web/static/uploads/")
ensure_dir(UPLOAD_FOLDER)
ensure_dir(os.path.join(UPLOAD_FOLDER, "crops"))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # Reduced to 8MB for faster uploads

# CONTENT CLASSIFICATION THRESHOLDS
CONTENT_CLASSIFICATION_THRESHOLDS = {
    'pure_hindi_ratio': 0.9,           # 90% Hindi characters for pure Hindi
    'hindi_dominant_ratio': 0.6,       # 60% Hindi characters for Hindi-dominant
    'pure_english_ratio': 0.9,         # 90% English characters for pure English
    'min_chars_for_classification': 5   # Minimum characters needed for classification
}

# ENHANCED MATCHING FEATURES
ENHANCED_MATCHING_FEATURES = {
    'cross_script_bonus_enabled': True,        # Cross-script matching bonus
    'fuzzy_matching_enabled': True,            # Fuzzy string matching
    'hindi_term_matching_enabled': True,       # Hindi festival/term matching
    'keyword_extraction_enabled': True,        # Focus on important keywords
    'partial_word_matching_enabled': True      # Partial word matching
}

# DEBUGGING AND LOGGING - NEW SECTION
DEBUG_CONFIG = {
    'enable_gps_debug': False,         # Enable GPS extraction debugging
    'enable_ocr_debug': False,         # Enable OCR debugging
    'enable_matching_debug': False,    # Enable matching debugging
    'save_intermediate_results': False, # Save intermediate processing results
    'verbose_logging': False          # Enable verbose logging
}

# LEGACY COMPATIBILITY
CONTENT_MATCH_THRESHOLD = DYNAMIC_CONTENT_MATCH_THRESHOLDS['english_dominant']

# STARTUP CONFIGURATION DISPLAY
print(f"ENHANCED Configuration loaded with Hindi support:")
print(f"   GPS extraction: {len(GPS_EXTRACTION_CONFIG)} optimizations")
print(f"   Hindi processing: {'ENABLED' if HINDI_PROCESSING_CONFIG['enable_hindi_ocr'] else 'DISABLED'}")
print(f"   CPU optimizations: {'ENABLED' if CPU_OPTIMIZATION_CONFIG['memory_optimization'] else 'DISABLED'}")
print(f"   Caching: {'ENABLED' if CACHE_CONFIG['enable_pattern_learning'] else 'DISABLED'}")
print(f"   Max banner processing time: {PERFORMANCE_OPTIMIZATIONS['max_processing_time_per_banner']}s")
print(f"   Dynamic thresholds: {len(DYNAMIC_CONTENT_MATCH_THRESHOLDS)} categories")