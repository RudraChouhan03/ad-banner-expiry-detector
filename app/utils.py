import os
import re
import math
import datetime
from typing import Tuple, Dict, List, Optional, Union, Any
import numpy as np
import cv2
from enum import Enum
import unicodedata

class ContentCategory(Enum):
    PURE_ENGLISH = "pure_english"
    ENGLISH_DOMINANT = "english_dominant"
    HINDI_DOMINANT = "hindi_dominant"
    PURE_HINDI = "pure_hindi"

# OPTIMIZED GPS PATTERNS - HIGHEST PRIORITY FIRST
GPS_PATTERN_CACHE = {
    'last_successful_pattern': None,
    'last_successful_region': None,
    'success_count_by_pattern': {}
}

# HINDI TEXT PROCESSING UTILITIES
HINDI_UNICODE_RANGE = (0x0900, 0x097F)  # Devanagari Unicode range
ENGLISH_UNICODE_RANGE = (0x0041, 0x007A)  # Basic Latin range

def detect_text_language(text: str) -> Dict[str, Any]:
    """
    Detect language composition of text with detailed analysis.
    FIXED: Proper Hindi vs English categorization logic.
    """
    if not text or len(text.strip()) < 2:
        return {
            'category': ContentCategory.PURE_ENGLISH.value,
            'hindi_chars': 0,
            'english_chars': 0,
            'total_chars': 0,
            'hindi_percentage': 0.0,
            'english_percentage': 0.0
        }
    
    hindi_count = 0
    english_count = 0
    total_alphabetic = 0
    
    # Count characters by script
    for char in text:
        code_point = ord(char)
        
        # Check if it's Hindi (Devanagari)
        if HINDI_UNICODE_RANGE[0] <= code_point <= HINDI_UNICODE_RANGE[1]:
            hindi_count += 1
            total_alphabetic += 1
        # Check if it's English (Basic Latin letters)
        elif char.isalpha() and ENGLISH_UNICODE_RANGE[0] <= code_point <= ENGLISH_UNICODE_RANGE[1]:
            english_count += 1
            total_alphabetic += 1
    
    if total_alphabetic == 0:
        return {
            'category': ContentCategory.PURE_ENGLISH.value,
            'hindi_chars': 0,
            'english_chars': 0,
            'total_chars': len(text),
            'hindi_percentage': 0.0,
            'english_percentage': 0.0
        }
    
    # Calculate percentages
    hindi_percentage = (hindi_count / total_alphabetic) * 100
    english_percentage = (english_count / total_alphabetic) * 100
    
    # FIXED: Correct categorization logic based on Hindi percentage
    if hindi_percentage >= 90:
        category = ContentCategory.PURE_HINDI.value
    elif hindi_percentage >= 60:
        category = ContentCategory.HINDI_DOMINANT.value
    elif english_percentage >= 90:
        category = ContentCategory.PURE_ENGLISH.value
    elif english_percentage >= 60:
        category = ContentCategory.ENGLISH_DOMINANT.value
    else:
        # Mixed content - determine dominant language
        if hindi_percentage > english_percentage:
            category = ContentCategory.HINDI_DOMINANT.value
        else:
            category = ContentCategory.ENGLISH_DOMINANT.value
    
    return {
        'category': category,
        'hindi_chars': hindi_count,
        'english_chars': english_count,
        'total_chars': len(text),
        'hindi_percentage': hindi_percentage,
        'english_percentage': english_percentage
    }

def normalize_hindi_text(text: str) -> str:
    """
    Normalize Hindi text for better matching.
    Handles common Devanagari variations and normalization.
    """
    if not text:
        return ""
    
    try:
        # Unicode normalization for Devanagari
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        # Remove common punctuation that might interfere
        normalized = re.sub(r'[।॥,.\-_\(\)\[\]{}\'\"]+', ' ', normalized)
        
        # Clean up multiple spaces again
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        
        return normalized
        
    except Exception as e:
        print(f"❌ Hindi normalization error: {e}")
        return text.strip()

def extract_hindi_keywords(text: str) -> List[str]:
    """
    Extract meaningful Hindi keywords from text.
    Filters out common stop words and short words.
    """
    if not text:
        return []
    
    # Common Hindi stop words to filter out
    hindi_stop_words = {
        'और', 'या', 'की', 'के', 'का', 'को', 'में', 'से', 'पर', 'है', 'हैं', 'था', 'थे', 'होना', 'करना',
        'यह', 'वह', 'इस', 'उस', 'एक', 'दो', 'तीन', 'लिए', 'साथ', 'बाद', 'पहले', 'अब', 'तब',
        'कि', 'जो', 'भी', 'अगर', 'मगर', 'लेकिन', 'फिर', 'तो', 'सो'
    }
    
    try:
        # Normalize text first
        normalized = normalize_hindi_text(text)
        
        # Split into words
        words = normalized.split()
        
        # Filter meaningful keywords
        keywords = []
        for word in words:
            # Skip if too short
            if len(word) < 2:
                continue
            
            # Skip if it's a stop word
            if word in hindi_stop_words:
                continue
            
            # Skip if it's purely numeric
            if word.isdigit():
                continue
            
            # Check if word contains Hindi characters
            lang_info = detect_text_language(word)
            if lang_info['hindi_percentage'] > 0:
                keywords.append(word)
        
        return keywords
        
    except Exception as e:
        print(f"❌ Hindi keyword extraction error: {e}")
        return text.split() if text else []

def compare_hindi_text_similarity(text1: str, text2: str) -> float:
    """
    Compare similarity between two Hindi texts.
    Returns similarity percentage (0-100).
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # Normalize both texts
        norm1 = normalize_hindi_text(text1.lower())
        norm2 = normalize_hindi_text(text2.lower())
        
        if not norm1 or not norm2:
            return 0.0
        
        # Extract keywords from both
        keywords1 = set(extract_hindi_keywords(norm1))
        keywords2 = set(extract_hindi_keywords(norm2))
        
        if not keywords1 or not keywords2:
            # Fallback to character-level comparison
            return calculate_character_similarity(norm1, norm2)
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        jaccard_score = (intersection / union) * 100
        
        # Also check for partial matches
        partial_matches = 0
        for word1 in keywords1:
            for word2 in keywords2:
                if len(word1) >= 3 and len(word2) >= 3:
                    if word1 in word2 or word2 in word1:
                        partial_matches += 1
                        break
        
        partial_score = (partial_matches / len(keywords1)) * 100 if keywords1 else 0
        
        # Return the higher of the two scores
        return max(jaccard_score, partial_score)
        
    except Exception as e:
        print(f"❌ Hindi similarity comparison error: {e}")
        return 0.0

def calculate_character_similarity(text1: str, text2: str) -> float:
    """
    Calculate character-level similarity between two texts.
    Useful for Hindi text comparison when word-level fails.
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # Simple character overlap calculation
        chars1 = set(text1)
        chars2 = set(text2)
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return (intersection / union) * 100 if union > 0 else 0.0
        
    except Exception as e:
        print(f"❌ Character similarity error: {e}")
        return 0.0

def preprocess_text_for_matching(text: str) -> Dict[str, Any]:
    """
    Comprehensive text preprocessing for both Hindi and English matching.
    Returns processed text with metadata for efficient matching.
    """
    if not text:
        return {
            'original': '',
            'cleaned': '',
            'normalized': '',
            'keywords': [],
            'language_info': detect_text_language(''),
            'match_ready': ''
        }
    
    try:
        # Detect language composition
        lang_info = detect_text_language(text)
        
        # Clean the text based on language composition
        if lang_info['hindi_percentage'] > 30:
            # Hindi-dominant text processing
            cleaned = clean_mixed_language_text(text)
            normalized = normalize_hindi_text(cleaned)
            keywords = extract_hindi_keywords(normalized)
        else:
            # English-dominant text processing
            cleaned = re.sub(r'[^\w\s%.-]', ' ', text)
            normalized = re.sub(r'\s+', ' ', cleaned.strip().lower())
            keywords = [word for word in normalized.split() if len(word) >= 3]
        
        # Create match-ready text (normalized + keywords combined)
        match_ready = normalized + ' ' + ' '.join(keywords)
        match_ready = re.sub(r'\s+', ' ', match_ready.strip())
        
        return {
            'original': text,
            'cleaned': cleaned,
            'normalized': normalized,
            'keywords': keywords,
            'language_info': lang_info,
            'match_ready': match_ready
        }
        
    except Exception as e:
        print(f"❌ Text preprocessing error: {e}")
        return {
            'original': text,
            'cleaned': text,
            'normalized': text.lower(),
            'keywords': text.split(),
            'language_info': detect_text_language(text),
            'match_ready': text.lower()
        }

def clean_mixed_language_text(text: str) -> str:
    """
    Clean text that contains both Hindi and English.
    Preserves both scripts while removing noise.
    """
    if not text:
        return ""
    
    try:
        # Remove special characters but preserve Hindi and English
        # Keep Devanagari (0x0900-0x097F), Basic Latin (0x0020-0x007F), and common punctuation
        cleaned_chars = []
        
        for char in text:
            code_point = ord(char)
            
            # Keep Hindi characters (Devanagari)
            if 0x0900 <= code_point <= 0x097F:
                cleaned_chars.append(char)
            # Keep English characters and basic punctuation
            elif 0x0020 <= code_point <= 0x007F:
                # Skip some problematic characters but keep most
                if char not in '()[]{}*+?^$|\\<>':
                    cleaned_chars.append(char)
            # Keep common spaces and line breaks
            elif char in ' \n\t':
                cleaned_chars.append(' ')
        
        # Join and clean up spaces
        cleaned = ''.join(cleaned_chars)
        cleaned = re.sub(r'\s+', ' ', cleaned.strip())
        
        return cleaned
        
    except Exception as e:
        print(f"❌ Mixed language cleaning error: {e}")
        return text.strip()

def parse_gps_coordinates(text: str, quiet_mode: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    HIGHLY OPTIMIZED GPS parsing - tries most successful patterns first.
    Uses caching to prioritize patterns that worked before.
    """
    if not text or len(text.strip()) < 10:
        return None, None
    
    if not quiet_mode:
        print(f"🔍 GPS Parsing: '{text[:40]}{'...' if len(text) > 40 else ''}'")
    
    # Clean text once
    cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
    
    # PRIORITY 1: Try last successful pattern first (HIGHEST PRIORITY)
    if GPS_PATTERN_CACHE['last_successful_pattern']:
        last_pattern = GPS_PATTERN_CACHE['last_successful_pattern']
        if not quiet_mode:
            print(f"🚀 Trying cached successful pattern first...")
        
        matches = re.search(last_pattern, cleaned_text, re.IGNORECASE)
        if matches:
            try:
                lat, lon = float(matches.group(1)), float(matches.group(2))
                if _validate_india_coordinates(lat, lon):
                    GPS_PATTERN_CACHE['success_count_by_pattern'][last_pattern] = \
                        GPS_PATTERN_CACHE['success_count_by_pattern'].get(last_pattern, 0) + 1
                    if not quiet_mode:
                        print(f"✅ GPS found via cached pattern: {lat}, {lon}")
                    return lat, lon
            except (ValueError, IndexError):
                pass
    
    # PRIORITY 2: Ordered patterns by historical success rate
    gps_patterns = _get_prioritized_patterns()
    
    for i, (pattern_name, pattern) in enumerate(gps_patterns):
        if pattern == GPS_PATTERN_CACHE.get('last_successful_pattern'):
            continue  # Skip already tried pattern
            
        matches = re.search(pattern, cleaned_text, re.IGNORECASE)
        if matches:
            try:
                lat, lon = float(matches.group(1)), float(matches.group(2))
                if _validate_india_coordinates(lat, lon):
                    # Update cache for future use
                    GPS_PATTERN_CACHE['last_successful_pattern'] = pattern
                    GPS_PATTERN_CACHE['success_count_by_pattern'][pattern] = \
                        GPS_PATTERN_CACHE['success_count_by_pattern'].get(pattern, 0) + 1
                    
                    if not quiet_mode:
                        print(f"✅ GPS found via {pattern_name} (priority {i+1}): {lat}, {lon}")
                    return lat, lon
            except (ValueError, IndexError):
                continue
    
    if not quiet_mode:
        print("❌ No GPS coordinates found with any pattern")
    return None, None

def _get_prioritized_patterns() -> List[Tuple[str, str]]:
    """Get GPS patterns ordered by historical success rate"""
    base_patterns = [
        ('camera_format', r'Lat\s+(\d+\.\d+)°?\s+Long\s+(\d+\.\d+)°?'),
        ('flexible_camera', r'Lat(?:itude)?:?\s*(\d+\.\d+)°?\s*Long(?:itude)?:?\s*(\d+\.\d+)°?'),
        ('gps_colon', r'GPS:?\s*(\d+\.?\d*)[,\s]+(\d+\.?\d*)'),
        ('standard_format', r'[Ll]at(?:itude)?:?\s*(\d+\.?\d*)[,\s]+[Ll]ong(?:itude)?:?\s*(\d+\.?\d*)'),
        ('coordinates', r'[Cc]oord(?:inate)?s?:?\s*(\d+\.?\d*)[,\s]+(\d+\.?\d*)'),
        ('position', r'[Pp]osition:?\s*(\d+\.?\d*)[,\s]+(\d+\.?\d*)'),
        ('location', r'[Ll]ocation:?\s*(\d+\.?\d*)[,\s]+(\d+\.?\d*)'),
        ('decimal_only', r'(\d{2}\.\d{4,8})[,\s]+(\d{2,3}\.\d{4,8})'),
        ('spaced_format', r'(\d{2})\s+(\d{2})\s+(\d{1,3})\s*[,\s]+\s*(\d{2,3})\s+(\d{2})\s+(\d{1,3})'),
        ('dms_format', r'(\d{1,2})°(\d{1,2})\'(\d{1,2})\"[NS][,\s]+(\d{1,3})°(\d{1,2})\'(\d{1,2})\"[EW]'),
    ]
    
    # Sort by success count (highest first)
    success_counts = GPS_PATTERN_CACHE['success_count_by_pattern']
    return sorted(base_patterns, key=lambda x: success_counts.get(x[1], 0), reverse=True)

def _validate_india_coordinates(lat: float, lon: float) -> bool:
    """Validate coordinates are within India bounds"""
    # Extended India bounds with buffer for accuracy
    return (15.0 <= lat <= 40.0) and (65.0 <= lon <= 100.0)

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the Haversine distance between two GPS coordinates in meters."""
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2) * math.sin(delta_phi/2) + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda/2) * math.sin(delta_lambda/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    return distance

def is_expired(expiry_date_str: str) -> Tuple[bool, int, str]:
    """Check if a banner is expired based on its expiry date."""
    try:
        # Try multiple date formats
        date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d']
        expiry_date = None
        
        for fmt in date_formats:
            try:
                expiry_date = datetime.datetime.strptime(expiry_date_str, fmt).date()
                break
            except ValueError:
                continue
        
        if expiry_date is None:
            return True, 0, "ERROR: Invalid date format"
        
        current_date = datetime.datetime.now().date()
        days_diff = (expiry_date - current_date).days
        
        if days_diff < 0:
            return True, abs(days_diff), f"EXPIRED ({abs(days_diff)} days ago)"
        elif days_diff == 0:
            return False, 0, "EXPIRES TODAY"
        else:
            return False, days_diff, f"VALID (Expires in {days_diff} days)"
    except Exception as e:
        return True, 0, f"ERROR: {str(e)}"

def generate_banner_id(image_name: str, banner_index: int = 0) -> str:
    """Generate a unique banner ID based on timestamp and index."""
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if banner_index > 0:
        return f"{base_name}_{timestamp}_{banner_index}"
    else:
        return f"{base_name}_{timestamp}"

def resize_image_for_display(image: np.ndarray, max_width: int = 1200, max_height: int = 800) -> np.ndarray:
    """Resize image for display while maintaining aspect ratio."""
    if image is None:
        return None
    
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    return image

def draw_detection_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
                        labels: List[str] = None, colors: List[Tuple[int, int, int]] = None) -> np.ndarray:
    """Draw detection boxes on an image with optional labels."""
    output_image = image.copy()
    
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(output_image, (x1, y1), (x2, y2), colors[i], 2)
        
        if labels and i < len(labels):
            text_size = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10),
                         (x1 + text_size[0] + 10, y1), colors[i], -1)
            cv2.putText(output_image, labels[i], (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return output_image

def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Performance optimization utilities
def optimize_image_for_cpu_processing(image: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
    """Optimize image size for CPU processing to reduce memory usage"""
    if image is None:
        return None
    
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def clear_gpu_cache():
    """Clear any GPU cache to prevent memory issues on CPU-only systems"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass

def compare_hindi_text_similarity(text1: str, text2: str) -> float:
    """
    Compare similarity between two Hindi texts.
    Returns similarity percentage (0-100).
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # Normalize both texts
        norm1 = normalize_hindi_text(text1.lower())
        norm2 = normalize_hindi_text(text2.lower())
        
        if not norm1 or not norm2:
            return 0.0
        
        # Extract keywords from both
        keywords1 = set(extract_hindi_keywords(norm1))
        keywords2 = set(extract_hindi_keywords(norm2))
        
        if not keywords1 or not keywords2:
            # Fallback to character-level comparison
            return calculate_character_similarity(norm1, norm2)
        
        # Calculate Jaccard similarity
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        if union == 0:
            return 0.0
        
        jaccard_score = (intersection / union) * 100
        
        # Also check for partial matches
        partial_matches = 0
        for word1 in keywords1:
            for word2 in keywords2:
                if len(word1) >= 3 and len(word2) >= 3:
                    if word1 in word2 or word2 in word1:
                        partial_matches += 1
                        break
        
        partial_score = (partial_matches / len(keywords1)) * 100 if keywords1 else 0
        
        # Return the higher of the two scores
        return max(jaccard_score, partial_score)
        
    except Exception as e:
        print(f"❌ Hindi similarity comparison error: {e}")
        return 0.0

def calculate_character_similarity(text1: str, text2: str) -> float:
    """
    Calculate character-level similarity between two texts.
    Useful for Hindi text comparison when word-level fails.
    """
    if not text1 or not text2:
        return 0.0
    
    try:
        # Simple character overlap calculation
        chars1 = set(text1)
        chars2 = set(text2)
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return (intersection / union) * 100 if union > 0 else 0.0
        
    except Exception as e:
        print(f"❌ Character similarity error: {e}")
        return 0.0

def preprocess_text_for_matching(text: str) -> Dict[str, Any]:
    """
    Comprehensive text preprocessing for both Hindi and English matching.
    Returns processed text with metadata for efficient matching.
    """
    if not text:
        return {
            'original': '',
            'cleaned': '',
            'normalized': '',
            'keywords': [],
            'language_info': detect_text_language(''),
            'match_ready': ''
        }
    
    try:
        # Detect language composition
        lang_info = detect_text_language(text)
        
        # Clean the text based on language composition
        if lang_info['hindi_percentage'] > 30:
            # Hindi-dominant text processing
            cleaned = clean_mixed_language_text(text)
            normalized = normalize_hindi_text(cleaned)
            keywords = extract_hindi_keywords(normalized)
        else:
            # English-dominant text processing
            cleaned = re.sub(r'[^\w\s%.-]', ' ', text)
            normalized = re.sub(r'\s+', ' ', cleaned.strip().lower())
            keywords = [word for word in normalized.split() if len(word) >= 3]
        
        # Create match-ready text (normalized + keywords combined)
        match_ready = normalized + ' ' + ' '.join(keywords)
        match_ready = re.sub(r'\s+', ' ', match_ready.strip())
        
        return {
            'original': text,
            'cleaned': cleaned,
            'normalized': normalized,
            'keywords': keywords,
            'language_info': lang_info,
            'match_ready': match_ready
        }
        
    except Exception as e:
        print(f"❌ Text preprocessing error: {e}")
        return {
            'original': text,
            'cleaned': text,
            'normalized': text.lower(),
            'keywords': text.split(),
            'language_info': detect_text_language(text),
            'match_ready': text.lower()
        }
