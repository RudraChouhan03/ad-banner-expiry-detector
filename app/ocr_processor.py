import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
import re
from utils import (parse_gps_coordinates, optimize_image_for_cpu_processing, 
                  detect_text_language, normalize_hindi_text, clean_mixed_language_text,
                  ContentCategory, preprocess_text_for_matching)
import config
import time
from enum import Enum
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# GPS EXTRACTION CACHE - For multiple banner processing
GPS_EXTRACTION_CACHE = {
    'last_successful_region': None,
    'last_successful_method': None,
    'region_success_count': {}
}

# OCR ENGINE CACHE - For efficient engine selection
OCR_ENGINE_CACHE = {
    'last_successful_engine': None,
    'engine_success_count': {},
    'language_engine_preference': {}
}

# POLYGLOT-FREE LANGUAGE DETECTION - Using character-based analysis
LANGUAGE_DETECTION_FALLBACK = {
    'use_character_based': True,
    'hindi_unicode_range': (0x0900, 0x097F),
    'english_unicode_range': (0x0041, 0x007A),
    'min_confidence_threshold': 0.6
}

# Configure Tesseract path
try:
    import pytesseract
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    if os.path.exists(tesseract_path):
        print(f"[OK] Tesseract configured: {tesseract_path}")
    else:
        print(f"[ERROR] Tesseract not found at: {tesseract_path}")
except ImportError:
    print("[ERROR] pytesseract not installed")

class OCRProcessor:
    """
    COMPLETE OCR Processor with full Hindi support and optimized GPS extraction.
    Works entirely without polyglot dependency using character-based language detection.
    Handles both Hindi (Devanagari) and English text extraction efficiently.
    """
    
    def __init__(self, langs: str = "hin+eng"):
        self.langs = langs
        self.available_engines = self._initialize_ocr_engines()
        self.processing_stats = {
            'total_images': 0,
            'successful_gps_extractions': 0,
            'avg_processing_time': 0,
            'hindi_extractions': 0,
            'english_extractions': 0,
            'mixed_extractions': 0,
            'polyglot_free_processing': True  # Flag to indicate polyglot-free operation
        }
        
        # Initialize language detection without polyglot
        self.language_detector = self._initialize_language_detection()
        
        print(f"[START] OCR Processor initialized with engines: {', '.join(self.available_engines)}")
        print(f"[LANG] Language support: {langs} (Polyglot-FREE)")
        print(f"[CONFIG] Using character-based Hindi detection")
    
    def _initialize_language_detection(self) -> Dict[str, Any]:
        """Initialize polyglot-free language detection system"""
        return {
            'method': 'character_based',
            'hindi_support': True,
            'english_support': True,
            'mixed_script_support': True,
            'fallback_detection': True
        }
    
    def _initialize_ocr_engines(self) -> List[str]:
        """Initialize available OCR engines with priority order for Hindi support"""
        available = []
        
        # PRIORITY 1: Tesseract (best for Hindi with proper config)
        try:
            import pytesseract
            if pytesseract.pytesseract.tesseract_cmd and os.path.exists(pytesseract.pytesseract.tesseract_cmd):
                # Test Hindi support
                try:
                    test_result = pytesseract.get_languages()
                    if 'hin' in test_result or 'Hindi' in str(test_result):
                        available.append("tesseract")
                        print("[OK] Tesseract with Hindi support (Priority 1)")
                    else:
                        available.append("tesseract")
                        print("[OK] Tesseract (English only)")
                except:
                    available.append("tesseract")
                    print("[OK] Tesseract (Basic)")
        except:
            print("[ERROR] Tesseract unavailable")
        
        # PRIORITY 2: EasyOCR (excellent for Hindi)
        try:
            import easyocr
            available.append("easyocr")
            print("[OK] EasyOCR with Hindi support (Priority 2)")
        except:
            print("[ERROR] EasyOCR unavailable")
        
        # PRIORITY 3: PaddleOCR (good backup for Hindi)
        try:
            import paddleocr
            available.append("paddleocr")
            print("[OK] PaddleOCR with Hindi support (Priority 3)")
        except:
            print("[ERROR] PaddleOCR unavailable")
        
        return available
    
    def extract_gps_from_image(self, image_path: str) -> Tuple[Optional[float], Optional[float], str]:
        """
        HIGHLY OPTIMIZED GPS extraction with caching and priority regions.
        Perfect for processing multiple banners from the same image.
        """
        start_time = time.time()
        print(f"[SEARCH] GPS extraction from: {os.path.basename(image_path)}")
        
        # Load and optimize image for CPU processing
        image = cv2.imread(image_path)
        if image is None:
            print("[ERROR] Could not load image")
            return None, None, ""
        
        # Optimize image size for CPU processing
        image = optimize_image_for_cpu_processing(image, max_dimension=1024)
        print(f"[OK] Image optimized: {image.shape}")
        
        # PRIORITY 1: Try last successful region first (HIGHEST PRIORITY)
        if GPS_EXTRACTION_CACHE['last_successful_region'] and GPS_EXTRACTION_CACHE['last_successful_method']:
            print("[START] Trying cached successful region/method first...")
            lat, lon, text = self._extract_from_cached_region(image)
            if lat is not None and lon is not None:
                processing_time = time.time() - start_time
                print(f"[OK] GPS found via cache in {processing_time:.2f}s: {lat}, {lon}")
                self._update_gps_stats(True, processing_time)
                return lat, lon, text
        
        # PRIORITY 2: High-success regions in order
        priority_regions = self._get_prioritized_regions(image)
        
        for region_name, region_extractor in priority_regions:
            print(f"[SEARCH] Trying region: {region_name}")
            lat, lon, text = region_extractor(image)
            
            if lat is not None and lon is not None:
                # Update cache for future use
                GPS_EXTRACTION_CACHE['last_successful_region'] = region_name
                GPS_EXTRACTION_CACHE['region_success_count'][region_name] = \
                    GPS_EXTRACTION_CACHE['region_success_count'].get(region_name, 0) + 1
                
                processing_time = time.time() - start_time
                print(f"[OK] GPS found in {region_name} in {processing_time:.2f}s: {lat}, {lon}")
                self._update_gps_stats(True, processing_time)
                return lat, lon, text
        
        processing_time = time.time() - start_time
        print(f"[ERROR] GPS extraction failed in {processing_time:.2f}s")
        self._update_gps_stats(False, processing_time)
        return None, None, "GPS extraction failed"
    
    def _get_prioritized_regions(self, image: np.ndarray) -> List[Tuple[str, callable]]:
        """Get GPS extraction regions ordered by historical success rate"""
        height, width = image.shape[:2]
        
        base_regions = [
            ('bottom_right_overlay', lambda img: self._extract_bottom_right_overlay(img)),
            ('bottom_strip_wide', lambda img: self._extract_bottom_strip_wide(img)),
            ('bottom_left_overlay', lambda img: self._extract_bottom_left_overlay(img)),
            ('bottom_center', lambda img: self._extract_bottom_center(img)),
            ('full_bottom_scan', lambda img: self._extract_full_bottom_scan(img)),
            ('top_right_overlay', lambda img: self._extract_top_right_overlay(img)),
            ('corners_scan', lambda img: self._extract_corners_scan(img)),
        ]
        
        # Sort by success count (highest first)
        success_counts = GPS_EXTRACTION_CACHE['region_success_count']
        return sorted(base_regions, key=lambda x: success_counts.get(x[0], 0), reverse=True)
    
    def _extract_from_cached_region(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS using cached successful region and method"""
        cached_region = GPS_EXTRACTION_CACHE['last_successful_region']
        
        if cached_region == 'bottom_right_overlay':
            return self._extract_bottom_right_overlay(image)
        elif cached_region == 'bottom_strip_wide':
            return self._extract_bottom_strip_wide(image)
        elif cached_region == 'bottom_left_overlay':
            return self._extract_bottom_left_overlay(image)
        elif cached_region == 'bottom_center':
            return self._extract_bottom_center(image)
        elif cached_region == 'full_bottom_scan':
            return self._extract_full_bottom_scan(image)
        elif cached_region == 'top_right_overlay':
            return self._extract_top_right_overlay(image)
        elif cached_region == 'corners_scan':
            return self._extract_corners_scan(image)
        
        return None, None, ""
    
    def _extract_bottom_right_overlay(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS from bottom-right overlay (most common location)"""
        height, width = image.shape[:2]
        
        # Multiple bottom-right regions with different sizes
        regions = [
            image[height-200:height, width-500:width],  # Standard
            image[height-150:height, width-400:width],  # Smaller
            image[height-250:height, width-600:width],  # Larger
            image[height-100:height, width-300:width],  # Compact
        ]
        
        for i, region in enumerate(regions):
            if region.size == 0:
                continue
            
            lat, lon, text = self._process_gps_region_fast(region, f"bottom_right_{i}")
            if lat is not None:
                return lat, lon, text
        
        return None, None, ""
    
    def _extract_bottom_strip_wide(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS from wide bottom strip"""
        height, width = image.shape[:2]
        bottom_strip = image[height-100:height, :]
        
        return self._process_gps_region_fast(bottom_strip, "bottom_strip_wide")
    
    def _extract_bottom_left_overlay(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS from bottom-left overlay"""
        height, width = image.shape[:2]
        region = image[height-200:height, :500]
        
        return self._process_gps_region_fast(region, "bottom_left")
    
    def _extract_bottom_center(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS from bottom-center region"""
        height, width = image.shape[:2]
        region = image[height-150:height, width//4:3*width//4]
        
        return self._process_gps_region_fast(region, "bottom_center")
    
    def _extract_full_bottom_scan(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Full bottom region scan as last resort"""
        height, width = image.shape[:2]
        region = image[height-300:height, :]
        
        return self._process_gps_region_fast(region, "full_bottom")
    
    def _extract_top_right_overlay(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Extract GPS from top-right overlay (some cameras)"""
        height, width = image.shape[:2]
        region = image[:150, width-500:width]
        
        return self._process_gps_region_fast(region, "top_right")
    
    def _extract_corners_scan(self, image: np.ndarray) -> Tuple[Optional[float], Optional[float], str]:
        """Scan all corners for GPS information"""
        height, width = image.shape[:2]
        
        corners = [
            image[:100, :400],  # Top-left
            image[:100, width-400:width],  # Top-right
            image[height-100:height, :400],  # Bottom-left
            image[height-100:height, width-400:width],  # Bottom-right
        ]
        
        for i, corner in enumerate(corners):
            if corner.size == 0:
                continue
            
            lat, lon, text = self._process_gps_region_fast(corner, f"corner_{i}")
            if lat is not None:
                return lat, lon, text
        
        return None, None, ""
    
    def _process_gps_region_fast(self, region: np.ndarray, region_name: str) -> Tuple[Optional[float], Optional[float], str]:
        """Fast GPS processing with minimal preprocessing"""
        if region.size == 0:
            return None, None, ""
        
        try:
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()
            
            # PRIORITY 1: Try original first (fastest)
            lat, lon, text = self._try_tesseract_gps_fast(gray, f"{region_name}_original")
            if lat is not None:
                return lat, lon, text
            
            # PRIORITY 2: Simple enhancement only if needed
            enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
            lat, lon, text = self._try_tesseract_gps_fast(enhanced, f"{region_name}_enhanced")
            if lat is not None:
                return lat, lon, text
            
            # PRIORITY 3: Upscale only as last resort
            h, w = gray.shape
            if max(h, w) < 200:  # Only upscale if very small
                upscaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                lat, lon, text = self._try_tesseract_gps_fast(upscaled, f"{region_name}_upscaled")
                if lat is not None:
                    return lat, lon, text
            
            # PRIORITY 4: Try other OCR engines if available
            if "easyocr" in self.available_engines:
                lat, lon, text = self._try_easyocr_gps(gray, f"{region_name}_easyocr")
                if lat is not None:
                    return lat, lon, text
            
        except Exception as e:
            print(f"[ERROR] Error processing region {region_name}: {e}")
        
        return None, None, ""
    
    def _try_tesseract_gps_fast(self, image: np.ndarray, context: str) -> Tuple[Optional[float], Optional[float], str]:
        """Fast Tesseract GPS extraction with minimal configs"""
        if "tesseract" not in self.available_engines:
            return None, None, ""
        
        try:
            import pytesseract
            
            # PRIORITY ORDER: Fastest configs first
            configs = [
                '--psm 6 --oem 3',  # Fastest general config
                '--psm 7 --oem 3',  # Single text line
                '--psm 8 --oem 3',  # Single word (for compact GPS)
                '--psm 6 -c tessedit_char_whitelist=0123456789.,:-LatLongGPS ',  # GPS specific
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, lang='eng', config=config).strip()
                    if len(text) > 8:  # Minimum reasonable GPS text length
                        lat, lon = parse_gps_coordinates(text, quiet_mode=True)
                        if lat is not None and lon is not None:
                            return lat, lon, text
                except Exception:
                    continue
            
        except Exception as e:
            pass
        
        return None, None, ""
    
    def _try_easyocr_gps(self, image: np.ndarray, context: str) -> Tuple[Optional[float], Optional[float], str]:
        """Try EasyOCR for GPS extraction"""
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            results = reader.readtext(image, detail=0, paragraph=True)
            
            if results:
                text = ' '.join(results)
                lat, lon = parse_gps_coordinates(text, quiet_mode=True)
                if lat is not None and lon is not None:
                    return lat, lon, text
            
        except Exception as e:
            pass
        
        return None, None, ""
    
    def _update_gps_stats(self, success: bool, processing_time: float):
        """Update GPS processing statistics"""
        self.processing_stats['total_images'] += 1
        if success:
            self.processing_stats['successful_gps_extractions'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['avg_processing_time']
        total_images = self.processing_stats['total_images']
        self.processing_stats['avg_processing_time'] = \
            ((current_avg * (total_images - 1)) + processing_time) / total_images
    
    def extract_banner_text(self, banner_image: np.ndarray) -> Dict[str, Any]:
        """
        COMPLETE banner text extraction with full Hindi and English support.
        Uses character-based language detection instead of polyglot.
        Automatically detects language and uses appropriate OCR engine.
        """
        start_time = time.time()
        
        result = {
            'original_text': '',
            'translated_text': '',
            'detected_language': 'en',
            'content_category': ContentCategory.PURE_ENGLISH.value,
            'confidence': 0.0,
            'processing_time': 0.0,
            'engines_used': [],
            'language_analysis': {},
            'preprocessing_info': {},
            'polyglot_free': True  # Flag to indicate polyglot-free processing
        }
        
        try:
            # Optimize image size for CPU
            optimized_image = optimize_image_for_cpu_processing(banner_image, max_dimension=800)
            
            # Prepare image variants for different scripts
            image_variants = self._prepare_image_variants(optimized_image)
            
            # Try extraction with different engines and configurations
            best_extraction = self._extract_with_best_engine(image_variants)
            
            if best_extraction['text']:
                # Analyze the extracted text using polyglot-free methods
                text_analysis = self._analyze_extracted_text_polyglot_free(best_extraction['text'])
                
                # Process based on detected language using character-based detection
                processed_result = self._process_text_by_language_polyglot_free(
                    best_extraction['text'], 
                    text_analysis
                )
                
                # Update result
                result.update(processed_result)
                result['engines_used'] = best_extraction['engines_used']
                result['confidence'] = best_extraction['confidence']
                result['language_analysis'] = text_analysis
                
                # Update extraction stats
                category = text_analysis['language_info']['category']
                if category == ContentCategory.PURE_HINDI.value or category == ContentCategory.HINDI_DOMINANT.value:
                    self.processing_stats['hindi_extractions'] += 1
                elif category == ContentCategory.PURE_ENGLISH.value or category == ContentCategory.ENGLISH_DOMINANT.value:
                    self.processing_stats['english_extractions'] += 1
                else:
                    self.processing_stats['mixed_extractions'] += 1
            
            result['processing_time'] = time.time() - start_time
            
        except Exception as e:
            print(f"[ERROR] Banner text extraction error: {e}")
            result['processing_time'] = time.time() - start_time
            result['error'] = str(e)
        
        return result
    
    def _analyze_extracted_text_polyglot_free(self, text: str) -> Dict[str, Any]:
        """
        Analyze extracted text for language composition and quality.
        Uses character-based detection instead of polyglot.
        """
        analysis = {
            'original_length': len(text),
            'word_count': len(text.split()),
            'language_info': detect_text_language(text),  # Uses character-based detection from utils
            'has_hindi': False,
            'has_english': False,
            'preprocessing_applied': [],
            'detection_method': 'character_based_polyglot_free'
        }
        
        try:
            # Check for Hindi and English content
            analysis['has_hindi'] = analysis['language_info']['hindi_percentage'] > 0
            analysis['has_english'] = analysis['language_info']['english_percentage'] > 0
            
            # Additional analysis
            analysis['mixed_script'] = analysis['has_hindi'] and analysis['has_english']
            analysis['dominant_script'] = 'hindi' if analysis['language_info']['hindi_percentage'] > 50 else 'english'
            
            # Character distribution analysis
            analysis['character_distribution'] = self._analyze_character_distribution(text)
            
        except Exception as e:
            print(f"[ERROR] Text analysis error: {e}")
        
        return analysis
    
    def _analyze_character_distribution(self, text: str) -> Dict[str, int]:
        """Analyze character distribution for better language detection"""
        distribution = {
            'devanagari_chars': 0,
            'latin_chars': 0,
            'numeric_chars': 0,
            'punctuation_chars': 0,
            'space_chars': 0,
            'other_chars': 0
        }
        
        for char in text:
            code_point = ord(char)
            
            if 0x0900 <= code_point <= 0x097F:  # Devanagari
                distribution['devanagari_chars'] += 1
            elif 0x0041 <= code_point <= 0x007A or 0x0061 <= code_point <= 0x007A:  # Latin
                distribution['latin_chars'] += 1
            elif char.isdigit():
                distribution['numeric_chars'] += 1
            elif char.isspace():
                distribution['space_chars'] += 1
            elif char in '.,!?;:()[]{}"\'-':
                distribution['punctuation_chars'] += 1
            else:
                distribution['other_chars'] += 1
        
        return distribution
    
    def _process_text_by_language_polyglot_free(self, text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process text based on detected language composition.
        Uses polyglot-free methods for all processing.
        """
        result = {
            'original_text': text,
            'translated_text': '',
            'detected_language': 'en',
            'content_category': ContentCategory.PURE_ENGLISH.value
        }
        
        try:
            lang_info = analysis['language_info']
            category = lang_info['category']
            
            result['content_category'] = category
            
            if category in [ContentCategory.PURE_HINDI.value, ContentCategory.HINDI_DOMINANT.value]:
                # Hindi processing using polyglot-free methods
                result['detected_language'] = 'hi'
                
                # Clean and normalize Hindi text using utils functions
                cleaned_hindi = clean_mixed_language_text(text)
                normalized_hindi = normalize_hindi_text(cleaned_hindi)
                
                result['translated_text'] = normalized_hindi
                
                # Update engine cache for Hindi content
                OCR_ENGINE_CACHE['language_engine_preference']['hindi'] = \
                    OCR_ENGINE_CACHE.get('last_successful_engine', 'tesseract')
                
                print(f"[HINDI] Processed as Hindi: {normalized_hindi[:50]}{'...' if len(normalized_hindi) > 50 else ''}")
                
            else:
                # English processing
                result['detected_language'] = 'en'
                
                # Clean English text
                cleaned_english = re.sub(r'[^\w\s%.-]', ' ', text)
                cleaned_english = re.sub(r'\s+', ' ', cleaned_english.strip().lower())
                
                result['translated_text'] = cleaned_english
                
                # Update engine cache for English content
                OCR_ENGINE_CACHE['language_engine_preference']['english'] = \
                    OCR_ENGINE_CACHE.get('last_successful_engine', 'tesseract')
                
                print(f"🔤 Processed as English: {cleaned_english[:50]}{'...' if len(cleaned_english) > 50 else ''}")
            
            # Update global engine cache
            if result['translated_text'] and len(result['translated_text']) > 5:
                OCR_ENGINE_CACHE['last_successful_engine'] = \
                    OCR_ENGINE_CACHE.get('last_successful_engine', 'tesseract')
            
        except Exception as e:
            print(f"[ERROR] Text processing error: {e}")
            result['translated_text'] = text.lower()
        
        return result
    
    def _prepare_image_variants(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Prepare different image variants optimized for different scripts"""
        variants = {}
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Original (good for clean text)
            variants['original'] = gray
            
            # Enhanced contrast (good for faded text)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
            variants['enhanced'] = enhanced
            
            # Binary threshold (good for high contrast text)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants['binary'] = binary
            
            # Adaptive threshold (good for varying lighting)
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            variants['adaptive'] = adaptive
            
            # Morphologically cleaned (good for noisy text)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            variants['morphological'] = morph
            
            # Upscaled (good for small text, especially Hindi)
            h, w = gray.shape
            if max(h, w) < 400:
                upscaled = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
                variants['upscaled'] = upscaled
            
        except Exception as e:
            print(f"[ERROR] Error preparing image variants: {e}")
            variants['original'] = image
        
        return variants
    
    def _extract_with_best_engine(self, image_variants: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract text using the best available engine for the content"""
        best_result = {
            'text': '',
            'confidence': 0.0,
            'engines_used': [],
            'variant_used': 'original'
        }
        
        # Define engine priority based on availability and cache
        engine_priorities = self._get_engine_priorities()
        
        for engine_name in engine_priorities:
            if engine_name not in self.available_engines:
                continue
            
            # Try each image variant with this engine
            for variant_name, variant_image in image_variants.items():
                try:
                    extraction_result = self._extract_with_engine(
                        variant_image, engine_name, variant_name
                    )
                    
                    if (extraction_result['confidence'] > best_result['confidence'] and 
                        len(extraction_result['text'].strip()) > len(best_result['text'].strip())):
                        
                        best_result = extraction_result
                        best_result['engines_used'] = [engine_name]
                        best_result['variant_used'] = variant_name
                        
                        # Early termination if we get very good results
                        if best_result['confidence'] > 85:
                            return best_result
                    
                except Exception as e:
                    print(f"[ERROR] Error with {engine_name} on {variant_name}: {e}")
                    continue
        
        return best_result
    
    def _get_engine_priorities(self) -> List[str]:
        """Get OCR engines in priority order based on cache and availability"""
        # Check for cached successful engine
        if OCR_ENGINE_CACHE['last_successful_engine'] in self.available_engines:
            cached_engine = OCR_ENGINE_CACHE['last_successful_engine']
            other_engines = [e for e in self.available_engines if e != cached_engine]
            return [cached_engine] + other_engines
        
        # Default priority: Tesseract -> EasyOCR -> PaddleOCR
        priorities = ['tesseract', 'easyocr', 'paddleocr']
        return [e for e in priorities if e in self.available_engines]
    
    def _extract_with_engine(self, image: np.ndarray, engine_name: str, variant_name: str) -> Dict[str, Any]:
        """Extract text with specific OCR engine"""
        result = {
            'text': '',
            'confidence': 0.0,
            'engine': engine_name
        }
        
        try:
            if engine_name == 'tesseract':
                result = self._extract_with_tesseract(image, variant_name)
            elif engine_name == 'easyocr':
                result = self._extract_with_easyocr(image, variant_name)
            elif engine_name == 'paddleocr':
                result = self._extract_with_paddleocr(image, variant_name)
            
            result['engine'] = engine_name
            
        except Exception as e:
            print(f"[ERROR] {engine_name} extraction failed: {e}")
        
        return result
    
    def _extract_with_tesseract(self, image: np.ndarray, variant_name: str) -> Dict[str, Any]:
        """Extract text using Tesseract with Hindi support"""
        try:
            import pytesseract
            
            # Try different language combinations
            lang_configs = [
                ('hin+eng', '--psm 6 --oem 3'),  # Hindi + English
                ('hin', '--psm 6 --oem 3'),      # Hindi only
                ('eng', '--psm 6 --oem 3'),      # English only
                ('hin+eng', '--psm 7 --oem 3'),  # Single line mode
                ('hin+eng', '--psm 8 --oem 3'),  # Single word mode
            ]
            
            best_result = {'text': '', 'confidence': 0.0}
            
            for lang, config in lang_configs:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(
                        image, lang=lang, config=config
                    ).strip()
                    
                    if len(text) > len(best_result['text']):
                        # Calculate confidence based on text length and content
                        confidence = self._calculate_text_confidence(text, variant_name)
                        if confidence > best_result['confidence']:
                            best_result = {
                                'text': text,
                                'confidence': confidence
                            }
                    
                    # Early termination for very good results
                    if best_result['confidence'] > 80 and len(best_result['text']) > 10:
                        break
                        
                except Exception as e:
                    continue
            
            return best_result
            
        except Exception as e:
            print(f"[ERROR] Tesseract error: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def _extract_with_easyocr(self, image: np.ndarray, variant_name: str) -> Dict[str, Any]:
        """Extract text using EasyOCR with Hindi support"""
        try:
            import easyocr
            
            # Initialize reader with Hindi and English
            reader = easyocr.Reader(['hi', 'en'], gpu=False, verbose=False)
            
            # Extract text
            results = reader.readtext(image, detail=1, paragraph=False)
            
            if results:
                # Combine all detected text
                combined_text = ' '.join([result[1] for result in results])
                
                # Calculate average confidence
                avg_confidence = sum([result[2] for result in results]) / len(results) * 100
                
                return {
                    'text': combined_text.strip(),
                    'confidence': min(avg_confidence, 95)  # Cap at 95%
                }
            
            return {'text': '', 'confidence': 0.0}
            
        except Exception as e:
            print(f"[ERROR] EasyOCR error: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def _extract_with_paddleocr(self, image: np.ndarray, variant_name: str) -> Dict[str, Any]:
        """Extract text using PaddleOCR with Hindi support"""
        try:
            import paddleocr
            
            # Initialize PaddleOCR
            ocr = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang='hi',  # Hindi
                use_gpu=False,
                show_log=False
            )
            
            # Extract text
            results = ocr.ocr(image, cls=True)
            
            if results and results[0]:
                # Combine all detected text
                combined_text = ' '.join([line[1][0] for line in results[0]])
                
                # Calculate average confidence
                confidences = [line[1][1] for line in results[0]]
                avg_confidence = sum(confidences) / len(confidences) * 100 if confidences else 0
                
                return {
                    'text': combined_text.strip(),
                    'confidence': min(avg_confidence, 95)
                }
            
            return {'text': '', 'confidence': 0.0}
            
        except Exception as e:
            print(f"[ERROR] PaddleOCR error: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def _calculate_text_confidence(self, text: str, variant_name: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text:
            return 0.0
        
        try:
            confidence = 50.0  # Base confidence
            
            # Length bonus (longer text usually better)
            length_bonus = min(len(text) * 2, 30)
            confidence += length_bonus
            
            # Character variety bonus
            unique_chars = len(set(text))
            variety_bonus = min(unique_chars, 15)
            confidence += variety_bonus
            
            # Penalize excessive special characters
            special_chars = len([c for c in text if not c.isalnum() and c not in ' \n\t'])
            if special_chars > len(text) * 0.3:
                confidence -= 20
            
            # Bonus for mixed Hindi-English content using character-based detection
            lang_info = detect_text_language(text)  # This uses character-based detection
            if lang_info['hindi_percentage'] > 0 and lang_info['english_percentage'] > 0:
                confidence += 10
            
            # Variant-specific adjustments
            if variant_name == 'enhanced':
                confidence += 5
            elif variant_name == 'upscaled':
                confidence += 3
            
            return max(0, min(confidence, 95))
            
        except Exception as e:
            return 50.0
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        total = self.processing_stats['total_images']
        
        return {
            'gps_extraction': {
                'total_processed': total,
                'successful_extractions': self.processing_stats['successful_gps_extractions'],
                'success_rate': (self.processing_stats['successful_gps_extractions'] / total * 100) if total > 0 else 0,
                'avg_processing_time': self.processing_stats['avg_processing_time'],
                'region_success_counts': GPS_EXTRACTION_CACHE['region_success_count'].copy()
            },
            'text_extraction': {
                'hindi_extractions': self.processing_stats['hindi_extractions'],
                'english_extractions': self.processing_stats['english_extractions'],
                'mixed_extractions': self.processing_stats['mixed_extractions'],
                'engine_success_counts': OCR_ENGINE_CACHE.get('engine_success_count', {}),
                'language_preferences': OCR_ENGINE_CACHE.get('language_engine_preference', {})
            },
            'engines_available': self.available_engines,
            'cache_info': {
                'gps_cache_size': len(GPS_EXTRACTION_CACHE.get('region_success_count', {})),
                'ocr_cache_size': len(OCR_ENGINE_CACHE.get('engine_success_count', {}))
            },
            'polyglot_free_processing': self.processing_stats['polyglot_free_processing'],
            'language_detection_method': self.language_detector['method']
        }
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        GPS_EXTRACTION_CACHE.clear()
        OCR_ENGINE_CACHE.clear()
        print("[CLEAN] All OCR caches cleared")
    
    def get_supported_features(self) -> Dict[str, bool]:
        """Get information about supported features"""
        return {
            'hindi_support': True,
            'english_support': True,
            'mixed_script_support': True,
            'gps_extraction': True,
            'polyglot_free_processing': True,
            'character_based_language_detection': True,
            'caching_enabled': True,
            'cpu_optimized': True,
            'multi_engine_ocr': len(self.available_engines) > 1
        }
