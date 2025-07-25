import pandas as pd
import os
from typing import List, Dict, Tuple, Optional, Any
import difflib
from datetime import datetime
from utils import (calculate_distance, is_expired, detect_text_language, normalize_hindi_text,
                  extract_hindi_keywords, compare_hindi_text_similarity, preprocess_text_for_matching,
                  ContentCategory)
import config
import re

# Try to import fuzzywuzzy for better fuzzy matching
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("[WARNING] fuzzywuzzy not available, using basic similarity")

# MATCHING CACHE - For processing multiple banners efficiently
MATCHING_CACHE = {
    'last_successful_threshold': None,
    'last_successful_method': None,
    'method_success_count': {},
    'hindi_patterns_cache': {},
    'english_patterns_cache': {}
}

class CSVMatcher:
    """
    COMPLETE CSV matcher with full Hindi support and efficient caching.
    Handles pure Hindi, pure English, and mixed language banner matching.
    """
    
    def __init__(self, csv_path: str = config.CSV_DATA_PATH):
        self.csv_path = csv_path
        
        # Load data with optimizations
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            try:
                self.data = pd.read_csv(csv_path, encoding='utf-8')
                self._convert_date_columns()
                self._preprocess_data_with_hindi_support()
                print(f"[OK] Loaded {len(self.data)} records from CSV with Hindi support")
            except Exception as e:
                print(f"[ERROR] Error loading CSV: {e}")
                self.data = pd.DataFrame()
        else:
            print(f"[WARNING] CSV file not found or empty: {csv_path}")
            self.data = pd.DataFrame()
        
        self.proximity_threshold = config.GPS_PROXIMITY_THRESHOLD
        
        # COMPLETE thresholds for all language combinations
        self.dynamic_thresholds = {
            'pure_english': 65.0,      # Pure English banners
            'english_dominant': 55.0,  # Mostly English with some Hindi
            'hindi_dominant': 45.0,    # Mostly Hindi with some English  
            'pure_hindi': 35.0,        # Pure Hindi banners (lower due to script complexity)
            'mixed_content': 50.0      # Balanced Hindi-English content
        }
        
        # Hindi-specific matching parameters
        self.hindi_matching_params = {
            'exact_match_bonus': 20,     # Bonus for exact Hindi word matches
            'partial_match_threshold': 3, # Minimum characters for partial matching
            'keyword_weight': 0.7,       # Weight for keyword matching vs full text
            'character_similarity_weight': 0.3  # Weight for character-level similarity
        }
        
        print(f"[LANG] CSV Matcher initialized with Hindi support - {len(self.data)} records")
    
    def _convert_date_columns(self):
        """Convert date columns efficiently with better error handling"""
        if 'expiry_date' not in self.data.columns:
            return
        
        date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']
        
        def parse_date_safe(date_str):
            if pd.isna(date_str):
                return None
            
            date_str = str(date_str).strip()
            
            # Handle common variations
            date_str = date_str.replace('/', '-').replace('.', '-')
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            return None
        
        try:
            self.data['expiry_date_parsed'] = self.data['expiry_date'].apply(parse_date_safe)
            success_rate = (self.data['expiry_date_parsed'].notna().sum() / len(self.data)) * 100
            print(f"[OK] Date parsing success rate: {success_rate:.1f}%")
        except Exception as e:
            print(f"[ERROR] Date conversion error: {e}")
    
    def _preprocess_data_with_hindi_support(self):
        """Preprocess CSV data with complete Hindi and English support"""
        if len(self.data) == 0:
            return
        
        try:
            print("[RELOAD] Preprocessing CSV data with Hindi support...")
            
            # Process company names
            if 'company_name' in self.data.columns:
                print("   Processing company names...")
                self.data['company_name_processed'] = self.data['company_name'].apply(
                    self._preprocess_single_field
                )
            
            # Process banner text
            if 'banner_text' in self.data.columns:
                print("   Processing banner text...")
                self.data['banner_text_processed'] = self.data['banner_text'].apply(
                    self._preprocess_single_field
                )
            
            # Create combined search field for better matching
            self._create_combined_search_fields()
            
            print("[OK] Data preprocessing with Hindi support completed")
            
        except Exception as e:
            print(f"[ERROR] Data preprocessing error: {e}")
    
    def _preprocess_single_field(self, text) -> Dict[str, Any]:
        """Preprocess a single text field with complete language analysis"""
        if pd.isna(text) or not str(text).strip():
            return {
                'original': '',
                'cleaned': '',
                'normalized': '',
                'keywords': [],
                'language_info': detect_text_language(''),
                'hindi_keywords': [],
                'english_keywords': [],
                'match_ready': ''
            }
        
        try:
            text_str = str(text).strip()
            
            # Complete preprocessing using utils function
            preprocessed = preprocess_text_for_matching(text_str)
            
            # Additional Hindi-specific processing
            lang_info = preprocessed['language_info']
            
            # Extract language-specific keywords
            hindi_keywords = []
            english_keywords = []
            
            if lang_info['hindi_percentage'] > 0:
                hindi_keywords = extract_hindi_keywords(text_str)
            
            if lang_info['english_percentage'] > 0:
                # Extract English keywords
                english_words = re.findall(r'[a-zA-Z]+', text_str.lower())
                english_keywords = [word for word in english_words if len(word) >= 3]
            
            # Create comprehensive result
            result = preprocessed.copy()
            result.update({
                'hindi_keywords': hindi_keywords,
                'english_keywords': english_keywords,
                'has_hindi': lang_info['hindi_percentage'] > 0,
                'has_english': lang_info['english_percentage'] > 0,
                'dominant_language': 'hindi' if lang_info['hindi_percentage'] > 50 else 'english'
            })
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Single field preprocessing error: {e}")
            return {
                'original': str(text),
                'cleaned': str(text),
                'normalized': str(text).lower(),
                'keywords': str(text).split(),
                'language_info': detect_text_language(str(text)),
                'hindi_keywords': [],
                'english_keywords': [],
                'match_ready': str(text).lower()
            }
    
    def _create_combined_search_fields(self):
        """Create combined search fields for better matching performance"""
        try:
            combined_data = []
            
            for _, row in self.data.iterrows():
                combined_text = ""
                combined_keywords = []
                combined_hindi_keywords = []
                combined_english_keywords = []
                
                # Combine company name and banner text
                for field in ['company_name_processed', 'banner_text_processed']:
                    if field in row and isinstance(row[field], dict):
                        field_data = row[field]
                        combined_text += " " + field_data.get('normalized', '')
                        combined_keywords.extend(field_data.get('keywords', []))
                        combined_hindi_keywords.extend(field_data.get('hindi_keywords', []))
                        combined_english_keywords.extend(field_data.get('english_keywords', []))
                
                combined_data.append({
                    'combined_text': combined_text.strip(),
                    'combined_keywords': list(set(combined_keywords)),
                    'combined_hindi_keywords': list(set(combined_hindi_keywords)),
                    'combined_english_keywords': list(set(combined_english_keywords))
                })
            
            # Add combined data to dataframe
            combined_df = pd.DataFrame(combined_data)
            for col in combined_df.columns:
                self.data[col] = combined_df[col]
            
            print("[OK] Combined search fields created")
            
        except Exception as e:
            print(f"[ERROR] Error creating combined fields: {e}")
    
    def find_nearby_banners(self, lat: float, lon: float) -> pd.DataFrame:
        """Find nearby banners with optimized distance calculation"""
        if len(self.data) == 0:
            return pd.DataFrame()
        
        try:
            # Check if we have required columns
            if 'latitude' not in self.data.columns or 'longitude' not in self.data.columns:
                print("[WARNING] GPS columns not found, returning all data")
                return self.data.copy()
            
            # Calculate distances efficiently
            distances = []
            for _, row in self.data.iterrows():
                try:
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        distance = calculate_distance(
                            lat, lon, 
                            float(row['latitude']), 
                            float(row['longitude'])
                        )
                        distances.append(distance)
                    else:
                        distances.append(float('inf'))
                except (ValueError, TypeError):
                    distances.append(float('inf'))
            
            # Filter by proximity
            nearby_mask = [d <= self.proximity_threshold for d in distances]
            nearby_banners = self.data[nearby_mask].copy()
            
            if len(nearby_banners) > 0:
                nearby_banners['distance'] = [d for d, m in zip(distances, nearby_mask) if m]
                nearby_banners = nearby_banners.sort_values('distance')
            
            print(f"[TARGET] Found {len(nearby_banners)} nearby banners within {self.proximity_threshold}m")
            return nearby_banners
            
        except Exception as e:
            print(f"[ERROR] Error finding nearby banners: {e}")
            return self.data.copy()
    
    def match_banner_content(self, banner_text: Dict[str, Any], nearby_banners: pd.DataFrame) -> Dict[str, Any]:
        """
        COMPLETE banner matching with full Hindi support.
        Handles pure Hindi, pure English, and mixed content efficiently.
        """
        if len(nearby_banners) == 0 or not banner_text.get('translated_text'):
            return {
                'matched': False,
                'message': "No registered banners in vicinity or no text extracted",
                'banner_data': None,
                'match_details': {
                    'extracted_text': banner_text.get('translated_text', ''),
                    'content_category': banner_text.get('content_category', 'unknown'),
                    'nearby_count': len(nearby_banners)
                }
            }
        
        extracted_text = banner_text.get('translated_text', '').strip()
        content_category = banner_text.get('content_category', 'pure_english')
        original_text = banner_text.get('original_text', '')
        
        print(f"[TARGET] Matching banner content:")
        print(f"   Extracted: '{extracted_text[:50]}{'...' if len(extracted_text) > 50 else ''}'")
        print(f"   Category: {content_category}")
        print(f"   Nearby banners: {len(nearby_banners)}")
        
        # Analyze the extracted text for matching strategy
        text_analysis = preprocess_text_for_matching(original_text)
        
        # Get appropriate threshold
        threshold = self._get_threshold_for_content(content_category, text_analysis)
        
        # PRIORITY 1: Try cached successful method first
        if MATCHING_CACHE['last_successful_method']:
            cached_method = MATCHING_CACHE['last_successful_method']
            cached_threshold = MATCHING_CACHE['last_successful_threshold']
            
            print(f"[START] Trying cached method: {cached_method}")
            
            best_match, best_score = self._try_matching_method(
                text_analysis, nearby_banners, cached_method, cached_threshold or threshold
            )
            
            if best_match is not None:
                # Update cache success count
                MATCHING_CACHE['method_success_count'][cached_method] = \
                    MATCHING_CACHE['method_success_count'].get(cached_method, 0) + 1
                
                return self._create_comprehensive_match_result(
                    best_match, best_score, cached_threshold or threshold, 
                    content_category, text_analysis, cached_method
                )
        
        # PRIORITY 2: Try methods based on content type and historical success
        matching_methods = self._get_prioritized_matching_methods(content_category)
        
        for method_name, method_func in matching_methods:
            if method_name == MATCHING_CACHE.get('last_successful_method'):
                continue  # Skip already tried method
            
            print(f"[SEARCH] Trying method: {method_name}")
            
            best_match, best_score = method_func(text_analysis, nearby_banners, threshold)
            
            if best_match is not None and best_score >= threshold:
                # Update cache
                MATCHING_CACHE['last_successful_method'] = method_name
                MATCHING_CACHE['last_successful_threshold'] = threshold
                MATCHING_CACHE['method_success_count'][method_name] = \
                    MATCHING_CACHE['method_success_count'].get(method_name, 0) + 1
                
                print(f"[OK] Match found via {method_name}: {best_score:.1f}%")
                return self._create_comprehensive_match_result(
                    best_match, best_score, threshold, content_category, text_analysis, method_name
                )
        
        print(f"[ERROR] No match found (best threshold: {threshold}%)")
        return {
            'matched': False,
            'message': f"No matching banner found (threshold: {threshold}%)",
            'best_score': 0,
            'threshold_used': threshold,
            'content_category': content_category,
            'match_details': {
                'extracted_text': extracted_text,
                'methods_tried': [method[0] for method in matching_methods],
                'nearby_count': len(nearby_banners),
                'text_analysis': text_analysis
            }
        }
    
    def _get_threshold_for_content(self, content_category: str, text_analysis: Dict[str, Any]) -> float:
        """Get appropriate threshold based on content type and analysis"""
        try:
            # Base threshold
            threshold = self.dynamic_thresholds.get(content_category, 55.0)
            
            # Adjust based on text characteristics
            lang_info = text_analysis.get('language_info', {})
            
            # Lower threshold for pure Hindi (harder to match exactly)
            if lang_info.get('hindi_percentage', 0) > 80:
                threshold = max(threshold - 10, 25)
            
            # Slightly higher threshold for pure English (easier to match)
            elif lang_info.get('english_percentage', 0) > 80:
                threshold = min(threshold + 5, 75)
            
            # Adjust for text length (shorter text harder to match reliably)
            text_length = len(text_analysis.get('normalized', ''))
            if text_length < 10:
                threshold = max(threshold - 15, 20)
            elif text_length > 50:
                threshold = min(threshold + 5, 80)
            
            return threshold
            
        except Exception as e:
            print(f"[ERROR] Threshold calculation error: {e}")
            return 50.0
    
    def _get_prioritized_matching_methods(self, content_category: str) -> List[Tuple[str, callable]]:
        """Get matching methods prioritized by content type and historical success"""
        
        # Define methods based on content category
        if content_category in ['pure_hindi', 'hindi_dominant']:
            base_methods = [
                ('hindi_exact_matching', self._hindi_exact_matching),
                ('hindi_keyword_matching', self._hindi_keyword_matching),
                ('hindi_fuzzy_matching', self._hindi_fuzzy_matching),
                ('mixed_script_matching', self._mixed_script_matching),
                ('character_similarity_matching', self._character_similarity_matching),
            ]
        elif content_category in ['pure_english', 'english_dominant']:
            base_methods = [
                ('english_exact_matching', self._english_exact_matching),
                ('english_fuzzy_matching', self._english_fuzzy_matching),
                ('english_keyword_matching', self._english_keyword_matching),
                ('partial_word_matching', self._partial_word_matching),
            ]
        else:  # Mixed content
            base_methods = [
                ('mixed_script_matching', self._mixed_script_matching),
                ('combined_keyword_matching', self._combined_keyword_matching),
                ('hindi_fuzzy_matching', self._hindi_fuzzy_matching),
                ('english_fuzzy_matching', self._english_fuzzy_matching),
                ('character_similarity_matching', self._character_similarity_matching),
            ]
        
        # Sort by success count (highest first)
        success_counts = MATCHING_CACHE['method_success_count']
        return sorted(base_methods, key=lambda x: success_counts.get(x[0], 0), reverse=True)
    
    def _try_matching_method(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                           method: str, threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Try a specific matching method"""
        method_map = {
            'hindi_exact_matching': self._hindi_exact_matching,
            'hindi_keyword_matching': self._hindi_keyword_matching,
            'hindi_fuzzy_matching': self._hindi_fuzzy_matching,
            'english_exact_matching': self._english_exact_matching,
            'english_fuzzy_matching': self._english_fuzzy_matching,
            'english_keyword_matching': self._english_keyword_matching,
            'mixed_script_matching': self._mixed_script_matching,
            'combined_keyword_matching': self._combined_keyword_matching,
            'partial_word_matching': self._partial_word_matching,
            'character_similarity_matching': self._character_similarity_matching,
        }
        
        if method in method_map:
            return method_map[method](text_analysis, nearby_banners, threshold)
        
        return None, 0.0
    
    def _hindi_exact_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                            threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Exact Hindi word matching with normalization"""
        best_match = None
        best_score = 0.0
        
        hindi_keywords = set(text_analysis.get('keywords', []))
        if not hindi_keywords:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                # Check company name
                if 'company_name_processed' in banner and isinstance(banner['company_name_processed'], dict):
                    company_data = banner['company_name_processed']
                    company_hindi_keywords = set(company_data.get('hindi_keywords', []))
                    
                    if company_hindi_keywords and hindi_keywords:
                        intersection = len(hindi_keywords.intersection(company_hindi_keywords))
                        union = len(hindi_keywords.union(company_hindi_keywords))
                        score = (intersection / union) * 100 if union > 0 else 0
                        
                        # Bonus for exact matches
                        exact_matches = len(hindi_keywords.intersection(company_hindi_keywords))
                        if exact_matches > 0:
                            score += self.hindi_matching_params['exact_match_bonus']
                        
                        if score > best_score:
                            best_score = score
                            best_match = banner
                
                # Check banner text
                if 'banner_text_processed' in banner and isinstance(banner['banner_text_processed'], dict):
                    banner_data = banner['banner_text_processed']
                    banner_hindi_keywords = set(banner_data.get('hindi_keywords', []))
                    
                    if banner_hindi_keywords and hindi_keywords:
                        intersection = len(hindi_keywords.intersection(banner_hindi_keywords))
                        union = len(hindi_keywords.union(banner_hindi_keywords))
                        score = (intersection / union) * 100 if union > 0 else 0
                        
                        # Bonus for exact matches
                        exact_matches = len(hindi_keywords.intersection(banner_hindi_keywords))
                        if exact_matches > 0:
                            score += self.hindi_matching_params['exact_match_bonus']
                        
                        if score > best_score:
                            best_score = score
                            best_match = banner
                
                # Early termination for very good matches
                if best_score >= 90:
                    break
            
        except Exception as e:
            print(f"[ERROR] Hindi exact matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _hindi_keyword_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                              threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Hindi keyword-based matching with partial matches"""
        best_match = None
        best_score = 0.0
        
        hindi_keywords = text_analysis.get('keywords', [])
        if not hindi_keywords:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check all relevant fields
                for field in ['company_name_processed', 'banner_text_processed']:
                    if field in banner and isinstance(banner[field], dict):
                        field_data = banner[field]
                        field_keywords = field_data.get('hindi_keywords', [])
                        
                        if field_keywords:
                            # Calculate keyword overlap
                            matches = 0
                            partial_matches = 0
                            
                            for ext_keyword in hindi_keywords:
                                for field_keyword in field_keywords:
                                    # Exact match
                                    if ext_keyword == field_keyword:
                                        matches += 1
                                        break
                                    # Partial match (for longer words)
                                    elif (len(ext_keyword) >= self.hindi_matching_params['partial_match_threshold'] and
                                          len(field_keyword) >= self.hindi_matching_params['partial_match_threshold']):
                                        if ext_keyword in field_keyword or field_keyword in ext_keyword:
                                            partial_matches += 0.5
                                            break
                            
                            total_matches = matches + partial_matches
                            score = (total_matches / len(hindi_keywords)) * 100 if hindi_keywords else 0
                            
                            max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 85:
                    break
            
        except Exception as e:
            print(f"[ERROR] Hindi keyword matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _hindi_fuzzy_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                            threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Fuzzy matching for Hindi text using custom similarity"""
        best_match = None
        best_score = 0.0
        
        normalized_text = text_analysis.get('normalized', '')
        if not normalized_text:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check company name
                if 'company_name_processed' in banner and isinstance(banner['company_name_processed'], dict):
                    company_data = banner['company_name_processed']
                    company_text = company_data.get('normalized', '')
                    
                    if company_text:
                        score = compare_hindi_text_similarity(normalized_text, company_text)
                        max_banner_score = max(max_banner_score, score)
                
                # Check banner text
                if 'banner_text_processed' in banner and isinstance(banner['banner_text_processed'], dict):
                    banner_data = banner['banner_text_processed']
                    banner_text = banner_data.get('normalized', '')
                    
                    if banner_text:
                        score = compare_hindi_text_similarity(normalized_text, banner_text)
                        max_banner_score = max(max_banner_score, score)
                
                # Check combined text
                if 'combined_text' in banner:
                    combined_text = str(banner['combined_text'])
                    if combined_text:
                        score = compare_hindi_text_similarity(normalized_text, combined_text)
                        max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 80:
                    break
            
        except Exception as e:
            print(f"[ERROR] Hindi fuzzy matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _english_exact_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                              threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Exact English word matching"""
        best_match = None
        best_score = 0.0
        
        english_keywords = set([kw.lower() for kw in text_analysis.get('keywords', [])])
        if not english_keywords:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check company name
                if 'company_name_processed' in banner and isinstance(banner['company_name_processed'], dict):
                    company_data = banner['company_name_processed']
                    company_keywords = set([kw.lower() for kw in company_data.get('english_keywords', [])])
                    
                    if company_keywords and english_keywords:
                        intersection = len(english_keywords.intersection(company_keywords))
                        union = len(english_keywords.union(company_keywords))
                        score = (intersection / union) * 100 if union > 0 else 0
                        max_banner_score = max(max_banner_score, score)
                
                # Check banner text
                if 'banner_text_processed' in banner and isinstance(banner['banner_text_processed'], dict):
                    banner_data = banner['banner_text_processed']
                    banner_keywords = set([kw.lower() for kw in banner_data.get('english_keywords', [])])
                    
                    if banner_keywords and english_keywords:
                        intersection = len(english_keywords.intersection(banner_keywords))
                        union = len(english_keywords.union(banner_keywords))
                        score = (intersection / union) * 100 if union > 0 else 0
                        max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 90:
                    break
            
        except Exception as e:
            print(f"[ERROR] English exact matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _english_fuzzy_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                              threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Fuzzy matching for English text"""
        best_match = None
        best_score = 0.0
        
        normalized_text = text_analysis.get('normalized', '').lower()
        if not normalized_text:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check company name
                if 'company_name_processed' in banner and isinstance(banner['company_name_processed'], dict):
                    company_data = banner['company_name_processed']
                    company_text = company_data.get('normalized', '').lower()
                    
                    if company_text:
                        if FUZZYWUZZY_AVAILABLE:
                            score = fuzz.partial_ratio(normalized_text, company_text)
                        else:
                            score = difflib.SequenceMatcher(None, normalized_text, company_text).ratio() * 100
                        max_banner_score = max(max_banner_score, score)
                
                # Check banner text
                if 'banner_text_processed' in banner and isinstance(banner['banner_text_processed'], dict):
                    banner_data = banner['banner_text_processed']
                    banner_text = banner_data.get('normalized', '').lower()
                    
                    if banner_text:
                        if FUZZYWUZZY_AVAILABLE:
                            score = fuzz.partial_ratio(normalized_text, banner_text)
                        else:
                            score = difflib.SequenceMatcher(None, normalized_text, banner_text).ratio() * 100
                        max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 85:
                    break
            
        except Exception as e:
            print(f"[ERROR] English fuzzy matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _english_keyword_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                                threshold: float) -> Tuple[Optional[pd.Series], float]:
        """English keyword-based matching with partial matches"""
        best_match = None
        best_score = 0.0
        
        english_keywords = [kw.lower() for kw in text_analysis.get('keywords', [])]
        if not english_keywords:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check all relevant fields
                for field in ['company_name_processed', 'banner_text_processed']:
                    if field in banner and isinstance(banner[field], dict):
                        field_data = banner[field]
                        field_keywords = [kw.lower() for kw in field_data.get('english_keywords', [])]
                        
                        if field_keywords:
                            matches = 0
                            partial_matches = 0
                            
                            for ext_keyword in english_keywords:
                                for field_keyword in field_keywords:
                                    # Exact match
                                    if ext_keyword == field_keyword:
                                        matches += 1
                                        break
                                    # Partial match
                                    elif len(ext_keyword) >= 4 and len(field_keyword) >= 4:
                                        if ext_keyword in field_keyword or field_keyword in ext_keyword:
                                            partial_matches += 0.5
                                            break
                            
                            total_matches = matches + partial_matches
                            score = (total_matches / len(english_keywords)) * 100 if english_keywords else 0
                            max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 85:
                    break
            
        except Exception as e:
            print(f"[ERROR] English keyword matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _mixed_script_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                             threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Matching for mixed Hindi-English content"""
        best_match = None
        best_score = 0.0
        
        try:
            # Get both Hindi and English keywords
            hindi_keywords = set(text_analysis.get('keywords', []))
            
            # Extract English keywords from normalized text
            normalized_text = text_analysis.get('normalized', '')
            english_words = re.findall(r'[a-zA-Z]+', normalized_text.lower())
            english_keywords = set([word for word in english_words if len(word) >= 3])
            
            if not hindi_keywords and not english_keywords:
                return None, 0.0
            
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check combined keywords
                if 'combined_hindi_keywords' in banner and 'combined_english_keywords' in banner:
                    banner_hindi = set(banner['combined_hindi_keywords'])
                    banner_english = set([kw.lower() for kw in banner['combined_english_keywords']])
                    
                    # Hindi score
                    hindi_score = 0.0
                    if hindi_keywords and banner_hindi:
                        hindi_intersection = len(hindi_keywords.intersection(banner_hindi))
                        hindi_union = len(hindi_keywords.union(banner_hindi))
                        hindi_score = (hindi_intersection / hindi_union) * 100 if hindi_union > 0 else 0
                    
                    # English score
                    english_score = 0.0
                    if english_keywords and banner_english:
                        english_intersection = len(english_keywords.intersection(banner_english))
                        english_union = len(english_keywords.union(banner_english))
                        english_score = (english_intersection / english_union) * 100 if english_union > 0 else 0
                    
                    # Combined score with weights
                    lang_info = text_analysis.get('language_info', {})
                    hindi_weight = lang_info.get('hindi_percentage', 0) / 100
                    english_weight = lang_info.get('english_percentage', 0) / 100
                    
                    total_weight = hindi_weight + english_weight
                    if total_weight > 0:
                        combined_score = (hindi_score * hindi_weight + english_score * english_weight) / total_weight
                        max_banner_score = max(max_banner_score, combined_score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 80:
                    break
            
        except Exception as e:
            print(f"[ERROR] Mixed script matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _combined_keyword_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                                 threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Combined keyword matching for all languages"""
        best_match = None
        best_score = 0.0
        
        all_keywords = set(text_analysis.get('keywords', []))
        if not all_keywords:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check combined keywords
                if 'combined_keywords' in banner:
                    banner_keywords = set(banner['combined_keywords'])
                    
                    if banner_keywords and all_keywords:
                        intersection = len(all_keywords.intersection(banner_keywords))
                        union = len(all_keywords.union(banner_keywords))
                        score = (intersection / union) * 100 if union > 0 else 0
                        max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 85:
                    break
            
        except Exception as e:
            print(f"[ERROR] Combined keyword matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _partial_word_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                             threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Partial word matching for approximate matches"""
        best_match = None
        best_score = 0.0
        
        normalized_text = text_analysis.get('normalized', '')
        words = [word for word in normalized_text.split() if len(word) >= 4]
        
        if not words:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check combined text
                if 'combined_text' in banner:
                    banner_text = str(banner['combined_text']).lower()
                    banner_words = [word for word in banner_text.split() if len(word) >= 4]
                    
                    if banner_words:
                        matches = 0
                        for word in words:
                            for banner_word in banner_words:
                                # Check if words contain each other
                                if word in banner_word or banner_word in word:
                                    matches += 1
                                    break
                                # Check if they share significant portion
                                elif len(word) >= 5 and len(banner_word) >= 5:
                                    similarity = difflib.SequenceMatcher(None, word, banner_word).ratio()
                                    if similarity >= 0.7:
                                        matches += similarity
                                        break
                        
                        score = (matches / len(words)) * 100 if words else 0
                        max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 75:
                    break
            
        except Exception as e:
            print(f"[ERROR] Partial word matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _character_similarity_matching(self, text_analysis: Dict[str, Any], nearby_banners: pd.DataFrame, 
                                     threshold: float) -> Tuple[Optional[pd.Series], float]:
        """Character-level similarity matching (last resort)"""
        best_match = None
        best_score = 0.0
        
        normalized_text = text_analysis.get('normalized', '')
        if len(normalized_text) < 5:
            return None, 0.0
        
        try:
            for _, banner in nearby_banners.iterrows():
                max_banner_score = 0.0
                
                # Check all text fields
                for field in ['company_name_processed', 'banner_text_processed']:
                    if field in banner and isinstance(banner[field], dict):
                        field_data = banner[field]
                        field_text = field_data.get('normalized', '')
                        
                        if field_text and len(field_text) >= 5:
                            # Character-level similarity
                            chars1 = set(normalized_text)
                            chars2 = set(field_text)
                            
                            if chars1 and chars2:
                                intersection = len(chars1.intersection(chars2))
                                union = len(chars1.union(chars2))
                                score = (intersection / union) * 100 if union > 0 else 0
                                
                                # Boost score if significant character overlap
                                if intersection >= min(len(chars1), len(chars2)) * 0.7:
                                    score *= 1.2
                                
                                max_banner_score = max(max_banner_score, score)
                
                if max_banner_score > best_score:
                    best_score = max_banner_score
                    best_match = banner
                
                # Early termination
                if best_score >= 70:
                    break
            
        except Exception as e:
            print(f"[ERROR] Character similarity matching error: {e}")
        
        return (best_match, best_score) if best_score >= threshold else (None, best_score)
    
    def _create_comprehensive_match_result(self, matched_banner: pd.Series, score: float, 
                                         threshold: float, content_category: str, 
                                         text_analysis: Dict[str, Any], method_used: str) -> Dict[str, Any]:
        """Create comprehensive match result with all details"""
        try:
            banner_dict = matched_banner.to_dict()
            
            # Check expiry status
            is_exp, days_diff, status_msg = self.is_expired(banner_dict)
            
            # Extract matched company info
            company_name = banner_dict.get('company_name', 'Unknown')
            banner_text = banner_dict.get('banner_text', '')
            location = banner_dict.get('location', 'Unknown')
            
            return {
                'matched': True,
                'banner_data': banner_dict,
                'match_score': score,
                'threshold_used': threshold,
                'content_category': content_category,
                'method_used': method_used,
                'is_expired': is_exp,
                'days_diff': days_diff,
                'status_message': status_msg,
                'match_details': {
                    'company_name': company_name,
                    'banner_text': banner_text,
                    'location': location,
                    'extracted_text': text_analysis.get('original', ''),
                    'language_analysis': text_analysis.get('language_info', {}),
                    'matching_method_effectiveness': {
                        method_used: score
                    }
                }
            }
        except Exception as e:
            print(f"[ERROR] Error creating comprehensive match result: {e}")
            return {
                'matched': False,
                'message': f"Error processing match: {e}",
                'banner_data': None
            }
    
    def is_expired(self, banner_data: Dict[str, Any]) -> Tuple[bool, int, str]:
        """Check banner expiry status efficiently"""
        try:
            if 'expiry_date_parsed' in banner_data and banner_data['expiry_date_parsed'] is not None:
                expiry_date = banner_data['expiry_date_parsed']
                current_date = datetime.now().date()
                days_diff = (expiry_date - current_date).days
                
                if days_diff < 0:
                    return True, abs(days_diff), f"EXPIRED ({abs(days_diff)} days ago)"
                elif days_diff == 0:
                    return False, 0, "EXPIRES TODAY"
                else:
                    return False, days_diff, f"VALID (Expires in {days_diff} days)"
            
            # Fallback to string parsing
            expiry_str = banner_data.get('expiry_date', '')
            return is_expired(str(expiry_str))
            
        except Exception as e:
            return True, 0, f"ERROR: {str(e)}"
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get comprehensive matching statistics"""
        return {
            'cache_info': {
                'last_successful_method': MATCHING_CACHE.get('last_successful_method'),
                'last_successful_threshold': MATCHING_CACHE.get('last_successful_threshold'),
                'method_success_counts': MATCHING_CACHE.get('method_success_count', {}),
                'total_cached_methods': len(MATCHING_CACHE.get('method_success_count', {}))
            },
            'data_info': {
                'total_banners': len(self.data),
                'has_company_name_processing': 'company_name_processed' in self.data.columns,
                'has_banner_text_processing': 'banner_text_processed' in self.data.columns,
                'has_combined_fields': 'combined_text' in self.data.columns
            },
            'thresholds': self.dynamic_thresholds.copy(),
            'hindi_parameters': self.hindi_matching_params.copy(),
            'proximity_threshold_meters': self.proximity_threshold
        }
    
    def clear_cache(self):
        """Clear matching cache to free memory"""
        MATCHING_CACHE.clear()
        MATCHING_CACHE.update({
            'last_successful_threshold': None,
            'last_successful_method': None,
            'method_success_count': {},
            'hindi_patterns_cache': {},
            'english_patterns_cache': {}
        })
        print("[CLEAN] Matching cache cleared")
    
    def debug_banner_data(self, banner_index: int = 0) -> Dict[str, Any]:
        """Debug function to inspect processed banner data"""
        if len(self.data) == 0 or banner_index >= len(self.data):
            return {'error': 'No data or invalid index'}
        
        banner = self.data.iloc[banner_index]
        
        debug_info = {
            'original_data': {
                'company_name': banner.get('company_name', ''),
                'banner_text': banner.get('banner_text', ''),
                'location': banner.get('location', ''),
                'expiry_date': banner.get('expiry_date', '')
            },
            'processed_data': {}
        }
        
        # Add processed fields if they exist
        for field in ['company_name_processed', 'banner_text_processed']:
            if field in banner and isinstance(banner[field], dict):
                debug_info['processed_data'][field] = banner[field]
        
        # Add combined fields
        for field in ['combined_text', 'combined_keywords', 'combined_hindi_keywords', 'combined_english_keywords']:
            if field in banner:
                debug_info['processed_data'][field] = banner[field]
        
        return debug_info
