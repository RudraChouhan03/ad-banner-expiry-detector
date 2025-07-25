#!/usr/bin/env python3
"""
Enhanced Ad Banner Expiry Detection System - Command Line Interface
Supports GPS extraction optimization and full Hindi text processing
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import cv2
import numpy as np

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import enhanced modules
try:
    import config
    from app.banner_detector import BannerDetector
    from app.ocr_processor import OCRProcessor
    from app.csv_matcher import CSVMatcher
    from app.hybrid_cropper import HybridCropper
    from app.utils import (
        generate_banner_id, create_directories, resize_image_for_display,
        detect_text_language, preprocess_text_for_matching, ContentCategory
    )
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please ensure all required modules are installed and paths are correct.")
    sys.exit(1)

class EnhancedBannerProcessor:
    """
    Enhanced Banner Processing System with GPS reliability and Hindi support
    """
    
    def __init__(self):
        print("[START] Initializing Enhanced Ad Banner Expiry Detection System...")
        
        # Initialize components
        try:
            self.banner_detector = BannerDetector()
            self.ocr_processor = OCRProcessor()
            self.csv_matcher = CSVMatcher()
            self.hybrid_cropper = HybridCropper()
            
            # Create required directories
            create_directories([
                config.RESULTS_DIR,
                config.AUTO_CROPS_DIR,
                config.MANUAL_CROPS_DIR,
                config.DETECTIONS_DIR,
                config.LOGS_DIR
            ])
            
            print("[OK] All components initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Initialization failed: {e}")
            sys.exit(1)
    
    def process_single_image(self, image_path: str, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Process a single image with enhanced GPS and Hindi detection
        """
        print(f"\n{'='*60}")
        print(f"[SEARCH] Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        results = {
            'image_path': image_path,
            'success': False,
            'gps_coordinates': None,
            'banners_processed': 0,
            'total_matches': 0,
            'processing_time': 0,
            'banners': []
        }
        
        try:
            # Step 1: GPS Extraction with enhanced reliability
            print("\n[LOCATION] Step 1: GPS Extraction (Enhanced)")
            lat, lon, gps_text = self.ocr_processor.extract_gps_from_image(image_path)
            
            if lat is not None and lon is not None:
                results['gps_coordinates'] = {'latitude': lat, 'longitude': lon}
                print(f"[OK] GPS Found: {lat:.6f}, {lon:.6f}")
                if args.debug_gps:
                    print(f"   GPS Text: '{gps_text[:50]}...'")
            else:
                print("[WARNING]  GPS coordinates not found")
                if args.skip_no_gps:
                    print("[ERROR] Skipping image due to missing GPS (--skip-no-gps enabled)")
                    return results
            
            # Step 2: Banner Detection
            print("\n[TARGET] Step 2: Banner Detection")
            detection_result = self.banner_detector.detect_banners(image_path)
            
            if not detection_result.get('success', False):
                print("[ERROR] No banners detected")
                return results
            
            detected_boxes = detection_result.get('boxes', [])
            print(f"[OK] Detected {len(detected_boxes)} banner(s)")
            
            # Step 3: Process Manual Crops (if provided)
            manual_crops = []
            if args.manual_crop:
                print("\n✂️  Step 3: Manual Cropping")
                manual_crops = self.hybrid_cropper.crop_banners_interactive(image_path)
                print(f"[OK] Created {len(manual_crops)} manual crops")
            
            # Step 4: Auto Crop Detected Banners
            print("\n✂️  Step 4: Auto Cropping")
            auto_crops = self.banner_detector.crop_banners(detection_result)
            print(f"[OK] Created {len(auto_crops)} auto crops")
            
            # Combine all crops
            all_crops = auto_crops + manual_crops
            if not all_crops:
                print("[ERROR] No banner crops available for processing")
                return results
            
            # Step 5: Process Each Banner with Enhanced Hindi Support
            print(f"\n[TEXT] Step 5: Processing {len(all_crops)} Banner(s)")
            
            # Find nearby banners for matching
            nearby_banners = None
            if lat is not None and lon is not None:
                nearby_banners = self.csv_matcher.find_nearby_banners(lat, lon)
                print(f"[TARGET] Found {len(nearby_banners)} nearby registered banners")
            else:
                print("[WARNING]  Using all registered banners (no GPS)")
                nearby_banners = self.csv_matcher.data
            
            banner_results = []
            for i, crop_info in enumerate(all_crops):
                print(f"\n--- Processing Banner {i+1}/{len(all_crops)} ---")
                
                try:
                    # Load crop image
                    crop_image = cv2.imread(crop_info['crop_path'])
                    if crop_image is None:
                        print(f"[ERROR] Failed to load crop: {crop_info['crop_path']}")
                        continue
                    
                    # Enhanced OCR with Hindi support
                    print("🔤 Extracting text with Hindi support...")
                    banner_text = self.ocr_processor.extract_banner_text(crop_image)
                    
                    # Enhanced language analysis
                    if banner_text.get('original_text'):
                        text_analysis = preprocess_text_for_matching(banner_text['original_text'])
                        banner_text['language_analysis'] = text_analysis
                        banner_text['content_category'] = text_analysis['language_info']['category']
                        
                        # Display language information
                        lang_info = text_analysis['language_info']
                        print(f"[LANG] Language Category: {banner_text['content_category']}")
                        print(f"   Hindi: {lang_info['hindi_percentage']:.1f}%, English: {lang_info['english_percentage']:.1f}%")
                    
                    # Display extracted text
                    original_text = banner_text.get('original_text', '')
                    processed_text = banner_text.get('translated_text', '')
                    
                    if original_text:
                        print(f"[TEXT] Original Text: '{original_text[:60]}{'...' if len(original_text) > 60 else ''}'")
                        if processed_text != original_text:
                            print(f"[RELOAD] Processed Text: '{processed_text[:60]}{'...' if len(processed_text) > 60 else ''}'")
                    else:
                        print("[WARNING]  No text extracted from banner")
                        continue
                    
                    # Enhanced matching with Hindi support
                    print("[SEARCH] Matching with database...")
                    match_result = self.csv_matcher.match_banner_content(banner_text, nearby_banners)
                    
                    # Create banner result
                    banner_result = {
                        'banner_id': generate_banner_id(image_path, i+1),
                        'crop_path': crop_info['crop_path'],
                        'box': crop_info.get('box', crop_info.get('original_box', [])),
                        'confidence': crop_info.get('confidence', 0.0),
                        'ocr_result': banner_text,
                        'match_result': match_result,
                        'crop_source': 'auto' if 'auto_crop' in crop_info['crop_path'] else 'manual'
                    }
                    
                    # Display results
                    if match_result.get('matched', False):
                        banner_data = match_result['banner_data']
                        company_name = banner_data.get('company_name', 'Unknown')
                        print(f"[OK] MATCH FOUND: {company_name}")
                        print(f"   Score: {match_result.get('match_score', 0):.1f}%")
                        print(f"   Method: {match_result.get('method_used', 'unknown')}")
                        print(f"   Status: {match_result.get('status_message', 'Unknown')}")
                        
                        if match_result.get('is_expired', False):
                            print(f"[WARNING]  EXPIRED: {match_result.get('days_diff', 0)} days ago")
                        else:
                            print(f"[OK] VALID: Expires in {match_result.get('days_diff', 0)} days")
                        
                        results['total_matches'] += 1
                    else:
                        print(f"[ERROR] No match found")
                        print(f"   Best score: {match_result.get('best_score', 0):.1f}%")
                        print(f"   Threshold: {match_result.get('threshold_used', 0):.1f}%")
                    
                    banner_results.append(banner_result)
                    results['banners_processed'] += 1
                    
                except Exception as e:
                    print(f"[ERROR] Error processing banner {i+1}: {e}")
                    continue
            
            results['banners'] = banner_results
            results['success'] = True
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            results['error'] = str(e)
        
        # Final timing and summary
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        
        print(f"\n{'='*60}")
        print(f"[STATS] Processing Summary")
        print(f"{'='*60}")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"[TARGET] Banners Processed: {results['banners_processed']}")
        print(f"[OK] Matches Found: {results['total_matches']}")
        print(f"[LOCATION] GPS Coordinates: {'Found' if results['gps_coordinates'] else 'Not Found'}")
        
        return results
    
    def process_batch(self, image_paths: List[str], args: argparse.Namespace) -> Dict[str, Any]:
        """
        Process multiple images in batch
        """
        print(f"\n[RELOAD] Batch Processing: {len(image_paths)} images")
        
        batch_results = {
            'total_images': len(image_paths),
            'successful_images': 0,
            'total_banners': 0,
            'total_matches': 0,
            'total_gps_found': 0,
            'total_processing_time': 0,
            'results': []
        }
        
        for i, image_path in enumerate(image_paths):
            print(f"\n📸 Image {i+1}/{len(image_paths)}")
            
            result = self.process_single_image(image_path, args)
            batch_results['results'].append(result)
            
            if result['success']:
                batch_results['successful_images'] += 1
            
            batch_results['total_banners'] += result['banners_processed']
            batch_results['total_matches'] += result['total_matches']
            batch_results['total_processing_time'] += result['processing_time']
            
            if result['gps_coordinates']:
                batch_results['total_gps_found'] += 1
            
            # Memory cleanup for batch processing
            if (i + 1) % config.MEMORY_CONFIG['clear_cache_frequency'] == 0:
                self.banner_detector.clear_cache()
                self.ocr_processor.clear_caches()
                self.csv_matcher.clear_cache()
                print("[CLEAN] Memory cache cleared")
        
        # Batch summary
        print(f"\n{'='*80}")
        print(f"[STATS] BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"📸 Total Images: {batch_results['total_images']}")
        print(f"[OK] Successful: {batch_results['successful_images']}")
        print(f"[TARGET] Total Banners: {batch_results['total_banners']}")
        print(f"[SEARCH] Total Matches: {batch_results['total_matches']}")
        print(f"[LOCATION] GPS Found: {batch_results['total_gps_found']}")
        print(f"⏱️  Total Time: {batch_results['total_processing_time']:.2f} seconds")
        print(f"📈 Success Rate: {(batch_results['successful_images'] / batch_results['total_images'] * 100):.1f}%")
        print(f"[TARGET] Match Rate: {(batch_results['total_matches'] / max(batch_results['total_banners'], 1) * 100):.1f}%")
        
        return batch_results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'gps_stats': self.ocr_processor.get_processing_stats(),
            'detection_stats': getattr(self.banner_detector, 'get_stats', lambda: {})(),
            'matching_stats': self.csv_matcher.get_matching_statistics(),
            'system_config': {
                'hindi_enabled': config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr'],
                'gpu_enabled': config.CPU_OPTIMIZATION_CONFIG['enable_gpu'],
                'cache_enabled': config.CACHE_CONFIG['enable_pattern_learning'],
                'max_processing_time': config.PERFORMANCE_OPTIMIZATIONS['max_processing_time_per_banner']
            }
        }

def main():
    """Enhanced main function with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(
        description='Enhanced Ad Banner Expiry Detection System with GPS & Hindi Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.jpg                    # Process single image
  %(prog)s *.jpg                        # Process multiple images
  %(prog)s image.jpg --manual-crop      # Include manual cropping
  %(prog)s image.jpg --debug-gps        # Debug GPS extraction
  %(prog)s image.jpg --skip-no-gps      # Skip images without GPS
  %(prog)s --stats                      # Show system statistics
        """
    )
    
    # Input arguments
    parser.add_argument('images', nargs='*', help='Image file(s) to process')
    parser.add_argument('--directory', '-d', help='Process all images in directory')
    
    # Processing options
    parser.add_argument('--manual-crop', '-m', action='store_true',
                       help='Enable manual cropping interface')
    parser.add_argument('--skip-no-gps', action='store_true',
                       help='Skip images without GPS coordinates')
    
    # Debug options
    parser.add_argument('--debug-gps', action='store_true',
                       help='Enable GPS extraction debugging')
    parser.add_argument('--debug-ocr', action='store_true',
                       help='Enable OCR debugging')
    parser.add_argument('--debug-matching', action='store_true',
                       help='Enable matching debugging')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output directory for results')
    parser.add_argument('--save-crops', action='store_true',
                       help='Save cropped banner images')
    parser.add_argument('--save-detections', action='store_true',
                       help='Save detection result images')
    
    # System options
    parser.add_argument('--stats', action='store_true',
                       help='Show system statistics and exit')
    parser.add_argument('--clear-cache', action='store_true',
                       help='Clear all caches before processing')
    parser.add_argument('--version', action='version', version='Enhanced Banner Detection System v2.0')
    
    args = parser.parse_args()
    
    # Initialize processor
    try:
        processor = EnhancedBannerProcessor()
    except Exception as e:
        print(f"[ERROR] Failed to initialize system: {e}")
        return 1
    
    # Handle stats request
    if args.stats:
        print("[STATS] System Statistics:")
        stats = processor.get_system_stats()
        
        print(f"\n[GPS]  GPS Extraction:")
        gps_stats = stats['gps_stats']
        print(f"   Success Rate: {gps_stats.get('success_rate', 0):.1f}%")
        print(f"   Avg Time: {gps_stats.get('avg_processing_time', 0):.2f}s")
        
        print(f"\n[SEARCH] Text Extraction:")
        print(f"   Hindi Extractions: {gps_stats.get('hindi_extractions', 0)}")
        print(f"   English Extractions: {gps_stats.get('english_extractions', 0)}")
        print(f"   Mixed Extractions: {gps_stats.get('mixed_extractions', 0)}")
        
        print(f"\n⚙️  System Config:")
        system_config = stats['system_config']
        print(f"   Hindi Support: {'[OK]' if system_config['hindi_enabled'] else '[ERROR]'}")
        print(f"   GPU Processing: {'[OK]' if system_config['gpu_enabled'] else '[ERROR]'}")
        print(f"   Caching: {'[OK]' if system_config['cache_enabled'] else '[ERROR]'}")
        
        return 0
    
    # Clear cache if requested
    if args.clear_cache:
        processor.banner_detector.clear_cache()
        processor.ocr_processor.clear_caches()
        processor.csv_matcher.clear_cache()
        print("[CLEAN] All caches cleared")
    
    # Determine images to process
    image_paths = []
    
    if args.directory:
        # Process directory
        if not os.path.isdir(args.directory):
            print(f"[ERROR] Directory not found: {args.directory}")
            return 1
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            image_paths.extend(glob.glob(os.path.join(args.directory, ext)))
            image_paths.extend(glob.glob(os.path.join(args.directory, ext.upper())))
    
    elif args.images:
        # Process specified images
        for pattern in args.images:
            if '*' in pattern or '?' in pattern:
                import glob
                image_paths.extend(glob.glob(pattern))
            else:
                if os.path.isfile(pattern):
                    image_paths.append(pattern)
                else:
                    print(f"[WARNING]  File not found: {pattern}")
    
    else:
        print("[ERROR] No images specified. Use --help for usage information.")
        return 1
    
    if not image_paths:
        print("[ERROR] No valid image files found")
        return 1
    
    # Process images
    if len(image_paths) == 1:
        # Single image processing
        result = processor.process_single_image(image_paths[0], args)
        return 0 if result['success'] else 1
    else:
        # Batch processing
        batch_result = processor.process_batch(image_paths, args)
        return 0 if batch_result['successful_images'] > 0 else 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[WARNING]  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)
