"""
Enhanced Flask Web Application for Ad Banner Expiry Detection
Supports GPS reliability improvements and Hindi text processing
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, session
from werkzeug.utils import secure_filename
import cv2
import numpy as np

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
    sys.exit(1)

# Flask app configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'banner-detection-2024')
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Global components
banner_detector = None
ocr_processor = None
csv_matcher = None
hybrid_cropper = None

def init_components():
    """Initialize enhanced system components"""
    global banner_detector, ocr_processor, csv_matcher, hybrid_cropper
    
    try:
        print("[START] Initializing Enhanced Banner Detection System...")
        
        banner_detector = BannerDetector()
        ocr_processor = OCRProcessor()
        csv_matcher = CSVMatcher()
        hybrid_cropper = HybridCropper()
        
        # Create required directories
        create_directories([
            config.UPLOAD_FOLDER,
            os.path.join(config.UPLOAD_FOLDER, "crops"),
            config.RESULTS_DIR,
            config.AUTO_CROPS_DIR,
            config.MANUAL_CROPS_DIR,
            config.DETECTIONS_DIR,
            config.LOGS_DIR
        ])
        
        print("[OK] All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] Component initialization failed: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def process_image_enhanced(image_path: str, filename: str, manual_crops: Optional[List] = None) -> Dict[str, Any]:
    """
    Enhanced image processing with GPS reliability and Hindi support
    """
    print(f"\n[SEARCH] Processing: {filename}")
    start_time = time.time()
    
    result = {
        'filename': filename,
        'success': False,
        'gps_coordinates': None,
        'banners': [],
        'processing_time': 0,
        'error': None,
        'language_stats': {
            'hindi_banners': 0,
            'english_banners': 0,
            'mixed_banners': 0
        }
    }
    
    try:
        # Step 1: Enhanced GPS Extraction
        print("[LOCATION] Extracting GPS coordinates...")
        lat, lon, gps_text = ocr_processor.extract_gps_from_image(image_path)
        
        if lat is not None and lon is not None:
            result['gps_coordinates'] = {
                'latitude': lat,
                'longitude': lon,
                'raw_text': gps_text
            }
            print(f"[OK] GPS Found: {lat:.6f}, {lon:.6f}")
        else:
            print("[WARNING]  GPS coordinates not found")
        
        # Step 2: Banner Detection
        print("[TARGET] Detecting banners...")
        detection_result = banner_detector.detect_banners(image_path)
        
        if not detection_result.get('success', False):
            print("[ERROR] No banners detected")
            result['error'] = "No banners detected in the image"
            return result
        
        detected_boxes = detection_result.get('boxes', [])
        print(f"[OK] Detected {len(detected_boxes)} banner(s)")
        
        # Step 3: Create crops
        print("✂️  Creating banner crops...")
        auto_crops = banner_detector.crop_banners(detection_result)
        
        # Combine with manual crops if provided
        all_crops = auto_crops + (manual_crops or [])
        
        if not all_crops:
            result['error'] = "No banner crops could be created"
            return result
        
        # Step 4: Find nearby banners for matching
        nearby_banners = None
        if lat is not None and lon is not None:
            nearby_banners = csv_matcher.find_nearby_banners(lat, lon)
            print(f"[TARGET] Found {len(nearby_banners)} nearby registered banners")
        else:
            print("[WARNING]  Using all registered banners (no GPS)")
            nearby_banners = csv_matcher.data
        
        # Step 5: Process each banner with enhanced Hindi support
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
                banner_text = ocr_processor.extract_banner_text(crop_image)
                
                # Enhanced language analysis
                if banner_text.get('original_text'):
                    text_analysis = preprocess_text_for_matching(banner_text['original_text'])
                    banner_text['language_analysis'] = text_analysis
                    banner_text['content_category'] = text_analysis['language_info']['category']
                    
                    # Update language statistics
                    category = banner_text['content_category']
                    if category in ['pure_hindi', 'hindi_dominant']:
                        result['language_stats']['hindi_banners'] += 1
                    elif category in ['pure_english', 'english_dominant']:
                        result['language_stats']['english_banners'] += 1
                    else:
                        result['language_stats']['mixed_banners'] += 1
                    
                    # Display language information
                    lang_info = text_analysis['language_info']
                    print(f"[LANG] Language Category: {category}")
                    print(f"   Hindi: {lang_info['hindi_percentage']:.1f}%, English: {lang_info['english_percentage']:.1f}%")
                else:
                    banner_text['content_category'] = 'unknown'
                    banner_text['language_analysis'] = {}
                
                # Display extracted text
                original_text = banner_text.get('original_text', '')
                processed_text = banner_text.get('translated_text', '')
                
                if original_text:
                    print(f"[TEXT] Original: '{original_text[:50]}{'...' if len(original_text) > 50 else ''}'")
                    if processed_text != original_text:
                        print(f"[RELOAD] Processed: '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")
                else:
                    print("[WARNING]  No text extracted")
                    continue
                
                # Enhanced matching with Hindi support
                print("[SEARCH] Matching with database...")
                match_result = csv_matcher.match_banner_content(banner_text, nearby_banners)
                
                # Prepare crop image for web display
                crop_display_path = None
                try:
                    crop_filename = f"crop_{i+1}_{int(time.time())}.jpg"
                    crop_display_path = os.path.join(config.UPLOAD_FOLDER, "crops", crop_filename)
                    os.makedirs(os.path.dirname(crop_display_path), exist_ok=True)
                    
                    # Resize for web display
                    display_crop = resize_image_for_display(crop_image, max_width=400, max_height=300)
                    cv2.imwrite(crop_display_path, display_crop)
                    crop_display_path = f"uploads/crops/{crop_filename}"
                except Exception as e:
                    print(f"[WARNING]  Could not save crop for display: {e}")
                
                # Create banner result
                banner_result = {
                    'banner_id': generate_banner_id(filename, i+1),
                    'crop_path': crop_display_path,
                    'crop_source': 'auto' if 'auto_crop' in crop_info['crop_path'] else 'manual',
                    'box': crop_info.get('box', crop_info.get('original_box', [])),
                    'confidence': crop_info.get('confidence', 0.0),
                    'ocr_result': banner_text,
                    'match_result': match_result
                }
                
                # Display match results
                if match_result.get('matched', False):
                    banner_data = match_result['banner_data']
                    company_name = banner_data.get('company_name', 'Unknown')
                    print(f"[OK] MATCH: {company_name} ({match_result.get('match_score', 0):.1f}%)")
                    print(f"   Method: {match_result.get('method_used', 'unknown')}")
                    print(f"   Status: {match_result.get('status_message', 'Unknown')}")
                else:
                    print(f"[ERROR] No match found (best: {match_result.get('best_score', 0):.1f}%)")
                
                banner_results.append(banner_result)
                
            except Exception as e:
                print(f"[ERROR] Error processing banner {i+1}: {e}")
                continue
        
        result['banners'] = banner_results
        result['success'] = True
        
        # Summary
        total_matches = sum(1 for b in banner_results if b['match_result'].get('matched', False))
        print(f"\n[STATS] Summary: {len(banner_results)} banners processed, {total_matches} matches found")
        
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        result['error'] = str(e)
        traceback.print_exc()
    
    result['processing_time'] = time.time() - start_time
    return result

@app.route('/')
def index():
    """Enhanced home page with language support info"""
    return render_template('index.html', 
                         hindi_support=config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr'],
                         gps_optimization=config.GPS_EXTRACTION_CONFIG['enable_region_caching'])

@app.route('/upload', methods=['POST'])
def upload_file():
    """Enhanced file upload with better error handling"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if manual cropping is requested
            enable_manual_crop = request.form.get('manual_crop') == 'on'
            
            if enable_manual_crop:
                return redirect(url_for('manual_crop', filename=filename))
            else:
                return redirect(url_for('process_image', filename=filename))
        else:
            flash('Invalid file format. Please upload JPG, JPEG, or PNG files.', 'error')
            return redirect(request.url)
            
    except Exception as e:
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(request.url)

@app.route('/manual_crop/<filename>')
def manual_crop(filename):
    """Manual cropping interface"""
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            flash('Image file not found', 'error')
            return redirect(url_for('index'))
        
        return render_template('manual_crop.html', filename=filename)
        
    except Exception as e:
        flash(f'Error loading manual crop interface: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/manual_crop', methods=['POST'])
def api_manual_crop():
    """API endpoint for manual cropping - ONLY manual crops"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        crops = data.get('crops', [])
        manual_only = data.get('manual_only', True)  # Flag for manual-only processing
        
        if not filename or not crops:
            return jsonify({'success': False, 'error': 'Invalid data'})
        
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image file not found'})
        
        # Process ONLY manual crops (ignore auto-detection)
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not load image'})
        
        manual_crops = []
        for i, crop_data in enumerate(crops):
            try:
                x1, y1, x2, y2 = int(crop_data['x1']), int(crop_data['y1']), int(crop_data['x2']), int(crop_data['y2'])
                
                # Validate coordinates
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))
                
                # Create crop
                crop_image = image[y1:y2, x1:x2]
                
                # Save crop
                crop_filename = f"manual_crop_{int(time.time())}_{i+1}.jpg"
                crop_path = os.path.join(config.MANUAL_CROPS_DIR, crop_filename)
                cv2.imwrite(crop_path, crop_image)
                
                manual_crops.append({
                    'crop_path': crop_path,
                    'box': [x1, y1, x2, y2],
                    'confidence': 1.0,  # Manual crops have full confidence
                    'source': 'manual'  # Mark as manual crop
                })
                
            except Exception as e:
                print(f"Error creating manual crop {i+1}: {e}")
                continue
        
        # Store manual crops info in session or temporary storage
        # This will be used by process_image to know it should use ONLY manual crops
        session[f'manual_crops_{filename}'] = manual_crops
        
        return jsonify({
            'success': True,
            'crops_created': len(manual_crops),
            'message': f'Created {len(manual_crops)} manual crop regions',
            'redirect_url': url_for('process_image', filename=filename, manual_crops='true')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def process_image_enhanced(image_path: str, filename: str, manual_crops_only: bool = False) -> Dict[str, Any]:
    """
    Enhanced image processing with selective crop processing.
    If manual_crops_only=True, ONLY process manual crops, ignore auto-detection.
    """
    print(f"\n🔍 Processing: {filename}")
    print(f"📋 Manual crops only: {'YES' if manual_crops_only else 'NO'}")
    
    start_time = time.time()
    
    result = {
        'filename': filename,
        'success': False,
        'gps_coordinates': None,
        'banners': [],
        'processing_time': 0,
        'error': None,
        'processing_mode': 'manual_only' if manual_crops_only else 'auto_detection',
        'language_stats': {
            'hindi_banners': 0,
            'english_banners': 0,
            'mixed_banners': 0
        }
    }
    
    try:
        # Step 1: Enhanced GPS Extraction (always needed)
        print("📍 Extracting GPS coordinates...")
        lat, lon, gps_text = ocr_processor.extract_gps_from_image(image_path)
        
        if lat is not None and lon is not None:
            result['gps_coordinates'] = {
                'latitude': lat,
                'longitude': lon,
                'raw_text': gps_text
            }
            print(f"✅ GPS Found: {lat:.6f}, {lon:.6f}")
        else:
            print("⚠️  GPS coordinates not found")
        
        if manual_crops_only:
            # MANUAL CROPS ONLY - Do NOT run auto-detection
            print("✂️  Processing MANUAL CROPS ONLY (auto-detection disabled)")
            
            # Get manual crops from session
            manual_crops = session.get(f'manual_crops_{filename}', [])
            
            if not manual_crops:
                result['error'] = "No manual crops found"
                return result
            
            all_crops = manual_crops
            print(f"✅ Using {len(manual_crops)} manual crops only")
            
        else:
            # AUTO-DETECTION MODE - Run banner detection
            print("🎯 Running AUTO-DETECTION (manual crops disabled)")
            detection_result = banner_detector.detect_banners(image_path)
            
            if not detection_result.get('success', False):
                print("❌ No banners detected")
                result['error'] = "No banners detected in the image"
                return result
            
            detected_boxes = detection_result.get('boxes', [])
            print(f"✅ Auto-detected {len(detected_boxes)} banner(s)")
            
            # Create auto crops
            auto_crops = banner_detector.crop_banners(detection_result)
            all_crops = auto_crops
        
        # Find nearby banners for matching
        nearby_banners = None
        if lat is not None and lon is not None:
            nearby_banners = csv_matcher.find_nearby_banners(lat, lon)
            print(f"🎯 Found {len(nearby_banners)} nearby registered banners")
        else:
            print("⚠️  Using all registered banners (no GPS)")
            nearby_banners = csv_matcher.data
        
        # Process each crop with enhanced Hindi support
        banner_results = []
        
        for i, crop_info in enumerate(all_crops):
            print(f"\n--- Processing Banner {i+1}/{len(all_crops)} ---")
            
            try:
                # Load crop image
                crop_image = cv2.imread(crop_info['crop_path'])
                if crop_image is None:
                    print(f"❌ Failed to load crop: {crop_info['crop_path']}")
                    continue
                
                # Enhanced OCR with Hindi support
                print("🔤 Extracting text with Hindi support...")
                banner_text = ocr_processor.extract_banner_text(crop_image)
                
                # Enhanced language analysis
                if banner_text.get('original_text'):
                    text_analysis = preprocess_text_for_matching(banner_text['original_text'])
                    banner_text['language_analysis'] = text_analysis
                    banner_text['content_category'] = text_analysis['language_info']['category']
                    
                    # Update language statistics
                    category = banner_text['content_category']
                    if category in ['pure_hindi', 'hindi_dominant']:
                        result['language_stats']['hindi_banners'] += 1
                    elif category in ['pure_english', 'english_dominant']:
                        result['language_stats']['english_banners'] += 1
                    else:
                        result['language_stats']['mixed_banners'] += 1
                    
                    # Display language information
                    lang_info = text_analysis['language_info']
                    print(f"🌐 Language Category: {category}")
                    print(f"   Hindi: {lang_info['hindi_percentage']:.1f}%, English: {lang_info['english_percentage']:.1f}%")
                else:
                    banner_text['content_category'] = 'unknown'
                    banner_text['language_analysis'] = {}
                
                # Display extracted text
                original_text = banner_text.get('original_text', '')
                processed_text = banner_text.get('translated_text', '')
                
                if original_text:
                    print(f"📝 Original: '{original_text[:50]}{'...' if len(original_text) > 50 else ''}'")
                    if processed_text != original_text:
                        print(f"🔄 Processed: '{processed_text[:50]}{'...' if len(processed_text) > 50 else ''}'")
                else:
                    print("⚠️  No text extracted")
                    continue
                
                # Enhanced matching with Hindi support
                print("🔍 Matching with database...")
                match_result = csv_matcher.match_banner_content(banner_text, nearby_banners)
                
                # Prepare crop image for web display
                crop_display_path = None
                try:
                    crop_filename = f"crop_{i+1}_{int(time.time())}.jpg"
                    crop_display_path = os.path.join(config.UPLOAD_FOLDER, "crops", crop_filename)
                    os.makedirs(os.path.dirname(crop_display_path), exist_ok=True)
                    
                    # Resize for web display
                    display_crop = resize_image_for_display(crop_image, max_width=400, max_height=300)
                    cv2.imwrite(crop_display_path, display_crop)
                    crop_display_path = f"uploads/crops/{crop_filename}"
                except Exception as e:
                    print(f"⚠️  Could not save crop for display: {e}")
                
                # Create banner result
                banner_result = {
                    'banner_id': generate_banner_id(filename, i+1),
                    'crop_path': crop_display_path,
                    'crop_source': crop_info.get('source', 'auto' if not manual_crops_only else 'manual'),
                    'box': crop_info.get('box', crop_info.get('original_box', [])),
                    'confidence': crop_info.get('confidence', 0.0),
                    'ocr_result': banner_text,
                    'match_result': match_result
                }
                
                # Display match results
                if match_result.get('matched', False):
                    banner_data = match_result['banner_data']
                    company_name = banner_data.get('company_name', 'Unknown')
                    print(f"✅ MATCH: {company_name} ({match_result.get('match_score', 0):.1f}%)")
                    print(f"   Method: {match_result.get('method_used', 'unknown')}")
                    print(f"   Status: {match_result.get('status_message', 'Unknown')}")
                else:
                    print(f"❌ No match found (best: {match_result.get('best_score', 0):.1f}%)")
                
                banner_results.append(banner_result)
                
            except Exception as e:
                print(f"❌ Error processing banner {i+1}: {e}")
                continue
        
        result['banners'] = banner_results
        result['success'] = True
        
        # Clean up session data for manual crops
        if manual_crops_only and f'manual_crops_{filename}' in session:
            del session[f'manual_crops_{filename}']
        
        # Summary
        total_matches = sum(1 for b in banner_results if b['match_result'].get('matched', False))
        mode = "MANUAL CROPS" if manual_crops_only else "AUTO-DETECTED BANNERS"
        print(f"\n📊 Summary: {len(banner_results)} {mode} processed, {total_matches} matches found")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        result['error'] = str(e)
        traceback.print_exc()
    
    result['processing_time'] = time.time() - start_time
    return result

@app.route('/process/<filename>')
def process_image(filename):
    """Enhanced image processing endpoint with selective processing"""
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(image_path):
            flash('Image file not found', 'error')
            return redirect(url_for('index'))
        
        # Check if this is manual crops processing
        manual_crops_mode = request.args.get('manual_crops') == 'true'
        
        # Process image with enhanced system
        result = process_image_enhanced(image_path, filename, manual_crops_only=manual_crops_mode)
        
        if result['success']:
            return render_template('results.html', 
                                 result=result,
                                 filename=filename,
                                 image_url=f'uploads/{filename}')
        else:
            flash(f'Processing failed: {result.get("error", "Unknown error")}', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Processing error: {str(e)}', 'error')
        traceback.print_exc()
        return redirect(url_for('index'))

@app.route('/api/system_stats')
def api_system_stats():
    """API endpoint for system statistics"""
    try:
        stats = {
            'gps_stats': ocr_processor.get_processing_stats() if ocr_processor else {},
            'matching_stats': csv_matcher.get_matching_statistics() if csv_matcher else {},
            'system_config': {
                'hindi_enabled': config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr'],
                'gpu_enabled': config.CPU_OPTIMIZATION_CONFIG['enable_gpu'],
                'cache_enabled': config.CACHE_CONFIG['enable_pattern_learning'],
                'max_processing_time': config.PERFORMANCE_OPTIMIZATIONS['max_processing_time_per_banner']
            }
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/clear_cache', methods=['POST'])
def api_clear_cache():
    """API endpoint to clear system caches"""
    try:
        if banner_detector:
            banner_detector.clear_cache()
        if ocr_processor:
            ocr_processor.clear_caches()
        if csv_matcher:
            csv_matcher.clear_cache()
        
        return jsonify({'success': True, 'message': 'All caches cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 8MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_error(error):
    flash('An internal error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

# Initialize components when app starts
with app.app_context():
    if not init_components():
        print("[ERROR] Failed to initialize components. Some features may not work.")

if __name__ == '__main__':
    print("[LANG] Starting Enhanced Banner Detection Web Application...")
    print(f"📁 Upload folder: {config.UPLOAD_FOLDER}")
    print(f"[HINDI] Hindi support: {'[OK]' if config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr'] else '[ERROR]'}")
    print(f"[LOCATION] GPS optimization: {'[OK]' if config.GPS_EXTRACTION_CONFIG['enable_region_caching'] else '[ERROR]'}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
