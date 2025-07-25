from email.mime import image
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import torch
from PIL import Image
import config
from utils import resize_image_for_display, draw_detection_boxes

class BannerDetector:
    """
    Handles banner detection using YOLOv8 model and provides
    functionality for automatic and manual cropping.
    """
    
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH):
        """
        Initialize the banner detector with YOLO model.
        
        Args:
            model_path: Path to trained YOLO model
        """
        self.model_path = model_path  # Store model path
        self.model = None  # Initialize model as None
        self.model_loaded = False  # Flag to track if model is loaded
        
        # Load YOLO model
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load YOLOv8 model with robust error handling.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Check if model file exists
            if not os.path.exists(self.model_path):
                print(f"Error: YOLO model file not found at {self.model_path}")
                return False
            
            # Check file size to make sure it's a valid model file
            try:
                file_size = os.path.getsize(self.model_path)
                if file_size < 10000:  # Model files should be at least 10KB
                    print(f"Error: YOLO model file at {self.model_path} appears to be invalid (too small: {file_size} bytes)")
                    return False
                print(f"Model file size: {file_size / (1024*1024):.2f} MB")
            except Exception as e:
                print(f"Warning: Could not check model file size: {e}")
                
            # APPROACH 1: Try loading directly via ultralytics
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                print(f"YOLOv8 model loaded successfully via ultralytics from {self.model_path}")
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"Warning: Could not load with ultralytics YOLO: {e}")
            
            # APPROACH 2: Try with torch.load and weights_only=False
            try:
                import torch
                # For PyTorch 2.6+ compatibility
                if hasattr(torch.serialization, 'add_safe_globals'):
                    try:
                        torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                        print("Added safe globals for PyTorch 2.6+")
                    except Exception as e:
                        print(f"Warning: Could not add safe globals: {e}")
                
                # Try loading with weights_only=False
                try:
                    self.model = torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=False)
                    print(f"YOLOv8 model loaded successfully with weights_only=False from {self.model_path}")
                    self.model_loaded = True
                    return True
                except Exception as e:
                    print(f"Warning: Could not load with weights_only=False: {e}")
            except Exception as e:
                print(f"Warning: Error in torch loading attempt: {e}")
            
            # APPROACH 3: Try PyTorch Hub as last resort
            try:
                import torch
                os.environ['TORCH_HOME'] = os.path.dirname(self.model_path)
                os.environ['TORCH_OFFLINE'] = '1'  # Try to avoid internet requests
                
                self.model = torch.hub.load('ultralytics/yolov8', 'custom', path=self.model_path, trust_repo=True)
                self.model.conf = config.YOLO_CONFIDENCE_THRESHOLD
                self.model.iou = config.YOLO_IOU_THRESHOLD
                
                print(f"YOLOv8 model loaded successfully via PyTorch Hub from {self.model_path}")
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"Failed to load YOLO model via PyTorch Hub: {e}")
            
            # If all approaches failed
            self.model = None
            self.model_loaded = False
            return False
            
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None
            self.model_loaded = False
            return False
            
    def detect_banners(self, image_path: str) -> Dict[str, Any]:
        """
        Detect banners in an image using the YOLOv8 model.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with detection results
        """
        # Load image first to avoid unnecessary model loading if image is invalid
        try:
            image = cv2.imread(image_path)  # Read image
            if image is None:
                return {
                    'success': False,  # Detection failed
                    'error': f"Could not load image: {image_path}",  # Error message
                    'boxes': [],  # Empty boxes list
                    'image': None,  # No image
                    'count': 0  # Zero detections
                }
        except Exception as e:
            return {
                'success': False,  # Detection failed
                'error': f"Error loading image: {e}",  # Error message
                'boxes': [],  # Empty boxes list
                'image': None,  # No image
                'count': 0  # Zero detections
            }
            
        # Check if YOLO model is loaded, otherwise use fallback detection
        if not self.model_loaded:
            print("Using fallback detection method because YOLO model is not loaded")
            return self._fallback_detection(image)
        
        # Use YOLO model for detection
        try:
            results = self.model(image)  # Run inference
            boxes = []
            
            # Process results - handle both YOLO formats correctly
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                # New YOLOv8 format (current ultralytics)
                result = results[0]  # Get first result
                
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Coordinates
                        confidence = float(box.conf[0].cpu().numpy())  # Confidence
                        class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0  # Class ID
                        
                        # Apply confidence threshold
                        if confidence >= config.YOLO_CONFIDENCE_THRESHOLD:
                            boxes.append({
                                'box': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id
                            })
                            
            elif hasattr(results, 'xyxy') and len(results.xyxy[0]) > 0:
                # Legacy format (older YOLO versions)
                for box in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, confidence, class_id = box
                    
                    # Apply confidence threshold
                    if confidence >= config.YOLO_CONFIDENCE_THRESHOLD:
                        boxes.append({
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        })
            # Create annotated image for visualization  
            box_coords = [box['box'] for box in boxes]  # Extract just coordinates
            annotated_img = draw_detection_boxes(image.copy(), box_coords)
            
            return {
                'success': True,  # Detection successful
                'boxes': boxes,  # Detected boxes
                'image': annotated_img,  # Annotated image
                'count': len(boxes),  # Number of detections
                'method': 'yolo'  # Detection method used
            }
        except Exception as e:
            print(f"YOLO detection failed, falling back to basic detection: {e}")
            return self._fallback_detection(image)
            
    def _fallback_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Simple fallback banner detection when YOLO model is not available.
        Uses basic image processing techniques to find potential banner areas.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with detection results
        """
        try:
            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Dilate to connect nearby edges
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on area and aspect ratio
            min_area = image.shape[0] * image.shape[1] * 0.01  # At least 1% of image
            boxes = []
            
            for contour in contours:
                try:
                    # Get bounding rect
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Skip if too small or aspect ratio is too extreme
                    if area < min_area or aspect_ratio > 10 or aspect_ratio < 0.1:
                        continue
                    
                    # Add to boxes
                    boxes.append({
                        'box': [x, y, x+w, y+h],
                        'confidence': 0.5,  # Default confidence for fallback detection
                        'class_id': 0
                    })
                except Exception as e:
                    print(f"Error processing contour: {e}")
                    continue
            
            # Merge overlapping boxes
            if boxes:
                boxes = self._merge_boxes(boxes)
            
            # Create annotated image
            box_coords = [box['box'] for box in boxes]  # Extract just coordinates  
            annotated_img = draw_detection_boxes(image.copy(), box_coords)
            
            return {
                'success': True,  # Detection successful
                'boxes': boxes,  # Detected boxes
                'image': annotated_img,  # Annotated image
                'count': len(boxes),  # Number of detections
                'method': 'fallback'  # Detection method used
            }
        except Exception as e:
            print(f"Error in fallback detection: {e}")
            return {
                'success': False,  # Detection failed
                'error': f"Fallback detection failed: {e}",  # Error message
                'boxes': [],  # Empty boxes
                'image': None,  # No image
                'count': 0  # Zero detections
            }
    
    def _merge_boxes(self, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping bounding boxes.
        
        Args:
            boxes: List of bounding box dictionaries
            
        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []
            
        # Extract coordinates
        rects = [box['box'] for box in boxes]
        confidences = [box['confidence'] for box in boxes]
        
        # Prepare for NMS
        rects_np = np.array(rects)
        confidences_np = np.array(confidences)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(rects, confidences, 0.3, 0.4)
        
        # Create result boxes
        result = []
        for i in indices:
            if isinstance(i, list) or isinstance(i, np.ndarray):
                i = i[0]  # Handle older OpenCV versions
            result.append(boxes[i])
            
        return result
    
    def crop_banner(self, image_path: str, detection_result: Dict[str, Any], 
                   crop_idx: int = 0, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Crop a detected banner from an image.
        
        Args:
            image_path: Path to original image
            detection_result: Result from detect_banners
            crop_idx: Index of the banner to crop
            save_path: Path to save the cropped image (optional)
            
        Returns:
            Dictionary with cropping results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': f"Could not load image: {image_path}"}
                
            # Check detection result
            if not detection_result.get('success', False):
                return {'success': False, 'error': "No successful detection to crop from"}
                
            boxes = detection_result.get('boxes', [])
            if not boxes or crop_idx >= len(boxes):
                return {'success': False, 'error': f"No box at index {crop_idx}"}
                
            # Get crop coordinates
            box = boxes[crop_idx]['box']
            x1, y1, x2, y2 = box
            
            # Crop image
            cropped = image[y1:y2, x1:x2]
            
            # Save if needed
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cropped)
            
            return {
                'success': True,
                'crop': cropped,
                'box': box,
                'save_path': save_path if save_path else None
            }
        except Exception as e:
            return {'success': False, 'error': f"Error cropping banner: {e}"}
    
    def manual_crop(self, image_path: str, coords: List[int], 
                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Manually crop a region from an image.
        
        Args:
            image_path: Path to original image
            coords: Coordinates [x1, y1, x2, y2] for cropping
            save_path: Path to save the cropped image (optional)
            
        Returns:
            Dictionary with cropping results
        """
        try:
            # Validate coordinates
            if len(coords) != 4:
                return {'success': False, 'error': "Coordinates must be [x1, y1, x2, y2]"}
                
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': f"Could not load image: {image_path}"}
                
            # Get crop coordinates
            x1, y1, x2, y2 = coords
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Check if valid crop size
            if x2 <= x1 or y2 <= y1:
                return {'success': False, 'error': "Invalid crop dimensions"}
                
            # Crop image
            cropped = image[y1:y2, x1:x2]
            
            # Save if needed
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, cropped)
            
            return {
                'success': True,
                'crop': cropped,
                'box': [x1, y1, x2, y2],
                'save_path': save_path if save_path else None
            }
        except Exception as e:
            return {'success': False, 'error': f"Error with manual crop: {e}"}

    def crop_banners(self, detection_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Crop all detected banners from detection results.
        Uses the existing crop_banner method for each individual banner.
        
        Args:
            detection_results: Results from detect_banners method
            
        Returns:
            List of crop information dictionaries
        """
        try:
            if not detection_results.get('success', False):
                print("[ERROR] No successful detection results to crop")
                return []
            
            boxes = detection_results.get('boxes', [])
            if not boxes:
                print("[ERROR] No boxes in detection results")
                return []
            
            print(f"[SEARCH] DEBUG: Cropping {len(boxes)} detected banners")
            
            crops = []
            for i, box_info in enumerate(boxes):
                try:
                    # Create temporary image from detection results
                    detection_image = detection_results.get('image')
                    if detection_image is None:
                        print(f"[ERROR] No image in detection results for crop {i+1}")
                        continue
                    
                    # Extract box coordinates
                    box = box_info['box']  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure coordinates are valid and within image bounds
                    h, w = detection_image.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(x1+1, min(x2, w))
                    y2 = max(y1+1, min(y2, h))
                    
                    # Crop the image directly
                    crop_image = detection_image[y1:y2, x1:x2]
                    
                    # Skip empty crops
                    if crop_image.size == 0:
                        print(f"[WARNING] Skipping empty crop {i+1}")
                        continue
                    
                    # Create output filename
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    crop_filename = f"auto_crop_{timestamp}_{i+1}.jpg"
                    
                    # Save to auto crops directory
                    crop_path = os.path.join(config.AUTO_CROPS_DIR, crop_filename)
                    
                    # Ensure directory exists
                    os.makedirs(config.AUTO_CROPS_DIR, exist_ok=True)
                    
                    # Save crop
                    cv2.imwrite(crop_path, crop_image)
                    
                    # Create crop info
                    crop_info = {
                        'crop_path': crop_path,
                        'box': [x1, y1, x2, y2],
                        'original_box': box,
                        'confidence': box_info.get('confidence', 0.0),
                        'class_id': box_info.get('class_id', 0),
                        'crop_index': i + 1
                    }
                    
                    crops.append(crop_info)
                    print(f"[OK] Created crop {i+1}: {crop_path}")
                    
                except Exception as e:
                    print(f"[ERROR] Error processing crop {i+1}: {e}")
                    continue
            
            print(f"[OK] Successfully created {len(crops)} crops")
            return crops
            
        except Exception as e:
            print(f"[ERROR] Error in crop_banners: {e}")
            return []  
