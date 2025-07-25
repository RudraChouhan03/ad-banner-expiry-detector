import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
import time
from datetime import datetime
import config

class HybridCropper:
    """
    Enhanced manual cropping interface with improved functionality
    """
    
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.display_image = None
        self.crop_regions = []
        self.current_crop = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.zoom_factor = 1.0
        self.pan_offset = [0, 0]
        
        # Window properties
        self.window_name = "Manual Banner Cropping - Enhanced"
        self.max_display_width = 1200
        self.max_display_height = 800
        
        print("🚀 Enhanced Manual Cropper initialized")
    
    def crop_banners_interactive(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Main interactive cropping interface with enhanced controls
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return []
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        print(f"✅ Loaded image: {image_path}")
        print(f"📐 Image dimensions: {self.original_image.shape}")
        
        # Initialize display
        self._initialize_display()
        
        # Setup mouse callback
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Main interaction loop
        self._run_interactive_session()
        
        # Create crops from selected regions
        crops = self._create_crops_from_regions(image_path)
        
        # Cleanup
        cv2.destroyAllWindows()
        
        return crops
    
    def _initialize_display(self):
        """Initialize the display window and image"""
        self.current_image = self.original_image.copy()
        
        # Calculate display scaling
        height, width = self.original_image.shape[:2]
        scale_x = self.max_display_width / width
        scale_y = self.max_display_height / height
        self.display_scale = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        # Create display image
        if self.display_scale < 1.0:
            new_width = int(width * self.display_scale)
            new_height = int(height * self.display_scale)
            self.display_image = cv2.resize(
                self.current_image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
        else:
            self.display_image = self.current_image.copy()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_image.shape[1], self.display_image.shape[0])
        
        print(f"📺 Display scale: {self.display_scale:.2f}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Enhanced mouse callback with zoom and pan support"""
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self._update_display()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                self._add_crop_region()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to remove last crop
            if self.crop_regions:
                self.crop_regions.pop()
                print(f"🗑️ Removed last crop. Total crops: {len(self.crop_regions)}")
                self._update_display()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Mouse wheel for zoom
            if flags > 0:
                self.zoom_factor = min(self.zoom_factor * 1.1, 3.0)
            else:
                self.zoom_factor = max(self.zoom_factor * 0.9, 0.5)
            self._update_display()
    
    def _add_crop_region(self):
        """Add a crop region from current selection"""
        if self.start_point and self.end_point:
            # Convert display coordinates to original image coordinates
            x1 = int(min(self.start_point[0], self.end_point[0]) / self.display_scale)
            y1 = int(min(self.start_point[1], self.end_point[1]) / self.display_scale)
            x2 = int(max(self.start_point[0], self.end_point[0]) / self.display_scale)
            y2 = int(max(self.start_point[1], self.end_point[1]) / self.display_scale)
            
            # Validate bounds
            height, width = self.original_image.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            # Check minimum size
            if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                crop_region = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'area': (x2 - x1) * (y2 - y1)
                }
                self.crop_regions.append(crop_region)
                print(f"✅ Added crop {len(self.crop_regions)}: {x1},{y1} to {x2},{y2} (area: {crop_region['area']})")
            else:
                print("⚠️ Crop too small, minimum size is 20x20 pixels")
            
            self._update_display()
    
    def _update_display(self):
        """Update the display with current crops and selection"""
        display_img = self.display_image.copy()
        
        # Draw existing crop regions
        for i, region in enumerate(self.crop_regions):
            # Convert to display coordinates
            x1 = int(region['x1'] * self.display_scale)
            y1 = int(region['y1'] * self.display_scale)
            x2 = int(region['x2'] * self.display_scale)
            y2 = int(region['y2'] * self.display_scale)
            
            # Draw rectangle
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"Crop {i+1}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display_img, (x1, y1-25), (x1+label_size[0]+10, y1), (0, 255, 0), -1)
            cv2.putText(display_img, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw current selection
        if self.drawing and self.start_point and self.end_point:
            cv2.rectangle(display_img, self.start_point, self.end_point, (255, 0, 0), 2)
        
        # Add instructions
        self._add_instructions(display_img)
        
        cv2.imshow(self.window_name, display_img)
    
    def _add_instructions(self, image):
        """Add instruction overlay to the image"""
        instructions = [
            "MANUAL CROPPING INSTRUCTIONS:",
            "• Left click + drag: Select crop region",
            "• Right click: Remove last crop",
            "• Mouse wheel: Zoom in/out",
            "• ENTER: Finish and save crops",
            "• ESC: Cancel and exit",
            f"• Crops selected: {len(self.crop_regions)}"
        ]
        
        y_offset = 30
        for instruction in instructions:
            # Background rectangle
            text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(image, (10, y_offset-20), (text_size[0]+20, y_offset+5), (0, 0, 0), -1)
            
            # Text
            color = (0, 255, 255) if instruction.startswith("•") else (255, 255, 255)
            cv2.putText(image, instruction, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
    
    def _run_interactive_session(self):
        """Main interactive session loop"""
        print("\n" + "="*60)
        print("🎯 MANUAL CROPPING SESSION STARTED")
        print("="*60)
        print("📋 Instructions:")
        print("   • Left click + drag to select banner regions")
        print("   • Right click to remove the last selection")
        print("   • Mouse wheel to zoom in/out")
        print("   • Press ENTER when finished")
        print("   • Press ESC to cancel")
        print("="*60)
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == ord('\r'):  # Enter key
                if self.crop_regions:
                    print(f"✅ Finished cropping with {len(self.crop_regions)} regions")
                    break
                else:
                    print("⚠️ No crops selected. Press ESC to exit or select regions.")
            
            elif key == 27:  # ESC key
                print("❌ Cropping cancelled")
                self.crop_regions = []
                break
            
            elif key == ord('r') or key == ord('R'):  # Reset
                self.crop_regions = []
                print("🔄 All crops cleared")
                self._update_display()
            
            elif key == ord('h') or key == ord('H'):  # Help
                self._show_help()
            
            elif key == ord('s') or key == ord('S'):  # Save preview
                self._save_preview()
    
    def _show_help(self):
        """Show detailed help"""
        print("\n" + "="*50)
        print("📖 DETAILED HELP")
        print("="*50)
        print("🖱️  Mouse Controls:")
        print("   • Left Click + Drag: Select crop region")
        print("   • Right Click: Remove last crop")
        print("   • Mouse Wheel: Zoom in/out")
        print("\n⌨️  Keyboard Controls:")
        print("   • ENTER: Finish and save crops")
        print("   • ESC: Cancel and exit")
        print("   • R: Reset (clear all crops)")
        print("   • H: Show this help")
        print("   • S: Save preview image")
        print("\n💡 Tips:")
        print("   • Select tight regions around banners")
        print("   • Minimum crop size: 20x20 pixels")
        print("   • You can select multiple regions")
        print("="*50)
    
    def _save_preview(self):
        """Save a preview image with crop regions marked"""
        try:
            preview_img = self.original_image.copy()
            
            for i, region in enumerate(self.crop_regions):
                cv2.rectangle(preview_img, 
                             (region['x1'], region['y1']), 
                             (region['x2'], region['y2']), 
                             (0, 255, 0), 3)
                
                label = f"Crop {i+1}"
                cv2.putText(preview_img, label, 
                           (region['x1']+5, region['y1']-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            preview_path = os.path.join(config.RESULTS_DIR, f"crop_preview_{timestamp}.jpg")
            cv2.imwrite(preview_path, preview_img)
            print(f"💾 Preview saved: {preview_path}")
            
        except Exception as e:
            print(f"❌ Error saving preview: {e}")
    
    def _create_crops_from_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """Create crop files from selected regions"""
        crops = []
        
        if not self.crop_regions:
            print("ℹ️ No crop regions selected")
            return crops
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, region in enumerate(self.crop_regions):
                # Extract crop
                crop_image = self.original_image[
                    region['y1']:region['y2'], 
                    region['x1']:region['x2']
                ]
                
                if crop_image.size == 0:
                    print(f"⚠️ Crop {i+1} is empty, skipping")
                    continue
                
                # Save crop
                crop_filename = f"manual_crop_{base_name}_{timestamp}_{i+1}.jpg"
                crop_path = os.path.join(config.MANUAL_CROPS_DIR, crop_filename)
                
                # Ensure directory exists
                os.makedirs(config.MANUAL_CROPS_DIR, exist_ok=True)
                
                cv2.imwrite(crop_path, crop_image)
                
                crop_info = {
                    'crop_path': crop_path,
                    'box': [region['x1'], region['y1'], region['x2'], region['y2']],
                    'confidence': 1.0,  # Manual crops have full confidence
                    'crop_index': i + 1,
                    'area': region['area']
                }
                
                crops.append(crop_info)
                print(f"✅ Created crop {i+1}: {crop_filename}")
            
            print(f"🎉 Successfully created {len(crops)} manual crops")
            
        except Exception as e:
            print(f"❌ Error creating crops: {e}")
        
        return crops
