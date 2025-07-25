#!/usr/bin/env python3
"""
Enhanced Development Server for Ad Banner Expiry Detection System
Includes monitoring, auto-restart, and enhanced debugging features
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

try:
    import config
    from app.utils import create_directories
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

class EnhancedFileHandler(FileSystemEventHandler):
    """Enhanced file system event handler for auto-restart"""
    
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_restart = 0
        self.restart_delay = 1  # Minimum seconds between restarts
        
        # Files/directories to watch
        self.watch_extensions = {'.py', '.html', '.css', '.js', '.json'}
        self.watch_directories = {'app', 'web', 'templates', 'static'}
        
        # Files to ignore
        self.ignore_patterns = {
            '__pycache__',
            '.pyc',
            '.pyo',
            '.git',
            'node_modules',
            '.pytest_cache',
            'logs'
        }
    
    def should_restart(self, file_path):
        """Determine if file change should trigger restart"""
        path = Path(file_path)
        
        # Check if any ignored pattern is in the path
        for ignore in self.ignore_patterns:
            if ignore in str(path):
                return False
        
        # Check if file extension is watched
        if path.suffix not in self.watch_extensions:
            return False
        
        # Check if directory is watched
        for watch_dir in self.watch_directories:
            if watch_dir in path.parts:
                return True
        
        # Check if it's a root-level Python file
        if path.parent == project_root and path.suffix == '.py':
            return True
        
        return False
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        current_time = time.time()
        if (current_time - self.last_restart) < self.restart_delay:
            return
        
        if self.should_restart(event.src_path):
            print(f"[RELOAD] File changed: {event.src_path}")
            self.last_restart = current_time
            self.restart_callback()

class EnhancedDevServer:
    """Enhanced development server with monitoring and auto-restart"""
    
    def __init__(self):
        self.process = None
        self.observer = None
        self.running = False
        self.restart_count = 0
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        try:
            required_dirs = [
                config.RESULTS_DIR,
                config.AUTO_CROPS_DIR,
                config.MANUAL_CROPS_DIR,
                config.DETECTIONS_DIR,
                config.LOGS_DIR,
                config.UPLOAD_FOLDER,
                os.path.join(config.UPLOAD_FOLDER, "crops"),
                # Cache directories
                os.path.join(config.RESULTS_DIR, "gps_cache"),
                os.path.join(config.RESULTS_DIR, "ocr_cache"),
                os.path.join(config.RESULTS_DIR, "matching_cache")
            ]
            
            create_directories(required_dirs)
            print(f"[OK] Created {len(required_dirs)} required directories")
            
        except Exception as e:
            print(f"[WARNING]  Error creating directories: {e}")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("[SEARCH] Checking dependencies...")
        
        critical_deps = [
            ('Flask', 'flask'),
            ('OpenCV', 'cv2'),
            ('NumPy', 'numpy'),
            ('Pandas', 'pandas'),
            ('Pillow', 'PIL'),
            ('Tesseract', 'pytesseract')
        ]
        
        optional_deps = [
            ('EasyOCR', 'easyocr'),
            ('PaddleOCR', 'paddleocr'),
            ('FuzzyWuzzy', 'fuzzywuzzy'),
            ('YOLO', 'ultralytics')
        ]
        
        missing_critical = []
        missing_optional = []
        
        for name, module in critical_deps:
            try:
                __import__(module)
                print(f"  [OK] {name}")
            except ImportError:
                missing_critical.append(name)
                print(f"  [ERROR] {name} (CRITICAL)")
        
        for name, module in optional_deps:
            try:
                __import__(module)
                print(f"  [OK] {name}")
            except ImportError:
                missing_optional.append(name)
                print(f"  [WARNING]  {name} (optional)")
        
        if missing_critical:
            print(f"\n[ERROR] Missing critical dependencies: {', '.join(missing_critical)}")
            print("   Install with: pip install -r requirements.txt")
            return False
        
        if missing_optional:
            print(f"\n[WARNING]  Missing optional dependencies: {', '.join(missing_optional)}")
            print("   Some features may not work optimally")
        
        return True
    
    def start_flask_app(self):
        """Start the Flask application"""
        try:
            env = os.environ.copy()
            env['FLASK_ENV'] = 'development'
            env['FLASK_DEBUG'] = '1'
            env['PYTHONPATH'] = str(project_root)
            
            cmd = [
                sys.executable, 
                str(project_root / 'web' / 'app.py')
            ]
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start Flask app: {e}")
            return False
    
    def monitor_output(self):
        """Monitor Flask app output"""
        if not self.process:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ''):
                if line.strip():
                    timestamp = time.strftime('%H:%M:%S')
                    print(f"[{timestamp}] {line.strip()}")
                
                if not self.running:
                    break
        except Exception as e:
            print(f"[WARNING]  Output monitoring error: {e}")
    
    def restart_app(self):
        """Restart the Flask application"""
        print("\n[RELOAD] Restarting application...")
        self.restart_count += 1
        
        # Stop current process
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        # Start new process
        if self.start_flask_app():
            print(f"[OK] Application restarted (restart #{self.restart_count})")
            
            # Start output monitoring in background
            output_thread = threading.Thread(target=self.monitor_output, daemon=True)
            output_thread.start()
        else:
            print("[ERROR] Failed to restart application")
    
    def setup_file_watcher(self):
        """Setup file system watcher for auto-restart"""
        try:
            self.observer = Observer()
            event_handler = EnhancedFileHandler(self.restart_app)
            
            # Watch project directory
            self.observer.schedule(event_handler, str(project_root), recursive=True)
            self.observer.start()
            
            print("👁️  File watcher started (auto-restart enabled)")
            return True
            
        except Exception as e:
            print(f"[WARNING]  Could not setup file watcher: {e}")
            return False
    
    def display_info(self):
        """Display startup information"""
        print("\n" + "="*60)
        print("[START] AD BANNER DETECTION SYSTEM")
        print("="*60)
        print(f"[LOCATION] Project Root: {project_root}")
        print(f"[LANG] Web URL: http://localhost:5000")
        print(f"📁 Upload Directory: {config.UPLOAD_FOLDER}")
        print(f"[CACHE] Results Directory: {config.RESULTS_DIR}")
        print(f"[HINDI] Hindi Support: {'[OK]' if config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr'] else '[ERROR]'}")
        print(f"[LOCATION] GPS Optimization: {'[OK]' if config.GPS_EXTRACTION_CONFIG['enable_region_caching'] else '[ERROR]'}")
        print(f"[CACHE] Caching: {'[OK]' if config.CACHE_CONFIG['enable_pattern_learning'] else '[ERROR]'}")
        print(f"🖥️  CPU Optimization: {'[OK]' if config.CPU_OPTIMIZATION_CONFIG['memory_optimization'] else '[ERROR]'}")
        print("="*60)
        print("\n📋 Available Commands:")
        print("  • Ctrl+C: Stop server")
        print("  • Ctrl+R: Manual restart")
        print("  • Ctrl+S: Show statistics")
        print("  • File changes trigger auto-restart")
        print("\n🔗 Access Points:")
        print("  • Main Interface: http://localhost:5000")
        print("  • System Stats: http://localhost:5000/api/system_stats")
        print("  • Clear Cache: POST http://localhost:5000/api/clear_cache")
        print("\n[FAST] Features Enabled:")
        
        features = []
        if config.HINDI_PROCESSING_CONFIG['enable_hindi_ocr']:
            features.append("[TEXT] Hindi Text Processing")
        if config.GPS_EXTRACTION_CONFIG['enable_region_caching']:
            features.append("[GPS]  Enhanced GPS Extraction")
        if config.CACHE_CONFIG['enable_pattern_learning']:
            features.append("[AI] Pattern Learning")
        if config.CPU_OPTIMIZATION_CONFIG['memory_optimization']:
            features.append("[FAST] CPU Optimization")
        
        for feature in features:
            print(f"  • {feature}")
        
        if not features:
            print("  • Basic functionality only")
        
        print("\n" + "="*60)
    
    def show_statistics(self):
        """Display system statistics"""
        try:
            print("\n[STATS] SYSTEM STATISTICS")
            print("-" * 40)
            
            # Directory sizes
            def get_dir_size(path):
                try:
                    total = 0
                    if os.path.exists(path):
                        for dirpath, dirnames, filenames in os.walk(path):
                            for filename in filenames:
                                total += os.path.getsize(os.path.join(dirpath, filename))
                    return total / (1024 * 1024)  # MB
                except:
                    return 0
            
            upload_size = get_dir_size(config.UPLOAD_FOLDER)
            results_size = get_dir_size(config.RESULTS_DIR)
            
            print(f"📁 Upload Directory: {upload_size:.1f} MB")
            print(f"[STATS] Results Directory: {results_size:.1f} MB")
            print(f"[RELOAD] App Restarts: {self.restart_count}")
            
            # Process info
            if self.process:
                print(f"🔗 Process ID: {self.process.pid}")
                print(f"⏰ Status: {'Running' if self.process.poll() is None else 'Stopped'}")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"[WARNING]  Error getting statistics: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[CLEAN] Cleaning up...")
        
        self.running = False
        
        # Stop file observer
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2)
                print("[OK] File watcher stopped")
            except Exception as e:
                print(f"[WARNING]  Error stopping file watcher: {e}")
        
        # Stop Flask process
        if self.process:
            try:
                print("[RELOAD] Stopping Flask application...")
                self.process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.process.wait(timeout=5)
                    print("[OK] Flask application stopped gracefully")
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                    print("[WARNING]  Flask application force killed")
                    
            except Exception as e:
                print(f"[WARNING]  Error stopping Flask app: {e}")
        
        print("[OK] Cleanup completed")
    
    def handle_keyboard_interrupt(self, signum, frame):
        """Handle Ctrl+C"""
        print("\n[WARNING]  Interrupt received...")
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main run method"""
        print("[START] Starting Enhanced Development Server...")
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_keyboard_interrupt)
        signal.signal(signal.SIGTERM, self.handle_keyboard_interrupt)
        
        # Check dependencies
        if not self.check_dependencies():
            print("[ERROR] Cannot start server due to missing dependencies")
            return False
        
        # Ensure directories
        self.ensure_directories()
        
        # Display system information
        self.display_info()
        
        # Start Flask app
        if not self.start_flask_app():
            print("[ERROR] Failed to start Flask application")
            return False
        
        self.running = True
        
        # Setup file watcher for auto-restart
        self.setup_file_watcher()
        
        # Start output monitoring
        output_thread = threading.Thread(target=self.monitor_output, daemon=True)
        output_thread.start()
        
        print(f"\n[OK] Server started successfully!")
        print(f"[LANG] Access your application at: http://localhost:5000")
        print(f"[TEXT] Press Ctrl+C to stop the server")
        
        # Main event loop
        try:
            while self.running:
                time.sleep(1)
                
                # Check if Flask process is still running
                if self.process and self.process.poll() is not None:
                    print("[WARNING]  Flask process stopped unexpectedly, restarting...")
                    self.restart_app()
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
        
        return True

def main():
    """Main entry point"""
    print("[CONFIG] Ad Banner Detection - Development Server")
    print(f"📅 Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if watchdog is available for file monitoring
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("[WARNING]  Watchdog not installed - auto-restart disabled")
        print("   Install with: pip install watchdog")
    
    # Create and run server
    server = EnhancedDevServer()
    
    try:
        success = server.run()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        exit_code = 1
    
    print(f"\n📅 Server stopped at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
