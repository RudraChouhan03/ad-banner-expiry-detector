# 🎯 Ad Banner Expiry Detection System

<div align="center">

**AI-powered pipeline for locating advertisement banners, reading multilingual text, matching with a registry and reporting expiry status – all with CPU-only compatibility.**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV 4.8](https://img.shields.io/badge/OpenCV-4.8-green?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?style=for-the-badge)](https://ultralytics.com)
[![Flask 2.3](https://img.shields.io/badge/Flask-2.3-red?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 What's New 

- **🌐 Polyglot-free Hindi & English OCR** with character-based language detection and tri-engine fusion (Tesseract + EasyOCR + PaddleOCR)
- **📍 Enhanced GPS extraction** – multi-region scanning, pattern caching and rapid fallback strategies
- **🎯 YOLOv8 + contour fallback detection** with CPU-optimised inference and auto-/manual hybrid cropping
- **🖱️ Interactive OpenCV cropper** (zoom, pan, undo, touch-friendly) plus HTML5 canvas cropper in the web UI
- **🧠 Dynamic, language-aware matching thresholds** and advanced fuzzy/keyword/character similarity for mixed scripts
- **💾 Caching & memory management** – clears GPU/CPU cache automatically, learns successful patterns, supports batch processing
- **⚡ Rich CLI & Dev server** with batch mode, manual-crop only mode, cache clearing, statistics output
- **🔌 REST endpoints**: `/api/system_stats`, `/api/clear_cache`, `/api/manual_crop`

---

## 📋 Table of Contents

- [🗺️ Overview](#️-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [🎮 Usage](#-usage)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📑 License & Author](#-license--author)

---

## ⚠️ Data Usage Disclaimer

**NOTICE:** The sample data provided in `data/company_data.csv` is for **educational and testing purposes only**. 

- This data does not represent actual business entities or registrations
- Users are solely responsible for ensuring compliance with local regulations when using real data
- The authors assume no liability for misuse of this sample data
- Replace with legitimate data sources before production deployment

**By using this software, you acknowledge that the sample data is for testing only and agree to use actual, authorized data sources in production environments.**

---

## 🗺️ Overview

The system ingests a photo (mobile, CCTV, drone, etc.), extracts embedded **GPS coordinates**, detects **ad banners**, performs **multilingual OCR**, matches the content against a **registered-banner CSV**, and finally determines **expiry status**. Results are available from both a Flask web dashboard and a CLI – **no GPU required**.

### 🎯 Workflow
```
📤 Image Upload → 📍 GPS Extract → 🎯 Banner Detect → ✂️ Crop → 🔤 OCR → 🔍 Match → 📅 Expiry Check
```

---

## ✨ Key Features

### 🌍 **Location Intelligence**
- **📍 Multi-region OCR** with whitelist configs, early exit on first hit
- **🗺️ Validates coordinates** against extended India bounds

### 🎯 **Advanced Detection**
- **🤖 YOLOv8** model autoload (or environment override)
- **📐 Contour-based fallback** for model-less scenarios
- **⚡ CPU-optimized** inference for broad deployment

### ✂️ **Hybrid Cropping**
- **🤖 Auto-cropping** of each detection box
- **🖱️ Manual cropping**: OpenCV window (CLI) or HTML5 canvas (web)
- **🔄 Interactive controls**: zoom, pan, undo, touch-friendly

### 🔤 **Multilingual OCR**
- **🧠 Character-based language detection** (Hindi ⇆ English)
- **🔀 Tri-engine extraction** with confidence ranking
- **📊 Smart result selection** from multiple OCR engines

### 🔍 **Smart Matching**
- **🎨 Dynamic thresholds**: `pure_hindi` → 35%, `english_dominant` → 55%, `mixed_script` → 45%
- **🔍 Multiple methods**: Hindi-exact, keyword, fuzzy, mixed-script and character-overlap
- **📊 Confidence scoring** for match quality assessment

### 📅 **Expiry Management**
- **📊 Multiple date formats** supported
- **🚦 Clear status**: **VALID** / **EXPIRED** / **EXPIRES TODAY**
- **⏰ Real-time validation** based on current date

### 📊 **Analytics & Cache**
- **📈 Success counters**: GPS, OCR and matching success rates
- **💾 Intelligent caching** with pattern learning
- **🔄 Auto cleanup** + manual cache clearing API

---

## 🏗️ System Architecture

```
┌────────────┐ Upload ┌───────────┐ Detect ┌────────────┐
│ Web / CLI  │───────►│ Detector  │───────►│  Cropper   │
└────────────┘        └───────────┘        └────────────┘
      ▲                                           │
      │ Match/                                    |
      | Expiry       OCR (3 engines)              ▼
      │              ┌────────────┐         ┌────────────┐
┌────────────┐       │ OCR +      │◄────────│ Language   │
|  CSV / DB  | ◄─────│ Language   |         │ Detection  │
└────────────┘       │ Detection  │         └────────────┘               
                     └────────────┘

```

---

## 📁 Project Structure

```
ad_banner_expiry_detector/
├── 📦 app/                         # Core Python Package - Main Processing Engine
│ ├── 📄 init.py                    # Package initializer - Sets up the app module and handles imports
│ ├── 🎯 banner_detector.py         # Banner Detection Engine - Uses YOLOv8 AI to find banners in images, and also includes backup OpenCV method when AI model unavailable
│ ├── 🔤 ocr_processor.py           # Text Reading Engine - Extracts Hindi/English text from banners. Using 3 different OCR engines (Tesseract, EasyOCR, PaddleOCR)
│ ├── 🔍 csv_matcher.py             # Smart Matching System - Compares extracted text with registered banner stored in database using fuzzy matching and language-aware algorithms
│ ├── ✂️ hybrid_cropper.py          # Interactive Cropping Tool - Lets users manually select banner regions with zoom, pan, and click-drag functionality
│ └── 🧰 utils.py                   # Utility Functions - Common helper functions for GPS parsing, distance calculations, date handling, and text processing
├── 🌐 web/                         # Web Interface - User-friendly website for the system
│ ├── 🚀 app.py                     # Main Web Server - Flask application that handles web requests, file uploads, and displays results in browser
│ ├── 📄 templates/                 # HTML Page Templates
│ │ ├── 🏠 index.html               # Home Page - Main upload interface with drag-drop functionality
│ │ ├── ✂️ manual_crop.html         # Cropping Page - Interactive banner selection interface
│ │ └── 📊 results.html             # Results Page - Shows processing results, matches, and expiry status
│ └── 🎨 static/                    # Website Assets
│ ├── 🎨 css/styles.css             # Styling File - Makes the website look professional and responsive
│ └── ⚡ js/                        # JavaScript Files
│ ├── 🖱️ main.js                    # Main Website Logic - Handles file uploads and user interactions
│ └── ✂️ manual_crop.js             # Cropping Logic - Interactive canvas-based banner selection
├── 📂 data/                        # Data Storage - All system data and files
│ ├── 🤖 models/                    # AI Model Storage - Houses the YOLOv8 banner detection model
│ │ └── 📦 yolov8s_banner.pt        # YOLO Model File - Pre-trained AI model for banner detection
│ ├── 🖼️ test_images/               # Sample Images - Test photos with banners for trying the system
│ ├── 📊 results/                   # Processing Results - Stores processed images and cached data
│ │ ├── 🤖 auto_crops/              # Auto-detected Crops - Banners found automatically by AI
│ │ ├── ✂️ manual_crops/            # Manual Crops - Banners selected manually by users
│ │ ├── 🎯 detections/              # Detection Results - Images with detected banner boxes drawn
│ │ └── 📝 logs/                    # System Logs - Processing history and error records
│ └── 📊 company_data.csv           # Banner Database - Registered banner information with expiry dates (Sample data for testing - replace with real data for production)
├── 🖥️ main.py                      # Command Line Interface - Run the system from terminal/command prompt with options like batch processing and manual cropping
├── 🌐 run_webapp.py                # Development Server - Starts the web application with auto-reload when code changes (for developers)
├── ⚙️ config.py                    # System Configuration - All settings, thresholds, and parameters can be customized here or via environment variables
└── 📋 requirements.txt             # Python Dependencies - List of all required libraries and packages to be installed with 'pip install -r requirements.txt'
```

---

### 📝 **Detailed File Explanations**

#### 🔧 **Core Processing Files (`app/` directory)**

**`banner_detector.py`** - The main AI brain of the system. This file contains the `BannerDetector` class that:
- Loads and runs the YOLOv8 artificial intelligence model to automatically find advertisement banners in photos
- Provides a backup detection method using OpenCV (traditional computer vision) when the AI model isn't available
- Crops detected banner regions from the original image for further processing
- Optimized to run on regular computers without requiring expensive graphics cards (GPU)

**`ocr_processor.py`** - The text reading specialist. This file handles all text extraction with:
- Support for reading both Hindi (Devanagari script) and English text from banner images
- Uses three different OCR (Optical Character Recognition) engines to maximize accuracy
- Automatically detects which language the text is in and adjusts processing accordingly
- Extracts GPS coordinates from camera overlay text for location-based banner matching

**`csv_matcher.py`** - The smart matching system. This file compares extracted text with the database:
- Takes text read from banners and finds matching entries in the registered banner database
- Uses "fuzzy matching" - can find matches even if the text has small errors or differences
- Handles different languages with specialized matching algorithms for Hindi and English
- Calculates confidence scores to determine how certain the match is

**`hybrid_cropper.py`** - The interactive cropping tool. This provides manual banner selection:
- Creates an interactive window where users can click and drag to select banner regions
- Includes zoom and pan functionality for precise selection on large images
- Provides keyboard shortcuts and mouse controls for efficient operation  
- Useful when automatic detection misses banners or needs refinement

**`utils.py`** - The toolbox of helper functions. Contains commonly used utilities:
- GPS coordinate parsing from various camera overlay text formats
- Distance calculations between GPS points using the Haversine formula
- Date parsing and expiry status determination with support for multiple date formats
- Text processing functions for Hindi language normalization and cleaning

#### 🌐 **Web Interface Files (`web/` directory)**

**`app.py`** - The main web server application. This Flask-based server:
- Handles all web requests like file uploads, processing requests, and result display
- Provides a user-friendly interface accessible through any web browser
- Manages user sessions and temporary file storage
- Offers REST API endpoints for programmatic access to system functions

**`templates/`** - HTML page templates that define the website structure:
- **`index.html`** - The main homepage with file upload interface, feature descriptions, and getting started guide
- **`manual_crop.html`** - Interactive page for manual banner selection with HTML5 canvas-based cropping tool
- **`results.html`** - Comprehensive results display showing GPS data, detected banners, OCR text, and match results

**`static/`** - Website assets that make the interface attractive and functional:
- **`css/styles.css`** - Styling that makes the website look professional, responsive, and user-friendly
- **`js/main.js`** - JavaScript for file upload handling, drag-drop functionality, and user interaction
- **`js/manual_crop.js`** - Advanced cropping interface with zoom, pan, and multi-region selection

#### 🚀 **Main Application Files (Root Directory)**

**`main.py`** - Command-line interface for power users and automation:
- Process single images or batch process multiple files
- Support for manual cropping mode and various debug options
- Statistical reporting and cache management
- Ideal for automated workflows and server environments

**`run_webapp.py`** - Enhanced development server with helpful features:
- Automatically restarts the web server when code changes are detected
- Provides detailed logging and error reporting for developers
- Includes dependency checking and system health monitoring

**`config.py`** - Central configuration hub for the entire system:
- Contains all system settings, thresholds, and parameters
- Supports environment variable overrides for different deployment environments
- Includes optimization settings for different hardware configurations
- Language processing parameters and matching thresholds

**`requirements.txt`** - Python package dependencies list:
- Specifies exact versions of all required libraries and packages
- Ensures consistent behavior across different installations
- Includes both essential packages and optional enhancements

#### 📂 **Data Directory Structure**

**`data/models/`** - Stores the AI model files used for banner detection
**`data/test_images/`** - Sample images for testing and demonstration
**`data/results/`** - Cached processing results and temporary files
**`data/company_data.csv`** - The banner registration database (sample data for testing)

---

## 🚀 Quick Start

### 1️⃣ **Clone & Setup**
```bash
git clone https://github.com/RudraChouhan03/ad-banner-expiry-detector.git
cd ad_banner_expiry_detector

python -m venv venv
# Windows: 
venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ **Install Tesseract + Hindi Language Pack**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-hin

# Windows: Download from UB-Mannheim
https://github.com/UB-Mannheim/tesseract/wiki

# macOS
brew install tesseract tesseract-lang
```

### 3️⃣ **Launch Application**
```bash
# Web interface (recommended)
python run_webapp.py
# → Open: http://localhost:5000

# CLI processing
python main.py sample.jpg --manual-crop --debug-gps
```

---

## ⚙️ Configuration

All settings are in **`config.py`** and can be overridden with environment variables:

| **Environment Variable**  | **Purpose** | **Default** |
|---------------------------|-------------|-------------|
| `BANNER_MODEL_PATH`       | Custom YOLO checkpoint | `data/models/yolov8s_banner.pt` |
| `BANNER_CSV_PATH`         | Banner registry database | `data/company_data.csv` |
| `BANNER_RESULTS_DIR`      | Results cache directory | `data/results/` |
| `BANNER_GPS_THRESHOLD`    | GPS proximity in meters | `2500` |
| `BANNER_OCR_LANGS`        | Tesseract language string | `hin+eng` |

### 📝 Example `.env` file:
```bash
BANNER_MODEL_PATH=data/models/custom_yolo.pt
BANNER_GPS_THRESHOLD=5000
BANNER_OCR_LANGS=hin+eng+san
FLASK_DEBUG=true
```

---

## 🎮 Usage

### 🌐 **Web Interface**
1. **Navigate** to http://localhost:5000
2. **Upload** image (drag-drop or click, max 8MB)
3. **Configure** options:
   - ☑️ **Manual crop** for precise banner selection
   - ☑️ **Skip no-GPS** images
4. **View results**: GPS, detections, OCR text, matches, expiry status

### 💻 **Command Line**

| **Command** | **Description** |
|-------------|-----------------|
| `python main.py img.jpg` | Process single image |
| `python main.py *.jpg` | Batch process with wildcards |
| `python main.py img.jpg --manual-crop` | Interactive cropping mode |
| `python main.py img.jpg --skip-no-gps` | Skip images without GPS |
| `python main.py dir/ --clear-cache` | Clear cache before processing |
| `python main.py --stats` | Show system statistics |

### 🔌 **REST API**

| **Endpoint** | **Method** | **Description** |
|--------------|------------|-----------------|
| `/upload` | `POST` | Upload image for processing |
| `/api/system_stats` | `GET` | Get processing statistics |
| `/api/clear_cache` | `POST` | Clear all caches |
| `/api/manual_crop` | `POST` | Submit manual crop coordinates |

**Example API usage:**
```bash
# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload

# Get system stats
curl http://localhost:5000/api/system_stats

# Clear cache
curl -X POST http://localhost:5000/api/clear_cache
```

---

## 🛠️ Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| **❌ TesseractNotFoundError** | Install Tesseract or set `pytesseract.pytesseract.tesseract_cmd` path |
| **❌ YOLO model missing** | System auto-falls back to contour detection, or set `BANNER_MODEL_PATH` |
| **⚠️ High RAM usage** | Reduce image resolution in config, or process in smaller batches |
| **⚠️ EasyOCR/PaddleOCR warnings** | Optional engines - install with `pip install easyocr paddleocr` or ignore |
| **❌ Permission denied (Linux)** | Run with `sudo` or check file permissions in data directory |
| **❌ Port 5000 already in use** | Change port: `python run_webapp.py --port 8080` |

### 🔍 **Debug Mode**
```bash
# Enable detailed logging
python main.py image.jpg --debug-gps --debug-ocr --debug-matching
```

### 📊 **Performance Issues**
```bash
# Check system stats
python main.py --stats

# Clear accumulated cache
python main.py --clear-cache
```

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 🔄 **Development Workflow**
1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Code** following PEP-8 standards
4. **Add** type hints and docstrings
5. **Format** with Black: `black .`
6. **Test** your changes: `pytest`
7. **Commit** with clear messages
8. **Submit** pull request with description

### 🧪 **Running Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run test suite
pytest tests/ -v

# Generate coverage report
pytest --cov=app tests/
```

### 📝 **Code Standards**
- Follow **PEP-8** style guidelines
- Add **type hints** for function parameters and returns
- Include **docstrings** for all public functions
- Keep functions **focused and small**
- Use **meaningful variable names**

---

## 📑 License & Author

### 📄 **License**
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 👤 **Author**
**Rudra Chouhan**  
📧 rudrachouhan0305@gmail.com  
🐙 [GitHub Profile](https://github.com/RudraChouhan03)

### 🙏 **Acknowledgments**
- **Ultralytics** for YOLOv8 framework
- **Tesseract OCR** community for multilingual support
- **OpenCV** team for computer vision tools
- **Flask** team for the web framework

---

<div align="center">

**⭐ Star this repo if it helped you! ⭐**

[🐛 Report Bug](https://github.com/RudraChouhan03/ad-banner-expiry-detector/issues) • [💡 Request Feature](https://github.com/RudraChouhan03/ad-banner-expiry-detector/issues) 

</div>2
