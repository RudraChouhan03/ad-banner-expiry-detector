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
├── 📦 app/                          # Core Python package
│   ├── 🎯 banner_detector.py        # YOLOv8 + contour detection
│   ├── 🔤 ocr_processor.py          # Multi-engine OCR processing
│   ├── 🔍 csv_matcher.py            # Smart matching algorithms
│   ├── ✂️ hybrid_cropper.py         # Interactive cropping tools
│   └── 🧰 utils.py                  # Utility functions & helpers
├── 🌐 web/                          # Flask front-end
│   ├── 🚀 app.py                    # Main Flask application
│   ├── 📄 templates/                # HTML templates
│   └── 🎨 static/                   # CSS, JS, and assets
├── 📂 data/
│   ├── 🤖 models/                   # YOLO model storage
│   ├── 🖼️ test_images/              # Sample test images
|   ├──  results/                    # Processed results cache 
│   └── 📊 company_data.csv          # Banner registry
├── 🚀 main.py                       # CLI entry-point
├── 🌐 run_webapp.py                 # Dev server with auto-reload
├── ⚙️ config.py                     # Central configuration
└── 📋 requirements.txt              # Dependencies
```

---

## 🚀 Quick Start

### 1️⃣ **Clone & Setup**
```bash
git clone https://github.com/your-username/ad-banner-expiry-detector.git
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
# https://github.com/UB-Mannheim/tesseract/wiki

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

[🐛 Report Bug](https://github.com/your-repo/issues) • [💡 Request Feature](https://github.com/your-repo/issues) • [💬 Discussions](https://github.com/your-repo/discussions)

</div>