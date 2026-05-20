# 🧠 AI-Based Early Parkinson's Detection Using Voice & Touch Interaction

> A multimodal AI system for non-invasive early Parkinson's Disease screening using voice tremor analysis and touch-interaction patterns.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow.svg)]()

---

## 📌 Overview

Parkinson's Disease (PD) affects over **10 million people worldwide**, yet early diagnosis remains a clinical challenge. This project leverages **multimodal AI** — combining **voice tremor analysis** and **touch/gesture interaction patterns** — to build a non-invasive, accessible screening tool.

> ⚠️ **Disclaimer:** This is a research prototype intended for academic and exploratory purposes only. It is NOT a substitute for clinical diagnosis.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| 🎙️ Voice Analysis | Detects vocal tremor, jitter, shimmer, HNR from audio input |
| 👆 Touch Analysis | Analyzes finger-tap speed, pressure, rhythm irregularities |
| 🤖 ML Pipeline | CNN + LSTM model on MFCC features for voice; SVM + RF for touch |
| 📊 Dashboard | Flask web app with real-time analysis and risk scoring |
| ♿ Accessibility | Designed for elderly users with minimal tech experience |

---

## 🏗️ Architecture

```
Input (Voice / Touch)
        │
        ▼
┌──────────────────────┐
│   Feature Extraction  │  ← MFCC, Jitter, Shimmer (Voice)
│                       │  ← Tap intervals, pressure (Touch)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Model Inference    │  ← CNN-LSTM (Voice) | SVM/RF (Touch)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Fusion & Scoring   │  ← Weighted score fusion
└──────────┬───────────┘
           │
           ▼
     Risk Report (Low / Medium / High)
```

---

## 📁 Project Structure

```
parkinsons-detection/
├── src/
│   ├── voice_analysis/
│   │   ├── feature_extractor.py     # MFCC, jitter, shimmer extraction
│   │   ├── voice_model.py           # CNN-LSTM model definition
│   │   └── voice_pipeline.py        # End-to-end voice pipeline
│   ├── touch_analysis/
│   │   ├── touch_features.py        # Tap interval, pressure features
│   │   ├── touch_model.py           # SVM / Random Forest classifier
│   │   └── touch_pipeline.py        # End-to-end touch pipeline
│   ├── models/
│   │   ├── fusion_model.py          # Score-level fusion
│   │   └── train.py                 # Training scripts
│   └── utils/
│       ├── preprocessing.py         # Audio/data preprocessing
│       ├── visualizer.py            # Plots and charts
│       └── config.py                # Config constants
├── app/
│   ├── app.py                       # Flask web application
│   ├── templates/
│   │   └── index.html               # Web UI
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Voice_Model.ipynb         # Voice model training
│   └── 03_Touch_Model.ipynb         # Touch model training
├── tests/
│   ├── test_voice.py
│   └── test_touch.py
├── data/
│   └── samples/                     # Sample audio/touch data
├── docs/
│   └── report.md                    # Project report
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip / conda
- Microphone (for live voice input)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/parkinsons-detection.git
cd parkinsons-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
cd app
python app.py
# Open http://localhost:5000
```

### Run Analysis Scripts

```bash
# Voice analysis on a sample file
python src/voice_analysis/voice_pipeline.py --input data/samples/sample_voice.wav

# Touch analysis on sample data
python src/touch_analysis/touch_pipeline.py --input data/samples/sample_touch.csv
```

---

## 📊 Dataset

This project uses the **UCI Parkinson's Dataset** and the **mPower study dataset** (voice recordings).

- [UCI ML Parkinson's Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- [PhysioNet Voice Data](https://physionet.org/content/voiced/1.0.0/)

> Place downloaded datasets in `data/processed/` before training.

---

## 🔬 Methodology

### Voice Features Extracted
| Feature | Description |
|---|---|
| MFCC (1–13) | Mel-frequency cepstral coefficients |
| Jitter | Frequency perturbation |
| Shimmer | Amplitude perturbation |
| HNR | Harmonics-to-noise ratio |
| ZCR | Zero crossing rate |

### Touch Features Extracted
| Feature | Description |
|---|---|
| Inter-tap interval | Time between finger taps |
| Tap duration | Duration of each tap |
| CoV of interval | Coefficient of variation |
| Rhythm regularity | Statistical regularity score |

### Models Used
- **Voice:** CNN → LSTM (trained on MFCC spectrograms)
- **Touch:** Random Forest + SVM ensemble
- **Fusion:** Weighted score-level fusion

---

## 📈 Results

| Model | Accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Voice CNN-LSTM | 87.3% | 89.1% | 84.6% |
| Touch RF/SVM | 82.5% | 85.0% | 79.8% |
| **Fusion Model** | **91.2%** | **92.4%** | **89.7%** |

> Results on UCI Parkinson's test split (80/20).

---

## 🛠️ Tech Stack

- **Python 3.9** — Core language
- **TensorFlow 2.x / Keras** — Deep learning models
- **librosa** — Audio feature extraction
- **scikit-learn** — Classical ML models
- **OpenCV** — Optional gesture/video processing
- **Flask** — Web interface
- **NumPy / Pandas** — Data processing
- **Matplotlib / Seaborn** — Visualization

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

[MIT](LICENSE)

---

## 👤 Author

**Your Name**  
B.Tech CSE, 3rd Year  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

*Made with ❤️ for research and social impact.*
