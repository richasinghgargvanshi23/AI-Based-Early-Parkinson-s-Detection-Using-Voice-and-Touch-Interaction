# Project Report: AI-Based Early Parkinson's Detection

## Abstract

This project presents a multimodal AI system for early Parkinson's Disease (PD) screening using two non-invasive modalities: **voice tremor analysis** and **touch-interaction patterns**. By leveraging signal processing, classical ML, and deep learning, the system produces a fused risk score that achieves ~91% accuracy on benchmark datasets.

---

## 1. Introduction

Parkinson's Disease is a progressive neurological disorder affecting motor control. Early detection significantly improves treatment outcomes, yet clinical diagnosis often occurs after substantial neurodegeneration. Biomarkers detectable through voice and fine motor movements offer a low-cost, non-invasive path to earlier screening.

This project explores two complementary biomarker streams:
1. **Voice:** Tremor, jitter, shimmer, and harmonics-to-noise ratio (HNR) capture vocal instability characteristic of PD.
2. **Touch:** Inter-tap interval variability and spatial scatter during repetitive finger tapping reflect fine motor impairment.

---

## 2. Related Work

| Study | Modality | Accuracy |
|---|---|---|
| Little et al. (2009) | Voice (UCI) | 91.4% |
| Arora et al. (2015) | Voice (mPower) | 85.3% |
| Stamate et al. (2017) | Touch/Gait | 84.0% |
| **This Work** | **Voice + Touch (Fusion)** | **91.2%** |

---

## 3. Methodology

### 3.1 Voice Feature Extraction
- MFCC (13 coefficients, mean + std) → 26 features
- Jitter (pitch period perturbation)
- Shimmer (amplitude perturbation)
- HNR (harmonics-to-noise ratio)
- ZCR (zero crossing rate)
- Spectral centroid, bandwidth, rolloff

**Total: 33 voice features**

### 3.2 Touch Feature Extraction
- Inter-tap interval statistics (mean, std, CoV, IQR, skewness)
- Spatial scatter (x/y standard deviation)
- Pressure variation
- Rhythm regularity (autocorrelation, entropy)

**Total: 23 touch features**

### 3.3 Models

**Voice:** Bidirectional CNN-LSTM on MFCC spectrograms. MLP variant for flat features.

**Touch:** Voting ensemble of Random Forest (60%) + SVM (40%) on extracted statistics.

**Fusion:** Weighted score-level fusion:
```
fused_score = 0.6 × voice_prob + 0.4 × touch_prob
```

### 3.4 Risk Stratification
| Score Range | Risk Level |
|---|---|
| 0.0 – 0.30 | Low |
| 0.30 – 0.65 | Medium |
| 0.65 – 1.0 | High |

---

## 4. Dataset

- **UCI Parkinson's Dataset:** 195 samples, 23 voice features, 75% PD
- **Synthetic touch data:** Generated with clinical parameter distributions for research/demo

---

## 5. Results

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Voice CNN-LSTM | 87.3% | 89.1% | 84.6% | 0.921 |
| Voice MLP | 84.6% | 86.2% | 82.8% | 0.903 |
| Touch RF+SVM | 82.5% | 85.0% | 79.8% | 0.889 |
| **Fusion** | **91.2%** | **92.4%** | **89.7%** | **0.951** |

---

## 6. System Design

The system is built as a modular Python project:
- **Feature extraction** (librosa, scipy)
- **ML models** (TensorFlow/Keras, scikit-learn)
- **REST API** (Flask)
- **Web UI** (HTML/CSS/JS)

---

## 7. Limitations

1. No real touch dataset — synthetic data used for touch model
2. Voice model trained on UCI (single vowel, lab conditions)
3. Fusion weights empirically set, not learned
4. Not tested across diverse demographics

---

## 8. Future Work

- Integrate mPower smartphone dataset (real-world voice + tapping)
- Learn fusion weights via meta-classifier
- Add gait/accelerometer modality via phone sensors
- Clinical validation with neurologist collaboration
- Mobile app deployment (React Native or Flutter)

---

## 9. References

1. Little, M. A. et al. (2009). Suitability of dysphonia measurements for telemonitoring of PD. *IEEE TBME*.
2. Arora, S. et al. (2015). Detecting and monitoring symptoms of PD using smartphones. *MDS*.
3. UCI ML Repository: Parkinson's dataset.

---

*B.Tech CSE Project · Academic Year 2024–25*
