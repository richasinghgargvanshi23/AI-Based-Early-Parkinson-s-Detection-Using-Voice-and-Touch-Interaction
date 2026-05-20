# src/utils/config.py

# ─────────────────────────────────────────
# Audio / Voice Config
# ─────────────────────────────────────────
SAMPLE_RATE = 22050          # Hz
DURATION = 5                 # seconds to record
N_MFCC = 13                  # number of MFCC features
HOP_LENGTH = 512
N_FFT = 2048
FRAME_LENGTH = 2048

# ─────────────────────────────────────────
# Touch / Tap Config
# ─────────────────────────────────────────
MIN_TAPS = 10                # minimum taps needed for analysis
TAP_TIMEOUT = 0.5            # seconds between taps to detect pause

# ─────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────
VOICE_MODEL_PATH = "models/saved/voice_cnn_lstm.h5"
TOUCH_MODEL_PATH = "models/saved/touch_rf.pkl"

# ─────────────────────────────────────────
# Fusion Weights
# ─────────────────────────────────────────
VOICE_WEIGHT = 0.6
TOUCH_WEIGHT = 0.4

# ─────────────────────────────────────────
# Risk Thresholds
# ─────────────────────────────────────────
LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.65

# Labels
RISK_LABELS = {
    "low": "Low Risk — No significant indicators detected.",
    "medium": "Medium Risk — Some indicators present. Consult a physician.",
    "high": "High Risk — Strong indicators detected. Please seek medical evaluation.",
}
