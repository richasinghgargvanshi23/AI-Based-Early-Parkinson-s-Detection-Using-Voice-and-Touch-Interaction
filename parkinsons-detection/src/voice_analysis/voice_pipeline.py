# src/voice_analysis/voice_pipeline.py

"""
End-to-End Voice Analysis Pipeline
Usage:
    python voice_pipeline.py --input <audio.wav>
    python voice_pipeline.py --record   (records mic input)
"""

import os
import argparse
import numpy as np
import librosa
import joblib
import tensorflow as tf

from src.voice_analysis.feature_extractor import extract_all_features, load_audio, extract_mfcc
from src.utils.config import (
    SAMPLE_RATE, DURATION, VOICE_MODEL_PATH,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, RISK_LABELS
)


def preprocess_for_cnn(filepath: str, max_len: int = 128) -> np.ndarray:
    """
    Load audio and create MFCC spectrogram for CNN-LSTM input.
    Returns shape: (1, max_len, n_mfcc, 1)
    """
    y, sr = load_audio(filepath)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)

    # Pad or truncate to fixed length
    if mfccs.shape[1] < max_len:
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]

    # Normalize
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-9)

    # Shape: (1, time, n_mfcc, 1)
    return mfccs.T[np.newaxis, :, :, np.newaxis]


def record_audio(duration: int = DURATION, sr: int = SAMPLE_RATE,
                 save_path: str = "data/samples/live_recording.wav"):
    """Record audio from microphone."""
    try:
        import pyaudio
        import wave

        print(f"🎙️  Recording for {duration} seconds... Speak now!")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1,
                        rate=sr, input=True,
                        frames_per_buffer=1024)
        frames = []
        for _ in range(0, int(sr / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sr)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f"✅ Recording saved to {save_path}")
        return save_path

    except ImportError:
        print("❌ pyaudio not installed. Install with: pip install pyaudio")
        return None


def predict_voice(filepath: str, model=None, use_mlp: bool = True) -> dict:
    """
    Run voice analysis on a WAV file.

    Args:
        filepath: Path to .wav file
        model: Pre-loaded model (optional)
        use_mlp: Use flat feature MLP instead of CNN-LSTM

    Returns:
        dict with probability, risk_level, features
    """
    print(f"\n📂 Analyzing: {filepath}")

    # Extract features
    features = extract_all_features(filepath)
    print(f"✅ Features extracted — shape: {features.shape}")

    # Load model if not provided
    if model is None:
        if os.path.exists(VOICE_MODEL_PATH):
            model = tf.keras.models.load_model(VOICE_MODEL_PATH)
            print("✅ Model loaded from disk.")
        else:
            # No trained model — use a mock prediction for demo purposes
            print("⚠️  No trained model found. Running feature-based heuristic.")
            return _heuristic_prediction(features)

    if use_mlp:
        x = features[np.newaxis, :]  # (1, 33)
    else:
        x = preprocess_for_cnn(filepath)

    prob = float(model.predict(x, verbose=0)[0][0])
    return _format_result(prob, features)


def _heuristic_prediction(features: np.ndarray) -> dict:
    """
    Simple heuristic when no trained model is available.
    Uses known clinical thresholds for jitter and shimmer.
    """
    # Feature indices (from get_feature_names)
    jitter_idx = 26
    shimmer_idx = 27
    hnr_idx = 28

    jitter = features[jitter_idx] if len(features) > jitter_idx else 0
    shimmer = features[shimmer_idx] if len(features) > shimmer_idx else 0
    hnr = features[hnr_idx] if len(features) > hnr_idx else 0

    # Clinical thresholds (approx)
    score = 0.0
    if jitter > 0.01:
        score += 0.3
    if shimmer > 0.05:
        score += 0.3
    if hnr < 15:
        score += 0.2

    prob = min(score, 0.95)
    return _format_result(prob, features)


def _format_result(prob: float, features: np.ndarray) -> dict:
    """Format the prediction result."""
    if prob < LOW_RISK_THRESHOLD:
        risk = "low"
    elif prob < HIGH_RISK_THRESHOLD:
        risk = "medium"
    else:
        risk = "high"

    return {
        "probability": round(prob, 4),
        "risk_level": risk,
        "risk_message": RISK_LABELS[risk],
        "features": features.tolist(),
    }


def run_pipeline(args):
    if args.record:
        filepath = record_audio()
        if not filepath:
            return
    else:
        filepath = args.input
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return

    result = predict_voice(filepath)

    print("\n" + "═" * 50)
    print("       VOICE ANALYSIS RESULT")
    print("═" * 50)
    print(f"  Risk Level  : {result['risk_level'].upper()}")
    print(f"  PD Score    : {result['probability']:.4f}")
    print(f"  Assessment  : {result['risk_message']}")
    print("═" * 50)
    print("\n⚠️  This tool is for research only, not clinical diagnosis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parkinson's Voice Analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Path to .wav audio file")
    group.add_argument("--record", action="store_true", help="Record from microphone")
    args = parser.parse_args()
    run_pipeline(args)
