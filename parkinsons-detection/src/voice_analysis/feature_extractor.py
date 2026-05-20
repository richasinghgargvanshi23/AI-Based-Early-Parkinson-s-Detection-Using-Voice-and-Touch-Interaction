# src/voice_analysis/feature_extractor.py

"""
Voice Feature Extraction for Parkinson's Detection
Extracts: MFCC, Jitter, Shimmer, HNR, ZCR, Spectral features
"""

import numpy as np
import librosa
import librosa.effects
import scipy.stats as stats
from src.utils.config import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT


def load_audio(filepath: str, sr: int = SAMPLE_RATE):
    """Load an audio file and return waveform + sample rate."""
    y, sr = librosa.load(filepath, sr=sr)
    return y, sr


def extract_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract MFCC features (mean + std of each coefficient)."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])  # shape: (N_MFCC*2,)


def extract_jitter(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate jitter (cycle-to-cycle frequency perturbation).
    Uses zero-crossing based pitch period estimation.
    """
    # Get fundamental frequency via librosa pyin
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_voiced = f0[voiced_flag]
    if len(f0_voiced) < 2:
        return 0.0
    periods = 1.0 / (f0_voiced + 1e-9)
    diffs = np.abs(np.diff(periods))
    jitter = np.mean(diffs) / (np.mean(periods) + 1e-9)
    return float(jitter)


def extract_shimmer(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate shimmer (cycle-to-cycle amplitude perturbation).
    """
    # RMS energy in short frames as proxy for amplitude
    frame_length = 512
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    if len(rms) < 2:
        return 0.0
    diffs = np.abs(np.diff(rms))
    shimmer = np.mean(diffs) / (np.mean(rms) + 1e-9)
    return float(shimmer)


def extract_hnr(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """
    Estimate Harmonics-to-Noise Ratio (HNR).
    Higher HNR → clearer voice; lower HNR → more noise/hoarseness.
    """
    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_energy = np.sum(harmonic ** 2)
    noise_energy = np.sum(percussive ** 2) + 1e-9
    hnr = 10 * np.log10(harmonic_energy / noise_energy + 1e-9)
    return float(hnr)


def extract_zcr(y: np.ndarray) -> float:
    """Zero Crossing Rate — related to voice roughness."""
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return float(np.mean(zcr))


def extract_spectral_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Extract spectral centroid, bandwidth, and rolloff."""
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return {
        "spectral_centroid": float(centroid),
        "spectral_bandwidth": float(bandwidth),
        "spectral_rolloff": float(rolloff),
    }


def extract_all_features(filepath: str) -> np.ndarray:
    """
    Full feature extraction pipeline for a voice file.
    Returns a 1D feature vector ready for model inference.
    """
    y, sr = load_audio(filepath)

    mfcc_feats = extract_mfcc(y, sr)                    # 26 features
    jitter = np.array([extract_jitter(y, sr)])           # 1
    shimmer = np.array([extract_shimmer(y, sr)])         # 1
    hnr = np.array([extract_hnr(y, sr)])                 # 1
    zcr = np.array([extract_zcr(y)])                     # 1
    spectral = extract_spectral_features(y, sr)
    spectral_arr = np.array(list(spectral.values()))     # 3

    feature_vector = np.concatenate([
        mfcc_feats, jitter, shimmer, hnr, zcr, spectral_arr
    ])
    return feature_vector   # shape: (33,)


def get_feature_names() -> list:
    """Return ordered list of feature names (for logging/display)."""
    mfcc_mean = [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    mfcc_std = [f"mfcc_std_{i}" for i in range(N_MFCC)]
    return mfcc_mean + mfcc_std + ["jitter", "shimmer", "hnr", "zcr",
                                    "spectral_centroid", "spectral_bandwidth",
                                    "spectral_rolloff"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        features = extract_all_features(path)
        names = get_feature_names()
        print("\n=== Extracted Voice Features ===")
        for name, val in zip(names, features):
            print(f"  {name:<25}: {val:.6f}")
    else:
        print("Usage: python feature_extractor.py <audio_file.wav>")
