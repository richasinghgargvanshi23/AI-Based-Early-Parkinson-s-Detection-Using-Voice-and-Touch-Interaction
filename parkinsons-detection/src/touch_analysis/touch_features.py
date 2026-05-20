# src/touch_analysis/touch_features.py

"""
Touch / Tap Feature Extraction for Parkinson's Detection
Input: CSV or list of tap events {timestamp, x, y, pressure, duration}
Features: Inter-tap interval stats, rhythm regularity, pressure variation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict


def load_tap_data(filepath: str) -> pd.DataFrame:
    """
    Load tap data from CSV.
    Expected columns: timestamp, x, y, pressure, duration
    """
    df = pd.read_csv(filepath)
    required_cols = ["timestamp", "x", "y"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_inter_tap_intervals(timestamps: np.ndarray) -> np.ndarray:
    """Compute time between consecutive taps (ms)."""
    return np.diff(timestamps) * 1000  # convert to ms


def extract_interval_features(intervals: np.ndarray) -> Dict[str, float]:
    """Statistical features of inter-tap intervals."""
    if len(intervals) == 0:
        return {}
    return {
        "iti_mean": float(np.mean(intervals)),
        "iti_std": float(np.std(intervals)),
        "iti_cv": float(np.std(intervals) / (np.mean(intervals) + 1e-9)),  # CoV
        "iti_median": float(np.median(intervals)),
        "iti_iqr": float(stats.iqr(intervals)),
        "iti_skewness": float(stats.skew(intervals)),
        "iti_kurtosis": float(stats.kurtosis(intervals)),
        "iti_min": float(np.min(intervals)),
        "iti_max": float(np.max(intervals)),
        "iti_range": float(np.max(intervals) - np.min(intervals)),
    }


def extract_spatial_features(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Spatial variation in tap positions.
    Parkinson's patients often show increased spatial scatter.
    """
    return {
        "x_std": float(np.std(x)),
        "y_std": float(np.std(y)),
        "x_range": float(np.max(x) - np.min(x)),
        "y_range": float(np.max(y) - np.min(y)),
        "spatial_scatter": float(np.sqrt(np.std(x) ** 2 + np.std(y) ** 2)),
    }


def extract_pressure_features(pressure: np.ndarray) -> Dict[str, float]:
    """
    Pressure variation features.
    (Available if device supports pressure sensing)
    """
    if pressure is None or len(pressure) == 0:
        return {"pressure_mean": 0.0, "pressure_std": 0.0, "pressure_cv": 0.0}
    return {
        "pressure_mean": float(np.mean(pressure)),
        "pressure_std": float(np.std(pressure)),
        "pressure_cv": float(np.std(pressure) / (np.mean(pressure) + 1e-9)),
    }


def extract_rhythm_regularity(intervals: np.ndarray) -> Dict[str, float]:
    """
    Rhythm regularity using DFA (Detrended Fluctuation Analysis) proxy.
    Regular rhythm → low CV; Parkinson's → higher irregularity.
    """
    if len(intervals) < 4:
        return {"rhythm_regularity": 0.0, "rhythm_entropy": 0.0}

    # Sample entropy approximation
    n = len(intervals)
    normalized = (intervals - np.mean(intervals)) / (np.std(intervals) + 1e-9)
    autocorr = np.correlate(normalized, normalized, mode='full')[n - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-9)

    regularity = float(autocorr[1]) if len(autocorr) > 1 else 0.0

    # Histogram entropy
    hist, _ = np.histogram(intervals, bins=min(10, len(intervals) // 2 + 1),
                           density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log(hist + 1e-9)))

    return {
        "rhythm_regularity": regularity,
        "rhythm_entropy": entropy,
    }


def extract_all_touch_features(df: pd.DataFrame) -> np.ndarray:
    """
    Full touch feature extraction from a tap dataframe.
    Returns a 1D feature vector.
    """
    timestamps = df["timestamp"].values
    x = df["x"].values
    y = df["y"].values
    pressure = df["pressure"].values if "pressure" in df.columns else np.array([])
    duration = df["duration"].values if "duration" in df.columns else np.array([])

    intervals = compute_inter_tap_intervals(timestamps)

    feat_dict = {}
    feat_dict.update(extract_interval_features(intervals))
    feat_dict.update(extract_spatial_features(x, y))
    feat_dict.update(extract_pressure_features(pressure))
    feat_dict.update(extract_rhythm_regularity(intervals))

    # Tap duration stats
    if len(duration) > 0:
        feat_dict["tap_duration_mean"] = float(np.mean(duration))
        feat_dict["tap_duration_std"] = float(np.std(duration))
    else:
        feat_dict["tap_duration_mean"] = 0.0
        feat_dict["tap_duration_std"] = 0.0

    # Total taps
    feat_dict["total_taps"] = float(len(timestamps))

    feature_vector = np.array(list(feat_dict.values()), dtype=np.float32)
    return feature_vector


def get_touch_feature_names() -> List[str]:
    """Return ordered feature names."""
    return [
        "iti_mean", "iti_std", "iti_cv", "iti_median", "iti_iqr",
        "iti_skewness", "iti_kurtosis", "iti_min", "iti_max", "iti_range",
        "x_std", "y_std", "x_range", "y_range", "spatial_scatter",
        "pressure_mean", "pressure_std", "pressure_cv",
        "rhythm_regularity", "rhythm_entropy",
        "tap_duration_mean", "tap_duration_std", "total_taps"
    ]


def generate_synthetic_tap_data(n_taps: int = 20, is_pd: bool = False,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic tap data for testing.
    PD taps: higher interval variance, irregular rhythm.
    """
    np.random.seed(seed)
    base_interval = 0.5  # seconds

    if is_pd:
        # Parkinson's: higher variability
        intervals = np.abs(np.random.normal(base_interval, 0.2, n_taps - 1))
        x_noise, y_noise = 20, 20
        pressure_std = 0.3
    else:
        # Healthy: regular intervals
        intervals = np.abs(np.random.normal(base_interval, 0.05, n_taps - 1))
        x_noise, y_noise = 5, 5
        pressure_std = 0.05

    timestamps = np.concatenate([[0], np.cumsum(intervals)])
    x = 100 + np.random.normal(0, x_noise, n_taps)
    y = 200 + np.random.normal(0, y_noise, n_taps)
    pressure = np.clip(np.random.normal(0.5, pressure_std, n_taps), 0.1, 1.0)
    duration = np.abs(np.random.normal(0.1, 0.02 if not is_pd else 0.05, n_taps))

    return pd.DataFrame({
        "timestamp": timestamps,
        "x": x,
        "y": y,
        "pressure": pressure,
        "duration": duration,
    })


if __name__ == "__main__":
    print("=== Synthetic Healthy Tap Data ===")
    df_healthy = generate_synthetic_tap_data(n_taps=20, is_pd=False)
    feats_healthy = extract_all_touch_features(df_healthy)
    print(f"Feature vector shape: {feats_healthy.shape}")

    print("\n=== Synthetic PD Tap Data ===")
    df_pd = generate_synthetic_tap_data(n_taps=20, is_pd=True)
    feats_pd = extract_all_touch_features(df_pd)

    names = get_touch_feature_names()
    print(f"\n{'Feature':<25} {'Healthy':>10} {'PD':>10}")
    print("-" * 48)
    for name, h, p in zip(names, feats_healthy, feats_pd):
        print(f"{name:<25} {h:>10.4f} {p:>10.4f}")
