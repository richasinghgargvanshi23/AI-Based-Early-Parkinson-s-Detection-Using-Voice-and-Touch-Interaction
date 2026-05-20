# src/models/train.py

"""
Training Script for Both Voice and Touch Models
Usage:
    python -m src.models.train --mode voice
    python -m src.models.train --mode touch
    python -m src.models.train --mode all
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# Voice Training
# ─────────────────────────────────────────────────────────────
def train_voice_model_uci():
    """
    Train voice MLP model on UCI Parkinson's dataset.
    Download: https://archive.ics.uci.edu/ml/datasets/parkinsons
    Expected file: data/processed/parkinsons.csv
    """
    from src.voice_analysis.voice_model import build_simple_mlp, get_callbacks

    data_path = "data/processed/parkinsons.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("   Download from: https://archive.ics.uci.edu/ml/datasets/parkinsons")
        print("   Generating synthetic data for demo...")
        X, y = _generate_synthetic_voice_data(300)
    else:
        df = pd.read_csv(data_path)
        # UCI dataset: drop name col, 'status' is label (1=PD, 0=healthy)
        y = df["status"].values
        X = df.drop(columns=["name", "status"], errors="ignore").values
        print(f"✅ Loaded UCI data: {X.shape[0]} samples, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    import joblib
    os.makedirs("models/saved", exist_ok=True)
    joblib.dump(scaler, "models/saved/voice_scaler.pkl")

    model = build_simple_mlp(input_dim=X_train.shape[1])
    model.summary()

    callbacks = get_callbacks("models/saved/voice_cnn_lstm.h5")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Final eval
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)[:3]
    print(f"\n✅ Voice Model — Test Accuracy: {acc:.4f} | AUC: {auc:.4f}")
    return model, history


def _generate_synthetic_voice_data(n: int = 300):
    """Generate synthetic voice features for demo when UCI data isn't available."""
    np.random.seed(42)
    X_healthy = np.random.randn(n // 2, 22) * 0.5
    X_pd = np.random.randn(n // 2, 22) * 1.2 + 0.8
    X = np.vstack([X_healthy, X_pd])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


# ─────────────────────────────────────────────────────────────
# Touch Training
# ─────────────────────────────────────────────────────────────
def train_touch_model_data():
    """
    Train touch model.
    Expects: data/processed/touch_data.csv with columns including 'label'.
    Falls back to synthetic data if file not found.
    """
    from src.touch_analysis.touch_features import (
        load_tap_data, extract_all_touch_features, generate_synthetic_tap_data
    )
    from src.touch_analysis.touch_model import train_touch_model, evaluate_touch_model

    data_path = "data/processed/touch_data.csv"
    if os.path.exists(data_path):
        df_all = pd.read_csv(data_path)
        # Expects: session_id, label, tap event columns
        print(f"✅ Loaded touch data: {len(df_all)} records")
        # Group by session, extract features per session
        samples, labels = [], []
        for session_id, group in df_all.groupby("session_id"):
            feats = extract_all_touch_features(group)
            samples.append(feats)
            labels.append(int(group["label"].iloc[0]))
        X = np.array(samples)
        y = np.array(labels)
    else:
        print("⚠️  Touch dataset not found. Using synthetic data.")
        samples, labels = [], []
        for i in range(200):
            is_pd = i >= 100
            df = generate_synthetic_tap_data(n_taps=30, is_pd=is_pd, seed=i)
            feats = extract_all_touch_features(df)
            samples.append(feats)
            labels.append(1 if is_pd else 0)
        X = np.array(samples)
        y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train_touch_model(X_train, y_train)
    metrics = evaluate_touch_model(model, X_test, y_test)

    print(f"\n✅ Touch Model — Test Accuracy: {metrics['accuracy']:.4f} "
          f"| AUC: {metrics['auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])
    return model


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parkinson's Detection Models")
    parser.add_argument("--mode", choices=["voice", "touch", "all"],
                        default="all", help="Which model to train")
    args = parser.parse_args()

    os.makedirs("models/saved", exist_ok=True)

    if args.mode in ("voice", "all"):
        print("\n" + "━" * 50)
        print("  Training VOICE model")
        print("━" * 50)
        train_voice_model_uci()

    if args.mode in ("touch", "all"):
        print("\n" + "━" * 50)
        print("  Training TOUCH model")
        print("━" * 50)
        train_touch_model_data()

    print("\n✅ Training complete. Models saved in models/saved/")
