# src/touch_analysis/touch_model.py

"""
Touch Pattern Classifier for Parkinson's Detection
Uses Random Forest + SVM ensemble on tap interaction features.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)


def build_touch_model() -> Pipeline:
    """
    Build a voting ensemble: Random Forest + SVM.
    Wrapped in a sklearn Pipeline with StandardScaler.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        voting="soft",
        weights=[0.6, 0.4],
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", ensemble),
    ])
    return pipeline


def train_touch_model(X_train: np.ndarray, y_train: np.ndarray,
                       save_path: str = "models/saved/touch_rf.pkl") -> Pipeline:
    """
    Train and save the touch model.

    Args:
        X_train: Feature matrix (n_samples, n_features)
        y_train: Labels (0=healthy, 1=PD)
        save_path: Where to save the trained model

    Returns:
        Fitted Pipeline
    """
    print(f"Training touch model on {X_train.shape[0]} samples...")

    model = build_touch_model()

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
    print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"✅ Model saved to {save_path}")
    return model


def evaluate_touch_model(model: Pipeline, X_test: np.ndarray,
                          y_test: np.ndarray) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred,
                                         target_names=["Healthy", "Parkinson's"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def load_touch_model(path: str) -> Pipeline:
    """Load a saved touch model."""
    return joblib.load(path)


def predict_touch(features: np.ndarray, model: Pipeline = None,
                  model_path: str = "models/saved/touch_rf.pkl") -> dict:
    """
    Predict PD probability from touch features.

    Returns:
        dict with probability and metadata
    """
    if model is None:
        if os.path.exists(model_path):
            model = load_touch_model(model_path)
        else:
            # Heuristic fallback
            return _heuristic_touch_prediction(features)

    prob = float(model.predict_proba(features[np.newaxis, :])[0, 1])
    return {"touch_pd_probability": round(prob, 4)}


def _heuristic_touch_prediction(features: np.ndarray) -> dict:
    """
    Rule-based fallback when no trained model exists.
    Feature indices match get_touch_feature_names() order.
    """
    iti_cv = features[2] if len(features) > 2 else 0
    spatial_scatter = features[14] if len(features) > 14 else 0

    score = 0.0
    if iti_cv > 0.3:       # High interval variability
        score += 0.4
    if spatial_scatter > 15:  # High spatial scatter
        score += 0.3

    prob = min(score, 0.9)
    return {"touch_pd_probability": round(prob, 4)}


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    from src.touch_analysis.touch_features import (
        generate_synthetic_tap_data, extract_all_touch_features
    )
    from sklearn.model_selection import train_test_split

    print("Generating synthetic dataset...")
    samples = []
    labels = []
    for i in range(100):
        is_pd = i >= 50
        df = generate_synthetic_tap_data(n_taps=25, is_pd=is_pd, seed=i)
        feats = extract_all_touch_features(df)
        samples.append(feats)
        labels.append(1 if is_pd else 0)

    X = np.array(samples)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_touch_model(X_train, y_train,
                               save_path="models/saved/touch_rf.pkl")
    metrics = evaluate_touch_model(model, X_test, y_test)

    print(f"\nTest Accuracy : {metrics['accuracy']:.4f}")
    print(f"Test AUC      : {metrics['auc']:.4f}")
    print("\nClassification Report:")
    print(metrics["report"])
