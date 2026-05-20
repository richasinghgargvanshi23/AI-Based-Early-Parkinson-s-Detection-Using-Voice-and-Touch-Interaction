# src/voice_analysis/voice_model.py

"""
CNN-LSTM Model for Parkinson's Voice Detection
Input: MFCC spectrogram (time x n_mfcc)
Output: Binary classification — Parkinson's (1) or Healthy (0)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def build_cnn_lstm_model(input_shape: tuple, dropout_rate: float = 0.3) -> keras.Model:
    """
    Build CNN-LSTM model.

    Args:
        input_shape: (time_steps, n_mfcc, 1) for CNN input
        dropout_rate: Dropout regularization rate

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape, name="mfcc_input")

    # ── CNN Block 1 ──────────────────────────────────────────────
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── CNN Block 2 ──────────────────────────────────────────────
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── CNN Block 3 ──────────────────────────────────────────────
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Reshape for LSTM ─────────────────────────────────────────
    # Merge spatial dims → time_steps x features
    shape = x.shape
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)

    # ── LSTM Block ───────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=dropout_rate)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=dropout_rate)
    )(x)

    # ── Classifier Head ──────────────────────────────────────────
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pd_probability")(x)

    model = keras.Model(inputs, outputs, name="PD_CNN_LSTM")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model


def build_simple_mlp(input_dim: int) -> keras.Model:
    """
    Lightweight MLP for flat feature vectors (MFCC stats + jitter/shimmer etc.)
    Faster to train, useful when spectrogram data is unavailable.
    """
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="PD_MLP")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model


def get_callbacks(model_save_path: str = "models/saved/voice_cnn_lstm.h5"):
    """Standard training callbacks."""
    return [
        EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True,
                      mode="max"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6),
        ModelCheckpoint(model_save_path, save_best_only=True,
                        monitor="val_auc", mode="max"),
    ]


def load_model(path: str) -> keras.Model:
    """Load a saved model."""
    return keras.models.load_model(path)


if __name__ == "__main__":
    # Quick architecture test
    model = build_cnn_lstm_model(input_shape=(128, 13, 1))
    model.summary()
    print("\nMLP variant:")
    mlp = build_simple_mlp(input_dim=33)
    mlp.summary()
