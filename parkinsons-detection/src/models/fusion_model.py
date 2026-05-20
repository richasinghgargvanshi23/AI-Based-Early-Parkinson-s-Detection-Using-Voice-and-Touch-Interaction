# src/models/fusion_model.py

"""
Score-Level Fusion for Parkinson's Detection
Combines voice and touch model probabilities into a final risk score.
"""

import numpy as np
from typing import Optional
from src.utils.config import (
    VOICE_WEIGHT, TOUCH_WEIGHT,
    LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD, RISK_LABELS
)


def fuse_scores(voice_prob: Optional[float],
                touch_prob: Optional[float]) -> dict:
    """
    Weighted average fusion of voice and touch probabilities.

    Args:
        voice_prob: PD probability from voice model (0–1), or None
        touch_prob: PD probability from touch model (0–1), or None

    Returns:
        dict with fused_score, risk_level, risk_message
    """
    if voice_prob is None and touch_prob is None:
        raise ValueError("At least one modality score must be provided.")

    if voice_prob is None:
        fused = float(touch_prob)
    elif touch_prob is None:
        fused = float(voice_prob)
    else:
        # Weighted fusion
        total_weight = VOICE_WEIGHT + TOUCH_WEIGHT
        fused = (VOICE_WEIGHT * voice_prob + TOUCH_WEIGHT * touch_prob) / total_weight

    fused = float(np.clip(fused, 0.0, 1.0))

    if fused < LOW_RISK_THRESHOLD:
        risk = "low"
    elif fused < HIGH_RISK_THRESHOLD:
        risk = "medium"
    else:
        risk = "high"

    return {
        "fused_score": round(fused, 4),
        "voice_score": round(voice_prob, 4) if voice_prob is not None else None,
        "touch_score": round(touch_prob, 4) if touch_prob is not None else None,
        "risk_level": risk,
        "risk_message": RISK_LABELS[risk],
        "modalities_used": (
            "voice+touch" if (voice_prob and touch_prob)
            else ("voice" if voice_prob else "touch")
        ),
    }


def confidence_score(voice_prob: Optional[float],
                     touch_prob: Optional[float]) -> float:
    """
    Estimate confidence based on agreement between modalities.
    High agreement → high confidence.
    """
    if voice_prob is None or touch_prob is None:
        return 0.6  # Single modality → moderate confidence

    agreement = 1.0 - abs(voice_prob - touch_prob)
    return round(float(agreement), 4)


def format_report(fusion_result: dict) -> str:
    """Format a human-readable report."""
    lines = [
        "=" * 55,
        "   PARKINSON'S SCREENING REPORT",
        "=" * 55,
        f"  Modalities Used   : {fusion_result['modalities_used']}",
        f"  Voice PD Score    : {fusion_result['voice_score']}",
        f"  Touch PD Score    : {fusion_result['touch_score']}",
        f"  Combined Score    : {fusion_result['fused_score']}",
        f"  Risk Level        : {fusion_result['risk_level'].upper()}",
        "",
        f"  {fusion_result['risk_message']}",
        "=" * 55,
        "",
        "  ⚠️  This is a research tool only.",
        "     NOT a substitute for medical diagnosis.",
        "=" * 55,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    # Example fusion scenarios
    scenarios = [
        (0.12, 0.18, "Both low → expected LOW"),
        (0.55, 0.70, "Both medium-high → expected HIGH"),
        (0.80, None, "Voice only (high) → HIGH"),
        (None, 0.25, "Touch only (low) → LOW"),
        (0.45, 0.50, "Borderline → MEDIUM"),
    ]

    for voice, touch, desc in scenarios:
        result = fuse_scores(voice, touch)
        conf = confidence_score(voice, touch)
        print(f"\n{desc}")
        print(f"  → Fused: {result['fused_score']:.4f}  |  "
              f"Risk: {result['risk_level'].upper()}  |  "
              f"Confidence: {conf:.4f}")
