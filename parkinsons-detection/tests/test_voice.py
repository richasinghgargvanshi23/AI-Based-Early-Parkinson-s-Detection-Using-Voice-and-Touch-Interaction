# tests/test_voice.py

"""Unit tests for voice analysis pipeline."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.voice_analysis.feature_extractor import (
    extract_mfcc, extract_shimmer, extract_zcr, get_feature_names
)
from src.models.fusion_model import fuse_scores


class TestVoiceFeatures(unittest.TestCase):

    def setUp(self):
        """Generate a synthetic audio signal for testing."""
        sr = 22050
        t = np.linspace(0, 3, sr * 3)
        # 440 Hz sine wave + noise (simulates sustained vowel)
        self.y = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
        self.sr = sr

    def test_mfcc_shape(self):
        feats = extract_mfcc(self.y, self.sr)
        # Expect N_MFCC * 2 = 26 features (mean + std)
        self.assertEqual(feats.shape[0], 26)

    def test_mfcc_no_nan(self):
        feats = extract_mfcc(self.y, self.sr)
        self.assertFalse(np.any(np.isnan(feats)))

    def test_shimmer_range(self):
        shimmer = extract_shimmer(self.y, self.sr)
        self.assertGreaterEqual(shimmer, 0.0)
        self.assertLess(shimmer, 10.0)

    def test_zcr_positive(self):
        zcr = extract_zcr(self.y)
        self.assertGreater(zcr, 0.0)

    def test_feature_names_count(self):
        names = get_feature_names()
        self.assertEqual(len(names), 33)


class TestFusionModel(unittest.TestCase):

    def test_fusion_both_low(self):
        result = fuse_scores(0.1, 0.15)
        self.assertEqual(result["risk_level"], "low")

    def test_fusion_both_high(self):
        result = fuse_scores(0.8, 0.75)
        self.assertEqual(result["risk_level"], "high")

    def test_fusion_voice_only(self):
        result = fuse_scores(0.5, None)
        self.assertIsNotNone(result["fused_score"])
        self.assertEqual(result["modalities_used"], "voice")

    def test_fusion_touch_only(self):
        result = fuse_scores(None, 0.2)
        self.assertEqual(result["risk_level"], "low")

    def test_fusion_raises_on_both_none(self):
        with self.assertRaises(ValueError):
            fuse_scores(None, None)

    def test_fused_score_in_range(self):
        result = fuse_scores(0.4, 0.6)
        self.assertGreaterEqual(result["fused_score"], 0.0)
        self.assertLessEqual(result["fused_score"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
