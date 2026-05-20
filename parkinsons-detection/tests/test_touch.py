# tests/test_touch.py

"""Unit tests for touch analysis pipeline."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.touch_analysis.touch_features import (
    generate_synthetic_tap_data,
    extract_all_touch_features,
    compute_inter_tap_intervals,
    get_touch_feature_names,
)


class TestTouchFeatures(unittest.TestCase):

    def setUp(self):
        self.df_healthy = generate_synthetic_tap_data(n_taps=20, is_pd=False)
        self.df_pd = generate_synthetic_tap_data(n_taps=20, is_pd=True)

    def test_feature_vector_shape(self):
        feats = extract_all_touch_features(self.df_healthy)
        expected_len = len(get_touch_feature_names())
        self.assertEqual(len(feats), expected_len)

    def test_no_nan_features(self):
        feats = extract_all_touch_features(self.df_pd)
        self.assertFalse(np.any(np.isnan(feats)))

    def test_pd_higher_variability(self):
        """PD data should have higher inter-tap interval CoV."""
        feats_healthy = extract_all_touch_features(self.df_healthy)
        feats_pd = extract_all_touch_features(self.df_pd)
        # CoV is index 2
        self.assertGreater(feats_pd[2], feats_healthy[2])

    def test_inter_tap_intervals_length(self):
        timestamps = self.df_healthy["timestamp"].values
        intervals = compute_inter_tap_intervals(timestamps)
        self.assertEqual(len(intervals), len(timestamps) - 1)

    def test_intervals_positive(self):
        timestamps = self.df_healthy["timestamp"].values
        intervals = compute_inter_tap_intervals(timestamps)
        self.assertTrue(np.all(intervals >= 0))

    def test_feature_names_match_vector(self):
        names = get_touch_feature_names()
        feats = extract_all_touch_features(self.df_healthy)
        self.assertEqual(len(names), len(feats))

    def test_total_taps_feature(self):
        feats = extract_all_touch_features(self.df_healthy)
        # Last feature is total_taps
        self.assertEqual(feats[-1], 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
