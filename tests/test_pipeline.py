"""
Юнит-тесты для проверки компонентов системы PuchinaAI.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_generator import generate_weather_and_heave_data
from src.features import compute_physical_features, compute_zero_crossings, extract_sequential_windows
from src.models.hybrid_nn import build_hybrid_cnn_lstm


class TestPuchinaPipeline(unittest.TestCase):
    def setUp(self):
        self.df = generate_weather_and_heave_data(
            start_date="2022-01-01",
            end_date="2023-12-31",
            random_seed=123,
        )

    def test_zero_crossings(self):
        temps = np.array([-2.0, 1.0, -1.0, -3.0, 2.0, 4.0])
        crossings = compute_zero_crossings(temps)
        # Transitions at indices: 1 (-2 to 1), 2 (1 to -1), 4 (-3 to 2)
        self.assertEqual(list(crossings), [0, 1, 1, 0, 1, 0])

    def test_feature_engineering(self):
        feat_df = compute_physical_features(self.df, window_days=30)
        self.assertIn("f_cross", feat_df.columns)
        self.assertIn("t_sum", feat_df.columns)
        self.assertIn("t_avg", feat_df.columns)
        self.assertIn("f_cross_decade", feat_df.columns)
        self.assertIn("cumulative_cycles", feat_df.columns)
        self.assertFalse(feat_df["f_cross"].isna().any())

    def test_sequence_extraction(self):
        seq_len = 45
        X_seq, indices = extract_sequential_windows(self.df, seq_length=seq_len)
        self.assertEqual(len(X_seq.shape), 3)
        self.assertEqual(X_seq.shape[1], seq_len)
        self.assertEqual(X_seq.shape[2], 8)
        self.assertEqual(len(X_seq), len(self.df) - seq_len + 1)

    def test_hybrid_nn_architecture(self):
        model, att_model = build_hybrid_cnn_lstm(seq_length=45, num_features=8, num_classes=3)
        dummy_input = np.zeros((4, 45, 8), dtype=np.float32)
        out = model(dummy_input)
        self.assertEqual(out.shape, (4, 3))
        # Softmax outputs sum to 1
        sums = np.sum(out.numpy(), axis=1)
        np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)

        # Attention weights shape: (4, 45, 1) and sum to 1
        att_w = att_model(dummy_input)
        self.assertEqual(att_w.shape, (4, 45, 1))
        att_sums = np.sum(att_w.numpy(), axis=1)
        np.testing.assert_allclose(att_sums, np.ones((4, 1)), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
