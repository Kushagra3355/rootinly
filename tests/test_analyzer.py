"""Unit tests for follicular density and scalp analyzer."""
import unittest
import numpy as np
from src.rootinly.core.analyzer import CrownAnalyzer
from src.rootinly.schemas.analysis import VisitMetrics

class TestCrownAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = CrownAnalyzer()

    def test_calculate_delta(self):
        # 20% increase
        self.assertEqual(self.analyzer.calculate_delta(12.0, 10.0), 20.0)
        # 50% decrease
        self.assertEqual(self.analyzer.calculate_delta(5.0, 10.0), -50.0)
        # zero baseline
        self.assertEqual(self.analyzer.calculate_delta(10.0, 0.0), 0.0)

    def test_compute_deltas(self):
        prev = VisitMetrics(
            hair_density_percent=60.0,
            scalp_visibility_percent=40.0,
            roi_total_pixels=1000,
            hair_pixels=600,
            scalp_pixels=400,
        )
        today = VisitMetrics(
            hair_density_percent=75.0,
            scalp_visibility_percent=25.0,
            roi_total_pixels=1000,
            hair_pixels=750,
            scalp_pixels=250,
        )
        deltas = self.analyzer.compute_deltas(today, prev)
        self.assertEqual(deltas.hair_density_change_pct, 25.0)
        self.assertEqual(deltas.scalp_visibility_change_pct, -37.5)

    def test_calculate_metrics_and_overlay(self):
        # Synthetic dark image (represents hair) with a binary mask
        synthetic_img = np.zeros((100, 100, 3), dtype=np.uint8)
        synthetic_mask = np.ones((100, 100), dtype=np.uint8)

        metrics, overlay = self.analyzer.calculate_metrics_and_overlay(synthetic_img, synthetic_mask)
        self.assertIsInstance(metrics, VisitMetrics)
        self.assertEqual(metrics.roi_total_pixels, 10000)
        self.assertEqual(overlay.shape, (100, 100, 3))
        self.assertGreaterEqual(metrics.hair_density_percent, 0.0)
        self.assertLessEqual(metrics.hair_density_percent, 100.0)

if __name__ == "__main__":
    unittest.main()
