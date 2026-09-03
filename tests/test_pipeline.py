"""
Unit and Integration Tests for Data Preprocessing, ML Estimators, and Visualizations.
"""

import unittest
import numpy as np
import pandas as pd

from src.data_loader import generate_synthetic_dass_dataset, score_to_severity
from src.preprocessing import DASSFeatureEngineer
from src.visualizer import create_radar_chart
from src.predictor import MindPulsePredictor


class TestMindPulsePipeline(unittest.TestCase):

    def test_dataset_generation(self):
        df = generate_synthetic_dass_dataset(n_samples=50, random_state=42)
        self.assertEqual(len(df), 50)
        self.assertTrue("Depression_Class" in df.columns)
        self.assertTrue("Anxiety_Class" in df.columns)
        self.assertTrue("Stress_Class" in df.columns)

    def test_score_to_severity(self):
        self.assertEqual(score_to_severity(4, "Depression"), "Normal")
        self.assertEqual(score_to_severity(12, "Depression"), "Mild")
        self.assertEqual(score_to_severity(16, "Depression"), "Moderate")
        self.assertEqual(score_to_severity(24, "Depression"), "Severe")
        self.assertEqual(score_to_severity(32, "Depression"), "Extremely Severe")

    def test_feature_engineering_dimensions(self):
        feat_eng = DASSFeatureEngineer()
        dummy_input = pd.DataFrame(np.zeros((10, 21)), columns=[f"Q{i+1}" for i in range(21)])
        out = feat_eng.fit_transform(dummy_input)
        # Should have 21 base + 3 subscale sums + 1 total + 3 clusters + 3 ratios = 31 features
        self.assertEqual(out.shape, (10, 31))

    def test_radar_chart_generation(self):
        chart_base64 = create_radar_chart(10.0, 15.0, 20.0, as_base64=True)
        self.assertTrue(chart_base64.startswith("data:image/png;base64,"))

    def test_full_predictor_flow(self):
        predictor = MindPulsePredictor()

        # Test normal response (all zeros)
        normal_responses = [0] * 21
        res_normal = predictor.predict(normal_responses)
        self.assertEqual(res_normal["clinical_severity"]["depression"], "Normal")
        self.assertEqual(res_normal["scores"]["depression"], 0)

        # Test severe response (all threes)
        severe_responses = [3] * 21
        res_severe = predictor.predict(severe_responses)
        self.assertEqual(res_severe["clinical_severity"]["depression"], "Extremely Severe")
        self.assertEqual(res_severe["scores"]["depression"], 42)
        self.assertTrue(len(res_severe["top_symptoms"]) > 0)


if __name__ == "__main__":
    unittest.main()
