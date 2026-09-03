"""
Feature Engineering and Normalization Pipeline using Pandas, NumPy, and Scikit-learn.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from .data_loader import DEPRESSION_INDICES, ANXIETY_INDICES, STRESS_INDICES


class DASSFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that enriches raw 21 DASS items
    with clinically meaningful aggregate, interaction, and sub-cluster indices.
    """

    def __init__(self):
        self.feature_names: List[str] = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            cols = [f"Q{i+1}" for i in range(21)]
            df = pd.DataFrame(X, columns=cols)

        # 1. Base item scores (21 features)
        features = df[[f"Q{i+1}" for i in range(21)]].copy()

        # 2. Subscale Raw Sums
        features["dep_sum"] = df[[f"Q{i+1}" for i in DEPRESSION_INDICES]].sum(axis=1)
        features["anx_sum"] = df[[f"Q{i+1}" for i in ANXIETY_INDICES]].sum(axis=1)
        features["str_sum"] = df[[f"Q{i+1}" for i in STRESS_INDICES]].sum(axis=1)

        # 3. Total Global Emotional Distress Score
        features["total_distress"] = features["dep_sum"] + features["anx_sum"] + features["str_sum"]

        # 4. Clinical Symptom Sub-clusters
        # Autonomic arousal (Heart rate, dry mouth, breathing: Q2, Q4, Q19)
        features["autonomic_arousal"] = df[["Q2", "Q4", "Q19"]].sum(axis=1)
        # Anhedonia / Dysphoria (Inability to feel joy, loss of initiative: Q3, Q5, Q16)
        features["anhedonia"] = df[["Q3", "Q5", "Q16"]].sum(axis=1)
        # Agitation & Restlessness (Over-reacting, nervous energy, difficulty relaxing: Q1, Q6, Q8, Q12)
        features["agitation"] = df[["Q1", "Q6", "Q8", "Q12"]].sum(axis=1)

        # 5. Ratios & Interactions (with safe epsilon)
        eps = 1e-4
        features["dep_ratio"] = features["dep_sum"] / (features["total_distress"] + eps)
        features["anx_ratio"] = features["anx_sum"] / (features["total_distress"] + eps)
        features["str_ratio"] = features["str_sum"] / (features["total_distress"] + eps)

        self.feature_names = list(features.columns)
        return features.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


def create_feature_pipeline():
    """Returns a full Scikit-learn preprocessing pipeline."""
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("feature_engineering", DASSFeatureEngineer()),
        ("scaler", StandardScaler())
    ])
