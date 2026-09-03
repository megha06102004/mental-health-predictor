"""
Psychometric Predictor and Clinical Assessment Engine.
Integrates both Scikit-learn Ensemble Estimators and TensorFlow/Keras Multi-Task Deep Neural Network.
"""

import os
import pickle
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .data_loader import (
    DASS21_QUESTIONS,
    DEPRESSION_INDICES,
    ANXIETY_INDICES,
    STRESS_INDICES,
    score_to_severity,
    SEVERITY_LEVELS
)
from .visualizer import create_radar_chart

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class MindPulsePredictor:
    """Hybrid Scikit-learn & TensorFlow/Keras inference engine for DASS-21."""

    def __init__(self):
        # 1. Load Preprocessor
        prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Missing preprocessor at {prep_path}. Run train.py first.")

        with open(prep_path, "rb") as f:
            bundle = pickle.load(f)
            self.feat_eng = bundle["feature_engineer"]
            self.scaler = bundle["scaler"]
            self.feature_names = bundle["feature_names"]

        # 2. Load Scikit-learn Estimators
        self.sklearn_models = {}
        for condition in ["depression", "anxiety", "stress"]:
            path = os.path.join(MODELS_DIR, f"{condition}_model.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.sklearn_models[condition.capitalize()] = pickle.load(f)

        # 3. Load TensorFlow/Keras Multi-Task Deep Neural Network (if available)
        self.deep_model = None
        self.label_encoder = None
        keras_path = os.path.join(MODELS_DIR, "multitask_dass_model.keras")
        le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")

        if os.path.exists(keras_path) and os.path.exists(le_path):
            try:
                import tensorflow as tf
                from tensorflow import keras
                self.deep_model = keras.models.load_model(keras_path)
                with open(le_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
            except Exception as exc:
                print(f"[WARN] Could not load Keras Deep Learning model: {exc}")

    def predict(self, responses: List[int]) -> Dict[str, Any]:
        """
        Takes 21 integer responses (each 0 to 3) and outputs a clinical assessment profile.
        Computes both Scikit-learn ensemble and Keras Multi-Task DNN predictions.
        """
        if len(responses) != 21:
            raise ValueError(f"Expected 21 questionnaire items, got {len(responses)}")

        # Convert to single-row DataFrame
        cols = [f"Q{i+1}" for i in range(21)]
        df_row = pd.DataFrame([responses], columns=cols)

        # Calculate clinical raw sums (DASS-21 doubled = DASS-42 scale)
        dep_sum = int(sum(responses[i] for i in DEPRESSION_INDICES) * 2)
        anx_sum = int(sum(responses[i] for i in ANXIETY_INDICES) * 2)
        str_sum = int(sum(responses[i] for i in STRESS_INDICES) * 2)

        clinical_dep = score_to_severity(dep_sum, "Depression")
        clinical_anx = score_to_severity(anx_sum, "Anxiety")
        clinical_str = score_to_severity(str_sum, "Stress")

        # Transform features via Scikit-learn pipeline
        features_eng = self.feat_eng.transform(df_row)
        features_scaled = self.scaler.transform(features_eng)

        # Scikit-learn Inference
        ml_results = {}
        for condition, model in self.sklearn_models.items():
            pred_class = model.predict(features_scaled)[0]
            probs = model.predict_proba(features_scaled)[0]
            class_probs = {cls: round(float(p), 3) for cls, p in zip(model.classes_, probs)}

            ml_results[condition] = {
                "predicted_severity": pred_class,
                "probabilities": class_probs
            }

        # TensorFlow/Keras Multi-Task Deep Learning Inference
        deep_results = {}
        if self.deep_model is not None and self.label_encoder is not None:
            deep_preds = self.deep_model.predict(features_scaled, verbose=0)
            dep_probs = deep_preds["depression_output"][0]
            anx_probs = deep_preds["anxiety_output"][0]
            str_probs = deep_preds["stress_output"][0]

            dep_class_idx = int(np.argmax(dep_probs))
            anx_class_idx = int(np.argmax(anx_probs))
            str_class_idx = int(np.argmax(str_probs))

            deep_results = {
                "Depression": {
                    "predicted_severity": self.label_encoder.inverse_transform([dep_class_idx])[0],
                    "confidence": round(float(dep_probs[dep_class_idx]), 3)
                },
                "Anxiety": {
                    "predicted_severity": self.label_encoder.inverse_transform([anx_class_idx])[0],
                    "confidence": round(float(anx_probs[anx_class_idx]), 3)
                },
                "Stress": {
                    "predicted_severity": self.label_encoder.inverse_transform([str_class_idx])[0],
                    "confidence": round(float(str_probs[str_class_idx]), 3)
                }
            }

        # Generate Matplotlib Radar Chart
        radar_base64 = create_radar_chart(dep_sum, anx_sum, str_sum, as_base64=True)

        # Top Symptom Attribution (Explainable AI)
        symptoms_ranked = []
        for idx, val in enumerate(responses):
            if val >= 2:  # High response
                q_meta = DASS21_QUESTIONS[idx]
                symptoms_ranked.append({
                    "id": q_meta["id"],
                    "subscale": q_meta["subscale"],
                    "statement": q_meta["text"],
                    "score": val
                })
        symptoms_ranked.sort(key=lambda x: x["score"], reverse=True)

        # Personalized Evidence-Based Coping Strategies
        coping_tips = self._generate_recommendations(clinical_dep, clinical_anx, clinical_str)

        return {
            "scores": {
                "depression": dep_sum,
                "anxiety": anx_sum,
                "stress": str_sum,
            },
            "clinical_severity": {
                "depression": clinical_dep,
                "anxiety": clinical_anx,
                "stress": clinical_str,
            },
            "ml_predictions": ml_results,
            "deep_learning_predictions": deep_results,
            "radar_chart_base64": radar_base64,
            "top_symptoms": symptoms_ranked[:5],
            "coping_strategies": coping_tips,
            "clinical_disclaimer": (
                "MindPulse DASS-21 is an automated psychological screening instrument for research and educational purposes. "
                "It does not replace a formal clinical psychiatric diagnosis."
            )
        }

    def _generate_recommendations(self, dep_sev: str, anx_sev: str, str_sev: str) -> List[str]:
        tips = []
        if dep_sev in ("Severe", "Extremely Severe") or anx_sev in ("Severe", "Extremely Severe"):
            tips.append("Professional Consultation: Consider reaching out to a licensed psychologist or healthcare professional.")

        if anx_sev in ("Moderate", "Severe", "Extremely Severe"):
            tips.append("Physiological Calming: Practice 4-7-8 diaphragmatic breathing or progressive muscle relaxation to reduce autonomic arousal.")

        if dep_sev in ("Moderate", "Severe", "Extremely Severe"):
            tips.append("Behavioral Activation: Schedule small, achievable micro-goals daily to rebuild dopamine and enthusiasm pathways.")

        if str_sev in ("Moderate", "Severe", "Extremely Severe"):
            tips.append("Cognitive Decompression: Audit weekly commitments, prioritize sleep hygiene (7-8 hrs), and establish digital downtime boundaries.")

        if not tips:
            tips.append("Maintenance & Wellness: Your scores indicate healthy baseline levels. Maintain regular physical exercise, social connection, and restorative sleep.")

        return tips
