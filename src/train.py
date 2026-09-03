"""
Model Training and Hyperparameter Tuning Script using Scikit-learn, Pandas, and NumPy.
Trains multi-target classifiers for Depression, Anxiety, and Stress with cross-validation.
"""

import os
import pickle
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_loader import generate_synthetic_dass_dataset, SEVERITY_LEVELS
from .preprocessing import DASSFeatureEngineer, create_feature_pipeline
from .visualizer import plot_feature_importance


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def train_models():
    print("=" * 65)
    print("  MINDPULSE: TRAINING SUPERVISED PSYCHOMETRIC MODELS")
    print("  Using Scikit-learn, Pandas, NumPy, and Matplotlib")
    print("=" * 65)

    # 1. Load / Synthesize Dataset
    print("[1/5] Generating psychometric dataset (N=3,000 samples)...")
    df = generate_synthetic_dass_dataset(n_samples=3000, random_state=42)
    raw_feature_cols = [f"Q{i+1}" for i in range(21)]
    X_raw = df[raw_feature_cols]

    y_dep = df["Depression_Class"]
    y_anx = df["Anxiety_Class"]
    y_str = df["Stress_Class"]

    # 2. Train/Test Split
    print("[2/5] Splitting data into 80% Train, 20% Test...")
    X_train_raw, X_test_raw, y_dep_tr, y_dep_te = train_test_split(
        X_raw, y_dep, test_size=0.20, random_state=42, stratify=y_dep
    )
    _, _, y_anx_tr, y_anx_te = train_test_split(
        X_raw, y_anx, test_size=0.20, random_state=42, stratify=y_anx
    )
    _, _, y_str_tr, y_str_te = train_test_split(
        X_raw, y_str, test_size=0.20, random_state=42, stratify=y_str
    )

    # 3. Fit Preprocessing Pipeline
    print("[3/5] Fitting Feature Engineering and Normalization Pipeline...")
    feat_eng = DASSFeatureEngineer()
    scaler = StandardScaler()

    X_train_eng = feat_eng.fit_transform(X_train_raw)
    X_train_scaled = scaler.fit_transform(X_train_eng)

    X_test_eng = feat_eng.transform(X_test_raw)
    X_test_scaled = scaler.transform(X_test_eng)

    feature_names = feat_eng.get_feature_names_out()

    # 4. Compare Classifiers on Depression
    print("[4/5] Benchmarking Scikit-learn Estimators on Depression subscale...")
    candidates = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42)
    }

    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_dep_tr)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_dep_te, preds)
        f1 = f1_score(y_dep_te, preds, average="weighted")
        print(f"  -> {name:20s} | Test Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")

    # 5. Fine-Tune with GridSearchCV for All 3 Subscales
    print("\n[5/5] Performing GridSearchCV Hyperparameter Optimization for Final Models...")
    param_grid = {
        "n_estimators": [80, 120],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5]
    }

    targets = [
        ("Depression", y_dep_tr, y_dep_te, "depression_model.pkl"),
        ("Anxiety", y_anx_tr, y_anx_te, "anxiety_model.pkl"),
        ("Stress", y_str_tr, y_str_te, "stress_model.pkl"),
    ]

    trained_models = {}

    for label, y_tr, y_te, filename in targets:
        print(f"\n--- Tuning {label} Model ---")
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_tr)
        best_model = grid.best_estimator_
        print(f"  Best Params: {grid.best_params_}")

        test_preds = best_model.predict(X_test_scaled)
        acc = accuracy_score(y_te, test_preds)
        print(f"  {label} Final Test Accuracy: {acc*100:.2f}%")
        print(f"  Classification Report for {label}:")
        print(classification_report(y_te, test_preds, zero_division=0))

        # Save model
        out_path = os.path.join(MODELS_DIR, filename)
        with open(out_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"  Saved model to: {out_path}")

        trained_models[label] = best_model

    # Save Preprocessing Pipeline
    prep_bundle = {
        "feature_engineer": feat_eng,
        "scaler": scaler,
        "feature_names": feature_names
    }
    with open(os.path.join(MODELS_DIR, "preprocessor.pkl"), "wb") as f:
        pickle.dump(prep_bundle, f)
    print("\nSaved preprocessor to models/preprocessor.pkl")

    # Generate Feature Importance Plot using Matplotlib
    dep_clf = trained_models["Depression"]
    plot_feature_importance(
        dep_clf.feature_importances_,
        feature_names,
        title="Predictive Importance of Psychological Features (Depression)",
        save_path=os.path.join(MODELS_DIR, "feature_importance_depression.png")
    )
    print("Saved feature importance chart using Matplotlib.")
    print("\nTraining complete! All models ready.")


if __name__ == "__main__":
    train_models()
