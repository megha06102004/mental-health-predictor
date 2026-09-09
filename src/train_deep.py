"""
Training script for Multi-Task Deep Neural Network (TensorFlow / Keras).
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .data_loader import load_psychometric_dataset, generate_synthetic_dass_dataset, SEVERITY_LEVELS
from .deep_learning import build_multitask_dnn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_deep_model():
    print("=" * 65)
    print("  MINDPULSE: TRAINING MULTI-TASK DEEP NEURAL NETWORK")
    print("  Trained on Authentic OpenPsychometrics DASS Dataset")
    print("=" * 65)

    # Load preprocessor
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    with open(prep_path, "rb") as f:
        bundle = pickle.load(f)
        feat_eng = bundle["feature_engineer"]
        scaler = bundle["scaler"]

    # Load dataset
    df = load_psychometric_dataset()
    raw_feature_cols = [f"Q{i+1}" for i in range(21)]
    X_raw = df[raw_feature_cols]

    # Feature transformation
    X_eng = feat_eng.transform(X_raw)
    X_scaled = scaler.transform(X_eng)

    # Encode target labels (ordered clinically)
    label_encoder = LabelEncoder()
    label_encoder.fit(SEVERITY_LEVELS)

    y_dep = label_encoder.transform(df["Depression_Class"])
    y_anx = label_encoder.transform(df["Anxiety_Class"])
    y_str = label_encoder.transform(df["Stress_Class"])

    # Split
    indices = np.arange(len(df))
    idx_tr, idx_te = train_test_split(indices, test_size=0.20, random_state=42)

    X_tr, X_te = X_scaled[idx_tr], X_scaled[idx_te]
    y_dep_tr, y_dep_te = y_dep[idx_tr], y_dep[idx_te]
    y_anx_tr, y_anx_te = y_anx[idx_tr], y_anx[idx_te]
    y_str_tr, y_str_te = y_str[idx_tr], y_str[idx_te]

    # Build Keras Multi-Task Model
    model = build_multitask_dnn(input_dim=X_tr.shape[1], num_classes=len(SEVERITY_LEVELS))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    print("\nTraining Multi-Task Deep Neural Network for 20 epochs...")
    history = model.fit(
        X_tr,
        {
            "depression_output": y_dep_tr,
            "anxiety_output": y_anx_tr,
            "stress_output": y_str_tr
        },
        validation_data=(
            X_te,
            {
                "depression_output": y_dep_te,
                "anxiety_output": y_anx_te,
                "stress_output": y_str_te
            }
        ),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    eval_results = model.evaluate(
        X_te,
        {
            "depression_output": y_dep_te,
            "anxiety_output": y_anx_te,
            "stress_output": y_str_te
        },
        verbose=0
    )
    print("\n--- Deep Learning Evaluation Results ---")
    for metric_name, val in zip(model.metrics_names, eval_results):
        print(f"  {metric_name}: {val:.4f}")

    # Save Keras model and label mapping
    model_save_path = os.path.join(MODELS_DIR, "multitask_dass_model.keras")
    model.save(model_save_path)
    print(f"\n[INFO] Saved Multi-Task Deep Learning model to: {model_save_path}")

    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print("[INFO] Saved label encoder mapping.")


if __name__ == "__main__":
    train_deep_model()
