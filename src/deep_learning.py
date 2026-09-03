"""
Multi-Task Deep Learning (MTL) Architecture using TensorFlow / Keras.
Shared latent representation with dedicated classification heads for Depression, Anxiety, and Stress.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import LabelEncoder

from .data_loader import SEVERITY_LEVELS

# Suppress verbose TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_multitask_dnn(input_dim: int = 31, num_classes: int = 5) -> keras.Model:
    """
    Constructs a Multi-Task Deep Neural Network in Keras.
    Shared representation captures common emotional distress features,
    while task-specific heads predict Depression, Anxiety, and Stress severity.
    """
    inputs = keras.Input(shape=(input_dim,), name="psychometric_features")

    # Shared Trunk (Representation Learning)
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    # Task-Specific Branch 1: Depression Head
    d_branch = layers.Dense(32, activation="relu", name="dep_dense")(x)
    d_branch = layers.Dropout(0.15)(d_branch)
    dep_out = layers.Dense(num_classes, activation="softmax", name="depression_output")(d_branch)

    # Task-Specific Branch 2: Anxiety Head
    a_branch = layers.Dense(32, activation="relu", name="anx_dense")(x)
    a_branch = layers.Dropout(0.15)(a_branch)
    anx_out = layers.Dense(num_classes, activation="softmax", name="anxiety_output")(a_branch)

    # Task-Specific Branch 3: Stress Head
    s_branch = layers.Dense(32, activation="relu", name="str_dense")(x)
    s_branch = layers.Dropout(0.15)(s_branch)
    str_out = layers.Dense(num_classes, activation="softmax", name="stress_output")(s_branch)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "depression_output": dep_out,
            "anxiety_output": anx_out,
            "stress_output": str_out
        },
        name="MindPulse_MultiTask_DNN"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "depression_output": "sparse_categorical_crossentropy",
            "anxiety_output": "sparse_categorical_crossentropy",
            "stress_output": "sparse_categorical_crossentropy"
        },
        metrics={
            "depression_output": "accuracy",
            "anxiety_output": "accuracy",
            "stress_output": "accuracy"
        }
    )

    return model
