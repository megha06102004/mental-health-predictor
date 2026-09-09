"""
Clinical DASS-21 Instrument Definitions, Scoring Norms, and Dataset Loader.
Loads authentic human psychometric data from the OpenPsychometrics DASS dataset.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REAL_DATA_PATH = os.path.join(BASE_DIR, "data", "real_dass_data.csv")

# The 21 standard validated clinical questions of DASS-21
DASS21_QUESTIONS = [
    {"id": "Q1", "subscale": "Stress", "text": "I found it hard to wind down."},
    {"id": "Q2", "subscale": "Anxiety", "text": "I was aware of dryness of my mouth."},
    {"id": "Q3", "subscale": "Depression", "text": "I couldn't seem to experience any positive feeling at all."},
    {"id": "Q4", "subscale": "Anxiety", "text": "I experienced breathing difficulty (e.g. excessively rapid breathing)."},
    {"id": "Q5", "subscale": "Depression", "text": "I found it difficult to work up the initiative to do things."},
    {"id": "Q6", "subscale": "Stress", "text": "I tended to over-react to situations."},
    {"id": "Q7", "subscale": "Anxiety", "text": "I experienced trembling (e.g. in the hands)."},
    {"id": "Q8", "subscale": "Stress", "text": "I felt that I was using a lot of nervous energy."},
    {"id": "Q9", "subscale": "Anxiety", "text": "I was worried about situations in which I might panic and make a fool of myself."},
    {"id": "Q10", "subscale": "Depression", "text": "I felt that I had nothing to look forward to."},
    {"id": "Q11", "subscale": "Stress", "text": "I found myself getting agitated."},
    {"id": "Q12", "subscale": "Stress", "text": "I found it difficult to relax."},
    {"id": "Q13", "subscale": "Depression", "text": "I felt down-hearted and blue."},
    {"id": "Q14", "subscale": "Stress", "text": "I was intolerant of anything that kept me from getting on with what I was doing."},
    {"id": "Q15", "subscale": "Anxiety", "text": "I felt I was close to panic."},
    {"id": "Q16", "subscale": "Depression", "text": "I was unable to become enthusiastic about anything."},
    {"id": "Q17", "subscale": "Depression", "text": "I felt I wasn't worth much as a person."},
    {"id": "Q18", "subscale": "Stress", "text": "I felt that I was rather touchy."},
    {"id": "Q19", "subscale": "Anxiety", "text": "I was aware of the action of my heart in the absence of physical exertion."},
    {"id": "Q20", "subscale": "Anxiety", "text": "I felt scared without any good reason."},
    {"id": "Q21", "subscale": "Depression", "text": "I felt that life was meaningless."}
]

DEPRESSION_INDICES = [2, 4, 9, 12, 15, 16, 20]
ANXIETY_INDICES = [1, 3, 6, 8, 14, 18, 19]
STRESS_INDICES = [0, 5, 7, 10, 11, 13, 17]

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

SEVERITY_THRESHOLDS = {
    "Depression": [
        ("Normal", 0, 9),
        ("Mild", 10, 13),
        ("Moderate", 14, 20),
        ("Severe", 21, 27),
        ("Extremely Severe", 28, 999),
    ],
    "Anxiety": [
        ("Normal", 0, 7),
        ("Mild", 8, 9),
        ("Moderate", 10, 14),
        ("Severe", 15, 19),
        ("Extremely Severe", 20, 999),
    ],
    "Stress": [
        ("Normal", 0, 14),
        ("Mild", 15, 18),
        ("Moderate", 19, 25),
        ("Severe", 26, 33),
        ("Extremely Severe", 34, 999),
    ]
}


def score_to_severity(score_dass42: float, condition: str) -> str:
    thresholds = SEVERITY_THRESHOLDS[condition]
    for label, low, high in thresholds:
        if low <= score_dass42 <= high:
            return label
    return "Extremely Severe"


def load_psychometric_dataset(csv_path: str = DEFAULT_REAL_DATA_PATH) -> pd.DataFrame:
    """
    Loads the authentic OpenPsychometrics DASS dataset.
    If the CSV is present, returns authentic human responses.
    Otherwise falls back to empirical distribution synthesis.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(df):,} authentic human responses from OpenPsychometrics dataset.")
        return df

    print(f"[WARN] {csv_path} not found. Synthesizing empirical dataset.")
    return generate_synthetic_dass_dataset()


def generate_synthetic_dass_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    cov_matrix = [
        [1.0, 0.65, 0.55],
        [0.65, 1.0, 0.70],
        [0.55, 0.70, 1.0]
    ]
    latents = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=n_samples)

    columns = [f"Q{i+1}" for i in range(21)]
    data = np.zeros((n_samples, 21), dtype=int)

    for i in range(n_samples):
        d_latent, a_latent, s_latent = latents[i]
        for idx in DEPRESSION_INDICES:
            data[i, idx] = int(np.clip(np.round(1.2 + 0.8 * d_latent + np.random.normal(0, 0.5)), 0, 3))
        for idx in ANXIETY_INDICES:
            data[i, idx] = int(np.clip(np.round(1.0 + 0.8 * a_latent + np.random.normal(0, 0.5)), 0, 3))
        for idx in STRESS_INDICES:
            data[i, idx] = int(np.clip(np.round(1.3 + 0.8 * s_latent + np.random.normal(0, 0.5)), 0, 3))

    df = pd.DataFrame(data, columns=columns)
    df["dep_sum"] = df[[f"Q{i+1}" for i in DEPRESSION_INDICES]].sum(axis=1) * 2
    df["anx_sum"] = df[[f"Q{i+1}" for i in ANXIETY_INDICES]].sum(axis=1) * 2
    df["str_sum"] = df[[f"Q{i+1}" for i in STRESS_INDICES]].sum(axis=1) * 2

    df["Depression_Class"] = df["dep_sum"].apply(lambda s: score_to_severity(s, "Depression"))
    df["Anxiety_Class"] = df["anx_sum"].apply(lambda s: score_to_severity(s, "Anxiety"))
    df["Stress_Class"] = df["str_sum"].apply(lambda s: score_to_severity(s, "Stress"))

    return df
