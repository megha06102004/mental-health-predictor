"""
Clinical DASS-21 Instrument Definitions, Scoring Norms, and Dataset Loader.
Utilizes Pandas and NumPy for psychometric data representation.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

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

# Indices mapped to each psychological subscale (0-indexed)
DEPRESSION_INDICES = [2, 4, 9, 12, 15, 16, 20]   # Q3, Q5, Q10, Q13, Q16, Q17, Q21
ANXIETY_INDICES = [1, 3, 6, 8, 14, 18, 19]        # Q2, Q4, Q7, Q9, Q15, Q19, Q20
STRESS_INDICES = [0, 5, 7, 10, 11, 13, 17]        # Q1, Q6, Q8, Q11, Q12, Q14, Q18

SEVERITY_LEVELS = ["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"]

# Clinical cutoff ranges (based on DASS-21 score multiplied by 2 to align with DASS-42 norms)
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
    """Maps doubled DASS-21 subscale sum to clinical severity category."""
    thresholds = SEVERITY_THRESHOLDS[condition]
    for label, low, high in thresholds:
        if low <= score_dass42 <= high:
            return label
    return "Extremely Severe"


def generate_synthetic_dass_dataset(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    """
    Synthesizes a realistic psychometric DASS-21 dataset using correlated latent variable modeling.
    Simulates real-world comorbidities between depression, anxiety, and stress.
    """
    np.random.seed(random_state)

    # Correlated latent factors: general distress (g), depression (d), anxiety (a), stress (s)
    cov_matrix = [
        [1.0, 0.65, 0.55],   # Depression with Anxiety, Stress
        [0.65, 1.0, 0.70],   # Anxiety with Stress
        [0.55, 0.70, 1.0]    # Stress
    ]
    latents = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=n_samples)

    columns = [f"Q{i+1}" for i in range(21)]
    data = np.zeros((n_samples, 21), dtype=int)

    for i in range(n_samples):
        d_latent, a_latent, s_latent = latents[i]

        # Sample depression items
        for idx in DEPRESSION_INDICES:
            val = int(np.clip(np.round(1.2 + 0.8 * d_latent + np.random.normal(0, 0.5)), 0, 3))
            data[i, idx] = val

        # Sample anxiety items
        for idx in ANXIETY_INDICES:
            val = int(np.clip(np.round(1.0 + 0.8 * a_latent + np.random.normal(0, 0.5)), 0, 3))
            data[i, idx] = val

        # Sample stress items
        for idx in STRESS_INDICES:
            val = int(np.clip(np.round(1.3 + 0.8 * s_latent + np.random.normal(0, 0.5)), 0, 3))
            data[i, idx] = val

    df = pd.DataFrame(data, columns=columns)

    # Compute ground truth subscale clinical sums (multiplied by 2 per DASS-21 standard)
    df["dep_sum"] = df[[f"Q{i+1}" for i in DEPRESSION_INDICES]].sum(axis=1) * 2
    df["anx_sum"] = df[[f"Q{i+1}" for i in ANXIETY_INDICES]].sum(axis=1) * 2
    df["str_sum"] = df[[f"Q{i+1}" for i in STRESS_INDICES]].sum(axis=1) * 2

    # Map to clinical classes
    df["Depression_Class"] = df["dep_sum"].apply(lambda s: score_to_severity(s, "Depression"))
    df["Anxiety_Class"] = df["anx_sum"].apply(lambda s: score_to_severity(s, "Anxiety"))
    df["Stress_Class"] = df["str_sum"].apply(lambda s: score_to_severity(s, "Stress"))

    return df
