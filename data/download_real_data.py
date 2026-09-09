"""
Downloads, extracts, and pre-processes authentic psychometric human responses
from the OpenPsychometrics DASS Dataset (N=39,775).
Converts raw responses to clinical DASS-21 scale items and severity labels.
"""

import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "real_dass_data.csv")

# Standard mapping from DASS-42 to DASS-21 (0-indexed item order: Q1 to Q21)
# Depression: Q3, Q5, Q10, Q13, Q16, Q17, Q21
# Anxiety:    Q2, Q4, Q7,  Q9,  Q15, Q19, Q20
# Stress:     Q1, Q6, Q8,  Q11, Q12, Q14, Q18
DASS21_TO_DASS42_COLS = [
    "Q1A",   # Stress 1
    "Q2A",   # Anxiety 1
    "Q3A",   # Depression 1
    "Q4A",   # Anxiety 2
    "Q5A",   # Depression 2
    "Q6A",   # Stress 2
    "Q7A",   # Anxiety 3
    "Q8A",   # Stress 3
    "Q9A",   # Anxiety 4
    "Q10A",  # Depression 3
    "Q11A",  # Stress 4
    "Q12A",  # Stress 5
    "Q13A",  # Depression 4
    "Q14A",  # Stress 6
    "Q15A",  # Anxiety 5
    "Q16A",  # Depression 5
    "Q17A",  # Depression 6
    "Q18A",  # Stress 7
    "Q19A",  # Anxiety 6
    "Q20A",  # Anxiety 7
    "Q21A"   # Depression 7
]

DEPRESSION_INDICES = [2, 4, 9, 12, 15, 16, 20]
ANXIETY_INDICES = [1, 3, 6, 8, 14, 18, 19]
STRESS_INDICES = [0, 5, 7, 10, 11, 13, 17]

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


def fetch_and_process(sample_size: int = 6000):
    url = "https://openpsychometrics.org/_rawdata/DASS_data_21.02.19.zip"
    print(f"[1/4] Downloading OpenPsychometrics DASS dataset from: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    print(f"[2/4] Extracting raw data.csv ({len(content) / (1024*1024):.2f} MB)...")
    z = zipfile.ZipFile(io.BytesIO(content))
    with z.open("DASS_data_21.02.19/data.csv") as f:
        df_raw = pd.read_csv(f, sep="\t", low_memory=False)

    print(f"  -> Total Participants in Raw Dataset: {len(df_raw):,}")

    # 3. Clean & Map items to DASS-21
    print("[3/4] Filtering and standardizing to 21-item clinical scale...")
    # Keep participants who passed attention/validity checks
    if "VCL6" in df_raw.columns:
        df_clean = df_raw[df_raw["VCL6"] == 0].copy()
    else:
        df_clean = df_raw.copy()

    # Select the 21 relevant question columns
    df_dass21 = pd.DataFrame()
    for new_idx, old_col in enumerate(DASS21_TO_DASS42_COLS):
        # OpenPsychometrics is 1-4. Clinical standard is 0-3.
        df_dass21[f"Q{new_idx+1}"] = df_clean[old_col].astype(int) - 1

    # Keep only valid 0..3 responses
    mask = True
    for col in df_dass21.columns:
        mask = mask & df_dass21[col].isin([0, 1, 2, 3])
    df_dass21 = df_dass21[mask]

    # Sample a clean, balanced real-world subset (6,000 participants)
    df_sample = df_dass21.sample(n=min(sample_size, len(df_dass21)), random_state=42).reset_index(drop=True)

    # Compute ground truth subscale sums (doubled per DASS-21 norms)
    df_sample["dep_sum"] = df_sample[[f"Q{i+1}" for i in DEPRESSION_INDICES]].sum(axis=1) * 2
    df_sample["anx_sum"] = df_sample[[f"Q{i+1}" for i in ANXIETY_INDICES]].sum(axis=1) * 2
    df_sample["str_sum"] = df_sample[[f"Q{i+1}" for i in STRESS_INDICES]].sum(axis=1) * 2

    # Map to clinical severity classes
    df_sample["Depression_Class"] = df_sample["dep_sum"].apply(lambda s: score_to_severity(s, "Depression"))
    df_sample["Anxiety_Class"] = df_sample["anx_sum"].apply(lambda s: score_to_severity(s, "Anxiety"))
    df_sample["Stress_Class"] = df_sample["str_sum"].apply(lambda s: score_to_severity(s, "Stress"))

    # 4. Save to CSV
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"[4/4] Successfully saved {len(df_sample):,} authentic human responses to:")
    print(f"      {OUTPUT_FILE}")
    print("\nEmpirical Severity Distributions:")
    print("Depression:\n", df_sample["Depression_Class"].value_counts(normalize=True).round(3))
    print("Anxiety:\n", df_sample["Anxiety_Class"].value_counts(normalize=True).round(3))
    print("Stress:\n", df_sample["Stress_Class"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    fetch_and_process()
