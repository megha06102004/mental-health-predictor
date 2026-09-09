# 🧠 MindPulse AI: Predicting Depression, Anxiety, and Stress (DASS-21)
## 🎓 Complete Technical Deep Dive & Master Interview Preparation Guide
**Candidate:** Megha Bhandari  
**Target Role:** Machine Learning Engineer / AI Engineer / Data Scientist  

---

## ⏱️ Part 1: The 60-Second Elevator Pitch
> *"When the interviewer asks: 'Tell me about your Mental Health Prediction project' — use this exact script:"*

*"For my machine learning project, I built **MindPulse AI**, an end-to-end clinical psychometric screening system based on the internationally validated **DASS-21 (Depression, Anxiety, and Stress Scales)** developed by the University of New South Wales.

Rather than using toy synthetic data, I trained the system on authentic human responses from the **OpenPsychometrics Research Archive** ($N = 39,775$, sampled $6,000$ validated participant records after cleaning for response latency). I engineered a custom Scikit-learn feature engineering transformer that extracts clinical sub-clusters like autonomic arousal and anhedonia, alongside cross-subscale distress ratios.

I evaluated both classical ensemble methods and deep learning: a tuned **Random Forest Classifier** achieved **96.1% test accuracy** across clinical severity tiers, while a **TensorFlow/Keras Multi-Task Deep Neural Network (MTL-DNN)** learned shared latent representations across all three emotional dimensions simultaneously. To provide actionable clinical utility, the system generates interactive **Matplotlib triad radar charts** and symptom attribution graphs for Explainable AI."*

---

## 🏗️ Part 2: End-to-End System Architecture

```
[ User Completes 21 Likert Questionnaire (0 to 3) ]
                       │
                       ▼
    [ Real Dataset: OpenPsychometrics DASS Archive ]
      - 39,775 international participants
      - Quality Filtering: VCL >= 10s (eliminates rapid clickers)
      - Standardized to 21 DASS items + clinical cutoff mapping
                       │
                       ▼
      [ Feature Engineering Pipeline (Pandas / NumPy) ]
      - Raw 21 Likert items (Q1 to Q21)
      - Subscale sums (dep_sum, anx_sum, str_sum)
      - Global emotional distress index (Total sum)
      - Clinical symptom sub-clusters:
        * Autonomic Arousal (Heart pounding, dry mouth: Q2, Q4, Q19)
        * Anhedonia / Dysphoria (Lack of initiative, no joy: Q3, Q5, Q16)
        * Agitation & Nervous Energy (Touchiness, tension: Q1, Q6, Q8, Q12)
      - Interaction Ratios (dep_ratio, anx_ratio, str_ratio)
      - Total: 31 Features -> StandardScaler (mean=0, std=1)
                       │
         ┌─────────────┴────────────────────────┐
         ▼                                      ▼
[ Scikit-Learn Supervised Ensembles ]   [ Keras Multi-Task Deep Neural Network ]
├── Random Forest (120 trees): 96.1% acc ├── Shared Latent Trunk: Dense(128) -> Dense(64)
├── Gradient Boosting: 95.8% acc         ├── Task Head 1: Depression Softmax (5 tiers)
└── Multinomial Logistic: 94.8% acc      ├── Task Head 2: Anxiety Softmax (5 tiers)
                                         └── Task Head 3: Stress Softmax (5 tiers)
                       │                                │
                       └──────────────┬─────────────────┘
                                      ▼
                      [ Diagnostic Report Generation ]
                      ├── Severity Badges: Normal, Mild, Moderate, Severe, Extremely Severe
                      ├── Confidence Scores %
                      ├── Matplotlib Dynamic Radar Chart
                      └── Evidence-Based Coping Strategies (CBT Guidance)
```

---

## 🔬 Part 3: Deep Technical Concepts & Clinical Foundations

### 1. What is the DASS-21 Instrument?
* Developed by Lovibond & Lovibond (1995) at the University of New South Wales (UNSW).
* It measures three related negative emotional states:
  * **Depression Subscale:** Dysphoria, hopelessness, devaluation of life, self-deprecation, lack of interest/involvement, anhedonia, and inertia ($Q_3, Q_5, Q_{10}, Q_{13}, Q_{16}, Q_{17}, Q_{21}$).
  * **Anxiety Subscale:** Autonomic arousal, skeletal musculature effects, situational anxiety, and subjective experience of anxious affect ($Q_2, Q_4, Q_7, Q_9, Q_{15}, Q_{19}, Q_{20}$).
  * **Stress Subscale:** Chronic non-specific arousal, difficulty relaxing, nervous arousal, being easily upset/agitated, irritable/over-reactive, and impatient ($Q_1, Q_6, Q_8, Q_{11}, Q_{12}, Q_{14}, Q_{18}$).
* Rated on a 4-point Likert scale: $0 = \text{Did not apply}$, $1 = \text{Applied sometimes}$, $2 = \text{Applied a considerable degree}$, $3 = \text{Applied very much}$.
* Multiplied by $2$ to map onto standard DASS-42 clinical severity norms:
  * **Normal, Mild, Moderate, Severe, Extremely Severe**.

### 2. Real Dataset Provenance (Crucial Interview Topic!)
* **Source:** OpenPsychometrics raw research archive.
* **Why raw data cleaning is required:** Internet surveys suffer from *careless responding*. Participants click randomly to reach the end.
* **Our Cleaning Protocol:**
  1. Filtered by *Validity Check Latency* ($VCL \ge 10\text{s}$), removing unrealistically fast survey completions.
  2. Mapped the 42 original questions to the 21 DASS-21 subsets.
  3. Extracted 6,000 clean participant records saved to `data/real_dass_data.csv`.

### 3. Feature Engineering: Beyond Raw Questions
* Raw questionnaires only give discrete integer values ($0-3$).
* To give the machine learning models inductive bias, we engineered:
  * **Symptom Sub-Clusters:** Grouping autonomic arousal ($Q_2+Q_4+Q_{19}$) isolates physiological panic from cognitive worry.
  * **Global Distress Score:** In psychology, negative affectivity shares a general factor ($g$-factor). Summing all 21 items creates a composite distress measure.
  * **Cross-Subscale Ratios:** $\frac{\text{dep\_sum}}{\text{total\_distress} + 10^{-4}}$ measures relative symptom dominance (e.g. is the patient primarily depressed or primarily anxious?).

### 4. Why Multi-Task Learning (MTL) in Keras?
* **Single-Task Learning Problem:** Training 3 independent neural networks assumes depression, anxiety, and stress are statistically independent. In reality, they share biological and cognitive pathways (high comorbidity).
* **Multi-Task Architecture:**
  * A **shared trunk** (Dense 128 $\to$ BatchNorm $\to$ Dense 64) learns common underlying affective representation.
  * Three **task-specific heads** branch off to specialize on unique subscale criteria.
* **Joint Loss Function:**
  $$\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{dep}} + 1.0 \times \mathcal{L}_{\text{anx}} + 1.0 \times \mathcal{L}_{\text{str}}$$
  Using `sparse_categorical_crossentropy` across all three heads.
* **Benefits:** Regularization effect (prevents overfitting to any single task), parameter efficiency (fewer weights than 3 separate networks), and joint inference speed.

### 5. Why Random Forest outperformed other classifiers?
* Random Forest achieved **96.1% accuracy** because:
  * Decision trees naturally mirror the clinical cutoff boundaries defined by psychiatric guidelines (e.g., if score $\ge 14$ and $< 21 \implies$ Moderate).
  * Bagging (120 bootstrapped trees) with random feature sub-sampling reduces variance and eliminates individual tree overfitting.

---

## 💻 Part 4: Tricky Code Lines Explained Line-by-Line

### 1. Multi-Target Joint Stratified Splitting (`src/train.py`):
```python
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_dep)
X_train, X_test = X[train_idx], X[test_idx]
y_dep_train, y_dep_test = y_dep[train_idx], y_dep[test_idx]
y_anx_train, y_anx_test = y_anx[train_idx], y_anx[test_idx]
y_str_train, y_str_test = y_str[train_idx], y_str[test_idx]
```
* **Why this is tricky:** Beginners call `train_test_split(X, y_dep)` and then `train_test_split(X, y_anx)`. That shuffles the rows differently for each target! The features for participant $i$ would no longer match the anxiety label for participant $i$. Splitting an **array of indices** guarantees that $X$, $y_{\text{dep}}$, $y_{\text{anx}}$, and $y_{\text{str}}$ remain perfectly synchronized across all train/test splits.

### 2. Safe Epsilon in Ratio Calculations (`src/preprocessing.py`):
```python
eps = 1e-4
features["dep_ratio"] = features["dep_sum"] / (features["total_distress"] + eps)
```
* **Why this is tricky:** If a healthy participant answers $0$ to all 21 questions, `total_distress` is $0$. Division by zero produces `NaN` or `Inf`, which crashes Scikit-learn scalers and causes neural networks to output `NaN` loss. Adding $\epsilon = 10^{-4}$ provides numerical stability without distorting the ratio.

### 3. Matplotlib Radar Chart Angle Wrapping (`src/visualizer.py`):
```python
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values += values[:1]
angles += angles[:1]
ax.plot(angles, values, color="#4F46E5", linewidth=2)
ax.fill(angles, values, color="#4F46E5", alpha=0.25)
```
* **Why this is tricky:** A polar plot requires a closed polygon. If you have 3 vertices, plotting 3 points leaves the shape open between the last point and the first point. Appending the first element to the end (`values += values[:1]`, `angles += angles[:1]`) closes the loop seamlessly.

---

## 🎯 Part 5: Top 7 Interview Questions & Winning Answers

#### Q1: "What real dataset did you use to train your models?"
**Your Answer:** *"I used the OpenPsychometrics DASS Research Archive, which contains raw responses from over 39,000 international participants. Because web survey data contains noise from rapid random clicking, I filtered records using their validity completion latency (VCL >= 10s) and mapped the questionnaire to standard DASS-21 Likert ratings (0 to 3), creating a clean benchmark cohort of 6,000 participants."*

#### Q2: "Why did you build both Scikit-learn models and a Deep Learning model?"
**Your Answer:** *"I wanted to compare inductive biases: clinical cutoff thresholds in psychological screening are piece-wise constant, making ensemble tree models like Random Forest exceptionally accurate (96.1% accuracy). However, Depression, Anxiety, and Stress are clinically comorbid. To capture shared latent representations across all three disorders simultaneously, I engineered a Multi-Task Deep Neural Network in Keras with shared layers and task-specific softmax heads, training all three tasks jointly."*

#### Q3: "What features did you engineer, and why?"
**Your Answer:** *"Starting from 21 raw Likert items, I engineered 10 clinical domain features: subscale sums, a global emotional distress index, cross-subscale interaction ratios, and symptom sub-clusters isolating autonomic arousal (heart rate, breathing) and anhedonia (inability to experience pleasure). This expanded the feature space to 31 normalized dimensions, providing explicit signals that improved classification performance."*

#### Q4: "How did you prevent data leakage in your preprocessing pipeline?"
**Your Answer:** *"Data leakage was prevented by encapsulating transformations inside a Scikit-learn Pipeline. The `StandardScaler` was fitted strictly on the training partition (`fit_transform(X_train)`), and only applied to the test partition (`transform(X_test)`). This ensured that test set statistics (mean and standard deviation) were never exposed to the model during training."*

#### Q5: "How do you handle class imbalance across severity levels?"
**Your Answer:** *"In psychometric populations, 'Normal' and 'Mild' cases are often more frequent than 'Extremely Severe' cases. I handled this by employing stratified train-test splitting (`stratify=y`) to maintain identical class ratios across partitions, evaluated models using weighted F1-scores rather than just raw accuracy, and configured balanced class weights in our classifiers."*

#### Q6: "How did you make the model's predictions explainable to a healthcare practitioner?"
**Your Answer:** *"Black-box predictions are unacceptable in healthcare. I implemented symptom attribution using Random Forest feature importance, highlighting the specific items (e.g., severe anhedonia or autonomic panic) driving the diagnosis. Furthermore, I rendered dynamic Matplotlib triad radar charts visually mapping the participant's severity polygon against population clinical thresholds."*

#### Q7: "What are the ethical considerations and limitations of this system?"
**Your Answer:** *"MindPulse AI is explicitly designed as a clinical decision-support and screening tool, not a diagnostic authority. In production, every assessment includes a prominent disclaimer that results must be validated by a licensed clinical psychologist or psychiatrist, alongside emergency helpline contacts (e.g., Tele-MANAS) for individuals screening in severe tiers."*
