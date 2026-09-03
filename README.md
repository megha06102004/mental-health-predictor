# 🧠 MindPulse AI: Predicting Depression, Anxiety, and Stress (DASS-21)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Pipelines-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-Multi--Task%20DNN-FF6F00.svg)](https://www.tensorflow.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Clinical%20Visualizations-11557C.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning psychometric screening platform based on the **Depression, Anxiety, and Stress Scales (DASS-21)** developed by the University of New South Wales (Lovibond & Lovibond). 

Combines classical **Scikit-learn ensemble estimators** with a **TensorFlow/Keras Multi-Task Deep Neural Network (MTL-DNN)** to classify psychological distress across 5 clinical severity tiers (*Normal, Mild, Moderate, Severe, Extremely Severe*) with Explainable AI symptom attribution.

---

## 🔬 Clinical Methodology & DASS-21 Instrument

The DASS-21 measures the negative emotional states of **Depression**, **Anxiety**, and **Stress** using 21 validated psychological items (7 items per subscale) rated on a 4-point Likert scale ($0$ to $3$):

* **Depression Items:** $Q_3, Q_5, Q_{10}, Q_{13}, Q_{16}, Q_{17}, Q_{21}$ (Dysphoria, hopelessness, anhedonia)
* **Anxiety Items:** $Q_2, Q_4, Q_7, Q_9, Q_{15}, Q_{19}, Q_{20}$ (Autonomic arousal, situational panic, tremors)
* **Stress Items:** $Q_1, Q_6, Q_8, Q_{11}, Q_{12}, Q_{14}, Q_{18}$ (Chronic tension, irritability, agitation)

### Clinical Severity Cutoffs (DASS-42 Equivalent Score = Raw Sum $\times 2$)

| Severity Tier | Depression Score | Anxiety Score | Stress Score |
| :--- | :--- | :--- | :--- |
| **Normal** | $0 - 9$ | $0 - 7$ | $0 - 14$ |
| **Mild** | $10 - 13$ | $8 - 9$ | $15 - 18$ |
| **Moderate** | $14 - 20$ | $10 - 14$ | $19 - 25$ |
| **Severe** | $21 - 27$ | $15 - 19$ | $26 - 33$ |
| **Extremely Severe** | $28+$ | $20+$ | $34+$ |

---

## 🏗️ System Architecture

```
                 User Responses (21 Likert Items)
                                │
                                ▼
         [Feature Engineering Pipeline - Pandas & NumPy]
         - Subscale sum aggregations (Dep, Anx, Str)
         - Autonomic arousal & anhedonia cluster scores
         - Total distress index & cross-subscale ratios
         - StandardScaler normalization
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
[Scikit-learn Ensemble]                [TensorFlow/Keras Multi-Task DNN]
- Tuned Random Forest (120 trees)      - Shared Representation Trunk (Dense 128->64)
- Gradient Boosting Classifier         - Task 1 Head: Depression Softmax (5 classes)
- Multinomial Logistic Regression      - Task 2 Head: Anxiety Softmax (5 classes)
- 5-Fold Cross-Validation              - Task 3 Head: Stress Softmax (5 classes)
          │                                           │
          └─────────────────────┬─────────────────────┘
                                ▼
                [Clinical Diagnostic Profile]
                ├── Severity Tier Badges & Confidence %
                ├── Matplotlib Triad Radar Chart
                ├── Key Symptom Attribution (Explainable AI)
                └── Evidence-Based Coping Strategies (CBT)
```

---

## 📊 Model Performance Benchmarks

### 1. Scikit-learn Estimators (Depression Subscale)
* **Random Forest Classifier:** **100.00%** Test Accuracy | Weighted F1: **1.0000**
* **Gradient Boosting Classifier:** **100.00%** Test Accuracy | Weighted F1: **1.0000**
* **Logistic Regression:** **99.83%** Test Accuracy | Weighted F1: **0.9983**

### 2. Multi-Task Deep Neural Network (TensorFlow / Keras)
* **Depression Head Test Accuracy:** **97.29%**
* **Anxiety Head Test Accuracy:** **94.00%**
* **Stress Head Test Accuracy:** **95.86%**
* Total Multi-Task Loss: **0.8154**

---

## 🚀 Quick Start

### 1. Installation
Ensure Python 3.10+ is installed:

```bash
git clone https://github.com/megha06102004/mental-health-predictor.git
cd mental-health-predictor
pip install -r requirements.txt
```

### 2. Launch Interactive Web Portal
Launches the responsive clinical assessment web application:

```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:8000`**

* Complete the 21 questions.
* Click **Analyze Psychological Profile** to generate instant severity ratings, deep learning confidence, and dynamic Matplotlib radar charts.

### 3. Run Training Pipeline (Optional)
To retrain both Scikit-learn models and TensorFlow Deep Learning from scratch:

```bash
# Train Scikit-learn models + GridSearchCV
python -m src.train

# Train TensorFlow / Keras Multi-Task DNN
python -m src.train_deep
```

### 4. Run Automated Unit Tests
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 📁 Repository Structure

```
mental-health-predictor/
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Clinical items, norms & empirical dataset generator
│   ├── preprocessing.py     # Feature engineering, sub-clusters, and scikit-learn scaling
│   ├── train.py             # Scikit-learn benchmarking & GridSearchCV tuning
│   ├── deep_learning.py     # TensorFlow/Keras Multi-Task Deep Neural Network architecture
│   ├── train_deep.py        # Keras MTL-DNN training & validation script
│   ├── visualizer.py        # Matplotlib radar charts & feature importance plots
│   └── predictor.py         # Hybrid Scikit-learn & Deep Learning inference engine
├── web/
│   ├── index.html           # Modern responsive questionnaire layout
│   ├── style.css            # Dark/light healthcare theme styling
│   └── app.js               # Likert scoring logic & REST API integration
├── models/
│   ├── depression_model.pkl # Trained Random Forest model
│   ├── anxiety_model.pkl    # Trained Random Forest model
│   ├── stress_model.pkl     # Trained Random Forest model
│   ├── multitask_dass_model.keras # Trained Keras Deep Learning model
│   ├── preprocessor.pkl     # Fitted feature engineering & scaler bundle
│   └── label_encoder.pkl    # Severity class label encoder
├── notebooks/
│   └── eda_and_modeling.ipynb # Step-by-step EDA & modeling Jupyter notebook
├── tests/
│   └── test_pipeline.py     # Automated unit tests
├── app.py                   # Python web server & API
├── requirements.txt         # Minimal verified dependencies
└── README.md                # Technical documentation
```

---

## 👩‍💻 Author
**Megha Bhandari**  
B.Tech Information Technology, Maharaja Surajmal Institute of Technology  
[GitHub Profile](https://github.com/megha06102004)
