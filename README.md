# Credit Card Fraud Detection

A complete end-to-end machine learning project: from exploratory data analysis to a deployed interactive dashboard, built on a real-world imbalanced dataset.

> Built as a portfolio project covering **supervised learning**, **unsupervised learning**, and **production deployment** via Streamlit.

**[Live Dashboard](https://bachirij-fraud-detection.streamlit.app)**

---

## Problem Statement

A bank wants to automatically detect fraudulent credit card transactions in real time.  
The dataset is highly imbalanced: only **0.17% of transactions are fraudulent**.  
A naive model that labels everything as legitimate would achieve 99.8% accuracy, and catch zero fraud.

The goal is to build a model that **maximizes fraud recall** (minimizing missed frauds) while keeping false positives at an acceptable level.

---

## Dataset

**Source:** [Credit Card Fraud Detection - Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Feature                 | Detail                                      |
| ----------------------- | ------------------------------------------- |
| Total transactions      | 284,807                                     |
| Fraudulent transactions | 492 (0.17%)                                 |
| Features                | V1–V28 (PCA-anonymized) + `Time` + `Amount` |
| Target                  | `Class` : 0 = legitimate, 1 = fraud         |
| Format                  | CSV (~150 MB)                               |

> The dataset is not included in this repository. It is automatically downloaded via `kagglehub` on first run.

---

## Project Structure

```
fraud-detection/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        ← auto-downloaded via kagglehub (git-ignored)
│   └── processed/                  ← generated at first run (git-ignored)
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── scaler.joblib
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_supervised.ipynb
│   ├── 04_unsupervised.ipynb
│   └── 05_summary.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluate.py
│   └── predict.py
└── app/
    ├── dashboard.py
    ├── tabs/
    │   ├── overview.py
    │   ├── comparison.py
    │   ├── threshold.py
    │   ├── features.py
    │   └── scorer.py
    └── utils/
        └── data_loader.py
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Class distribution and imbalance visualization
- `Amount` and `Time` distributions by class
- Correlation heatmap: V17, V14, V12 most discriminative features
- Key finding: fraudulent transactions are concentrated between 0€ and 150€

### 2. Preprocessing

- Normalization of `Amount` and `Time` with `StandardScaler`
- Stratified train/test split (80/20): fraud ratio preserved in both sets
- Class imbalance handling via SMOTE on training set only (394 → 227,451 fraud samples)

### 3. Supervised Learning

| Model               | Notes                                                        |
| ------------------- | ------------------------------------------------------------ |
| Logistic Regression | Baseline, trained on SMOTE-resampled data, `max_iter=1000`   |
| Decision Tree       | Visual, explainable, prone to overfitting                    |
| Random Forest       | Ensemble of 100 trees, trained on SMOTE-resampled data       |
| XGBoost             | Sequential boosting, `scale_pos_weight=577` on original data |

### 4. Unsupervised Learning

| Method        | Goal                                                                             |
| ------------- | -------------------------------------------------------------------------------- |
| K-Means + PCA | Segment transactions: Cluster 3 concentrates 41% of frauds at a 3.36% fraud rate |
| DBSCAN        | Anomaly detection: outliers as potential fraud signals                           |

---

## Key Results

| Model               | Precision | Recall   | F1       | AUC-ROC | FN  | FP   |
| ------------------- | --------- | -------- | -------- | ------- | --- | ---- |
| Logistic Regression | 0.06      | 0.92     | 0.11     | 0.9698  | 8   | 1458 |
| Decision Tree       | 0.03      | 0.90     | 0.06     | 0.9522  | 10  | 2896 |
| Random Forest       | 0.82      | 0.82     | 0.82     | 0.9688  | 18  | 17   |
| **XGBoost**         | **0.88**  | **0.85** | **0.86** | 0.9652  | 15  | 11   |

**Selected model: XGBoost** - best F1 (0.86) and fewest false positives (11).  
Top features: V14, V10, V4, V12 - consistent across XGBoost and Random Forest.

---

## Evaluation Metrics

Accuracy is **not** a valid metric here due to class imbalance.  
The project focuses on:

- **Recall**: minimize missed frauds (false negatives = real financial loss)
- **F1-score**: harmonic mean of precision and recall
- **AUC-ROC**: model comparison independent of decision threshold

---

## Dashboard

The project includes an interactive Streamlit dashboard with five pages:

| Page                  | Description                                                                       |
| --------------------- | --------------------------------------------------------------------------------- |
| 📊 Business Overview  | KPIs, fraud exposure in €, amount distribution by class, class balance            |
| 🤖 Model Comparison   | Metrics table with best-value highlighting, interactive overlaid ROC curves       |
| 🎚️ Threshold Explorer | Real-time threshold adjustment - metrics, confusion matrix, financial impact      |
| 🔬 Feature Importance | XGBoost vs Random Forest top features, cross-model consensus analysis             |
| 💳 Transaction Scorer | Score individual transactions, fraud probability gauge, top contributing features |

---

## Business Recommendation

Deploy XGBoost with a decision threshold of **0.30** (vs default 0.50).  
At this threshold on the test set: **15 missed frauds** vs **11 false alarms**.

Recommended deployment via a FastAPI scoring endpoint with real-time monitoring of the false positive rate to minimize unnecessary friction for legitimate customers.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/bachirij/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
conda create -n ml_env python=3.12
conda activate ml_env

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/dashboard.py
```

> The dataset is downloaded automatically on first run via `kagglehub`.  
> Kaggle API credentials required — place them in `~/.kaggle/kaggle.json`.  
> Get your credentials at: [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token.

---

## Requirements

Full list in [`requirements.txt`](./requirements.txt).

---

## Roadmap

- [x] Project structure and README
- [x] EDA — class imbalance, distributions, correlation analysis
- [x] Preprocessing — StandardScaler, stratified split, SMOTE
- [x] Supervised learning — Logistic Regression, Decision Tree, Random Forest, XGBoost
- [x] Unsupervised learning — K-Means + PCA, DBSCAN
- [x] Modular `src/` refactoring
- [x] Streamlit dashboard (5 pages)
- [x] Streamlit Cloud deployment
- [ ] SHAP values for local explainability
- [ ] FastAPI scoring endpoint

---

## License

MIT License - feel free to fork, adapt, and use as a learning reference.
