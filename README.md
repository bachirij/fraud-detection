# Credit Card Fraud Detection

A complete end-to-end machine learning project: from exploratory data analysis to model evaluation and business recommendation, built on a real-world imbalanced dataset.

> Built as a portfolio project covering both **supervised** and **unsupervised** learning techniques.

---

## Problem Statement

A bank wants to automatically detect fraudulent credit card transactions in real time.  
The dataset is highly imbalanced: only **0.17% of transactions are fraudulent**.  
A naive model that labels everything as legitimate would achieve 99.8% accuracy, and catch zero fraud.

The goal is to build a model that **maximizes fraud recall** (minimizing missed frauds) while keeping false positives at an acceptable level.

---

## 📂 Dataset

**Source:** [Credit Card Fraud Detection - Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Feature                 | Detail                                      |
| ----------------------- | ------------------------------------------- |
| Total transactions      | 284,807                                     |
| Fraudulent transactions | 492 (0.17%)                                 |
| Features                | V1–V28 (PCA-anonymized) + `Time` + `Amount` |
| Target                  | `Class` : 0 = legitimate, 1 = fraud         |
| Format                  | CSV (~150 MB)                               |

> The CSV is not included in this repository. Download it from the Kaggle link above and place it in the `data/` folder.

---

## Project Structure

```
fraud-detection/
├── README.md
├── requirements.txt
├── data/                        ← place creditcard.csv here (not tracked by git)
└── notebooks/
    └── fraud_detection.ipynb    ← main notebook
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Class distribution, imbalance visualization
- `Amount` and `Time` distributions by class
- Correlation heatmap of V1–V28 features

### 2. Preprocessing

- Normalization of `Amount` and `Time` with `StandardScaler`
- Stratified train/test split (80/20)
- Class imbalance handling: `class_weight='balanced'` + optional SMOTE

### 3. Supervised Learning

| Model               | Notes                                                 |
| ------------------- | ----------------------------------------------------- |
| Logistic Regression | Baseline — linear decision boundary                   |
| Decision Tree       | Visual, explainable, prone to overfitting             |
| Random Forest       | Ensemble of 100 trees, reduces variance               |
| XGBoost             | Sequential boosting, `scale_pos_weight` for imbalance |

### 4. Unsupervised Learning

| Method        | Goal                                                    |
| ------------- | ------------------------------------------------------- |
| K-Means + PCA | Segment transactions, check if fraud clusters emerge    |
| DBSCAN        | Anomaly detection — outliers as potential fraud signals |

---

## Key Results

| Model               | Precision (fraud) | Recall (fraud) | F1  | AUC-ROC |
| ------------------- | ----------------- | -------------- | --- | ------- |
| Logistic Regression | —                 | —              | —   | —       |
| Decision Tree       | —                 | —              | —   | —       |
| Random Forest       | —                 | —              | —   | —       |
| XGBoost             | —                 | —              | —   | —       |

> _Results will be filled in once the notebook is complete._

---

## Evaluation Metrics

Accuracy is **not** a valid metric here due to class imbalance.  
The project focuses on:

- **Recall** — minimize missed frauds (false negatives = real financial loss)
- **F1-score** — harmonic mean of precision and recall
- **AUC-ROC** — model comparison independent of decision threshold

---

## Business Recommendation

> _To be completed after model evaluation._

Preliminary recommendation: deploy XGBoost with a lowered decision threshold (~0.3) via a FastAPI scoring endpoint, with real-time monitoring of false positive rate to minimize unnecessary friction for legitimate customers.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/bachirij/fraud-detection.git
cd fraud-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebooks/fraud_detection.ipynb
```

---

## Requirements

Full list in [`requirements.txt`](./requirements.txt).

---

## Roadmap

- [x] Project structure and README
- [ ] EDA + Preprocessing
- [ ] Supervised learning (4 models)
- [ ] Unsupervised learning (K-Means, DBSCAN)
- [ ] Comparative evaluation + business recommendation
- [ ] FastAPI deployment (stretch goal)

---

## License

MIT License - feel free to fork, adapt, and use as a learning reference.
