# SocialShield AI – Model Evaluation Report

> [!WARNING]
> **SYNTHETIC DATA NOTICE — NOT REAL-WORLD PERFORMANCE**
> The metrics below were measured on synthetic dataset `backend/data/raw/synthetic_SAMPLE_ONLY.csv`.
> These verify end-to-end mathematical correctness, calibration, and pipeline reproducibility.
> They must NOT be cited as real-world account detection accuracy.

## 1. Dataset & Test Split Summary

- **Dataset Source:** `SYNTHETIC SAMPLE (backend/data/raw/synthetic_SAMPLE_ONLY.csv)`
- **Held-Out Test Set Size:** 75 accounts
- **Class Distribution in Test Set:** Genuine: 45, Suspicious/Fake: 30
- **Selected Best Architecture:** `Logistic Regression` (Calibrated via Sigmoid CalibratedClassifierCV)

## 2. Test Set Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.9867 | 0.9677 | 1.0000 | 0.9836 | 1.0000 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Logistic Regression (Calibrated)** | 0.9867 | 0.9677 | 1.0000 | 0.9836 | 1.0000 |

## 3. Probability Calibration Quality

- **Brier Score (Calibrated Logistic Regression):** `0.0070` *(lower is better, 0.0 is perfect)*
- **Calibration Method:** Sigmoid / Platt scaling with 3-fold cross-validation on the training set.
- **Why Calibration Matters:** In a risk system, a score of 0.8 must reflect an 80% empirical risk probability, not an uncalibrated margin score.

## 4. Evaluation Plots

- **ROC Curves:** Saved to `docs/roc_curves.png`
- **Confusion Matrices:** Saved to `docs/confusion_matrices.png`
- **Reliability Diagram:** Saved to `docs/calibration_curve.png`

## 5. Limitations & Ethical Caveats

1. **Risk Signal vs Proof:** Outputs are suspicious indicators; false positives occur on sparse profiles.
2. **Platform Generalization:** Models trained on single-platform tabular ratios have degraded cross-platform transferability.
3. **Missing Data Handling:** Unknown profile fields trigger missing indicators rather than being masked as zero risk.