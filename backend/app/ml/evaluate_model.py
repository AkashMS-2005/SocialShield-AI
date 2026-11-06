"""
ml/evaluate_model.py – Evaluation on held-out test split, confusion matrix,
ROC curves, calibration curve generation, and documentation generation.

Rules:
  - Real, unmanipulated metrics only.
  - Generates docs/MODEL_EVALUATION.md.
  - Saves confusion matrix and ROC curve PNGs to docs/.
  - Explicit warning if dataset is synthetic.
"""
from __future__ import annotations

import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    brier_score_loss,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BACKEND_DIR / "app" / "models"
DOCS_DIR = BACKEND_DIR.parent / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SPLITS_PATH = MODELS_DIR / "test_data.joblib"
EVALUATION_REPORT_PATH = DOCS_DIR / "MODEL_EVALUATION.md"
ROC_PLOT_PATH = DOCS_DIR / "roc_curves.png"
CM_PLOT_PATH = DOCS_DIR / "confusion_matrices.png"
CALIB_PLOT_PATH = DOCS_DIR / "calibration_curve.png"


def run_evaluation():
    print("=" * 75)
    print("SocialShield AI – Comprehensive Test Set Evaluation")
    print("=" * 75)

    if not DATASET_SPLITS_PATH.exists():
        raise FileNotFoundError(f"Missing {DATASET_SPLITS_PATH}. Please run train_model.py first.")

    data = joblib.load(DATASET_SPLITS_PATH)
    X_test_trans = data["X_test_trans"]
    y_test = data["y_test"]
    best_estimators = data["best_estimators"]
    winner_name = data["winner_name"]
    calibrated_model = data["calibrated_model"]
    dataset_name = data["dataset_name"]
    is_synthetic = data["is_synthetic"]

    models_eval = {}

    # Evaluate each of the 3 candidate models on test set
    for name, est in best_estimators.items():
        probs = est.predict_proba(X_test_trans)[:, 1]
        preds = est.predict(X_test_trans)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probs)
        cm = confusion_matrix(y_test, preds)

        models_eval[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm,
            "probs": probs,
        }

    # Evaluate the Calibrated winner
    calib_probs = calibrated_model.predict_proba(X_test_trans)[:, 1]
    calib_preds = (calib_probs >= 0.5).astype(int)
    calib_brier = brier_score_loss(y_test, calib_probs)
    calib_cm = confusion_matrix(y_test, calib_preds)

    models_eval[f"{winner_name} (Calibrated)"] = {
        "accuracy": float(accuracy_score(y_test, calib_preds)),
        "precision": float(precision_score(y_test, calib_preds, zero_division=0)),
        "recall": float(recall_score(y_test, calib_preds, zero_division=0)),
        "f1": float(f1_score(y_test, calib_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, calib_probs)),
        "confusion_matrix": calib_cm,
        "probs": calib_probs,
        "brier_score": float(calib_brier),
    }

    print("\n--- Held-Out Test Set Performance ---")
    for name, m in models_eval.items():
        print(f"[{name}] Acc: {m['accuracy']:.4f} | Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------
    # 1. ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, m in models_eval.items():
        fpr, tpr, _ = roc_curve(y_test, m["probs"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {m['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Held-Out Test Set")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=150)
    plt.close()

    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (name, est) in enumerate(best_estimators.items()):
        cm = models_eval[name]["confusion_matrix"]
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(name)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Genuine (0)", "Suspicious (1)"])
        ax.set_yticklabels(["Genuine (0)", "Suspicious (1)"])
        
        # Annotate numbers
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH, dpi=150)
    plt.close()

    # 3. Calibration Curve Plot
    prob_true, prob_pred = calibration_curve(y_test, calib_probs, n_bins=5, strategy="uniform")
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{winner_name} (Calibrated)")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram / Calibration Curve")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CALIB_PLOT_PATH, dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # Write Markdown Report
    # -----------------------------------------------------------------------
    report_lines = [
        "# SocialShield AI – Model Evaluation Report",
        "",
    ]
    if is_synthetic:
        report_lines.extend([
            "> [!WARNING]",
            "> **SYNTHETIC DATA NOTICE — NOT REAL-WORLD PERFORMANCE**",
            "> The metrics below were measured on synthetic dataset `backend/data/raw/synthetic_SAMPLE_ONLY.csv`.",
            "> These verify end-to-end mathematical correctness, calibration, and pipeline reproducibility.",
            "> They must NOT be cited as real-world account detection accuracy.",
            "",
        ])

    report_lines.extend([
        "## 1. Dataset & Test Split Summary",
        "",
        f"- **Dataset Source:** `{dataset_name}`",
        f"- **Held-Out Test Set Size:** {len(y_test)} accounts",
        f"- **Class Distribution in Test Set:** Genuine: {sum(y_test == 0)}, Suspicious/Fake: {sum(y_test == 1)}",
        f"- **Selected Best Architecture:** `{winner_name}` (Calibrated via Sigmoid CalibratedClassifierCV)",
        "",
        "## 2. Test Set Model Comparison",
        "",
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |",
        "|-------|----------|-----------|--------|----------|---------|",
    ])

    for name, m in models_eval.items():
        report_lines.append(
            f"| **{name}** | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} |"
        )

    report_lines.extend([
        "",
        "## 3. Probability Calibration Quality",
        "",
        f"- **Brier Score (Calibrated {winner_name}):** `{calib_brier:.4f}` *(lower is better, 0.0 is perfect)*",
        "- **Calibration Method:** Sigmoid / Platt scaling with 3-fold cross-validation on the training set.",
        "- **Why Calibration Matters:** In a risk system, a score of 0.8 must reflect an 80% empirical risk probability, not an uncalibrated margin score.",
        "",
        "## 4. Evaluation Plots",
        "",
        "- **ROC Curves:** Saved to `docs/roc_curves.png`",
        "- **Confusion Matrices:** Saved to `docs/confusion_matrices.png`",
        "- **Reliability Diagram:** Saved to `docs/calibration_curve.png`",
        "",
        "## 5. Limitations & Ethical Caveats",
        "",
        "1. **Risk Signal vs Proof:** Outputs are suspicious indicators; false positives occur on sparse profiles.",
        "2. **Platform Generalization:** Models trained on single-platform tabular ratios have degraded cross-platform transferability.",
        "3. **Missing Data Handling:** Unknown profile fields trigger missing indicators rather than being masked as zero risk.",
    ])

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Generated evaluation report at: {EVALUATION_REPORT_PATH}")
    print(f"✅ Generated ROC curves at: {ROC_PLOT_PATH}")
    print(f"✅ Generated confusion matrices at: {CM_PLOT_PATH}")


if __name__ == "__main__":
    run_evaluation()

