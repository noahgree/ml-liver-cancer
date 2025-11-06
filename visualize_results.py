#!/usr/bin/env python3
"""
visualize_results.py — Journal-style plots for a trained logistic regression model.

Usage:
  python visualize_results.py \
      --model model.pkl \
      --test preprocessed/test.csv \
      --target liver_cancer \
      --threshold 0.44

Notes:
- Uses only matplotlib (no seaborn).
- Saves figures to ./figures
- Will attempt to import helper functions from logistic_regression.py if available,
  but can also work without it.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    precision_score, recall_score,
    brier_score_loss,
    roc_auc_score
)

# ---------------------------
# Matplotlib aesthetic setup
# ---------------------------
# Soft journal-like style: muted palette, thin lines, readable fonts
plt.rcParams.update({
    "figure.figsize": (7.0, 5.0),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.3,
    "axes.titleweight": "semibold",
    "axes.labelweight": "regular",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.size": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})

# Soft colors (explicitly set because user requested soft journal style)
COLORS = {
    "primary": "#4C78A8",      # muted blue
    "secondary": "#F58518",    # soft orange
    "accent": "#72B7B2",       # teal
    "neutral": "#9EA3A8",      # grey
    "danger": "#E45756",       # muted red
    "safe": "#54A24B"          # muted green
}


def try_import_helpers():
    """Try to import helpers from logistic_regression.py if present."""
    try:
        from logistic_regression import prepare_features_and_target
        return prepare_features_and_target
    except Exception:
        return None


def fallback_prepare_features_and_target(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def plot_roc(y_true, y_proba, outdir):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, color=COLORS["primary"], label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color=COLORS["neutral"], linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    ensure_dir(outdir)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curve.png"))
    plt.show()


def plot_pr(y_true, y_proba, outdir):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = np.trapz(precision, recall)

    plt.figure()
    plt.plot(recall, precision, linewidth=2, color=COLORS["accent"], label=f"PR AUC ≈ {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="upper right")
    ensure_dir(outdir)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "precision_recall_curve.png"))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, outdir):
    cm = confusion_matrix(y_true, y_pred)
    # Manual heatmap using imshow (no seaborn)
    plt.figure()
    im = plt.imshow(cm, cmap="Greys")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")

    # Annotate counts
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f"{val}", ha="center", va="center", color="black")

    plt.tight_layout()
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"))
    plt.show()


def plot_threshold_sweep(y_true, y_proba, selected_threshold, outdir):
    thresholds = np.linspace(0.05, 0.95, 200)
    precisions, recalls = [], []
    for t in thresholds:
        y_hat = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_true, y_hat, zero_division=0))
        recalls.append(recall_score(y_true, y_hat, zero_division=0))

    plt.figure()
    plt.plot(thresholds, precisions, color=COLORS["primary"], label="Precision", linewidth=2)
    plt.plot(thresholds, recalls, color=COLORS["secondary"], label="Recall", linewidth=2)
    if selected_threshold is not None:
        plt.axvline(selected_threshold, color=COLORS["neutral"], linestyle="--", linewidth=1.5, label=f"Selected τ={selected_threshold:.2f}")
    plt.xlabel("Threshold (τ)")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, "threshold_sweep.png"))
    plt.show()


def plot_coefficients(model, feature_names, outdir, top_k=12):
    # Handle pipelines or calibrated models
    coef = None
    clf = model
    if hasattr(model, "coef_"):
        coef = model.coef_[0]
    elif hasattr(model, "base_estimator_"):  # CalibratedClassifierCV
        base = getattr(model, "base_estimator_")
        if hasattr(base, "coef_"):
            coef = base.coef_[0]
            clf = base
    elif hasattr(model, "named_steps"):
        # Try to pull from pipeline last step if it's LR
        last = list(model.named_steps.values())[-1]
        if hasattr(last, "coef_"):
            coef = last.coef_[0]
            clf = last

    if coef is None:
        print("[warn] Could not extract coefficients from the provided model. Skipping coefficient plot.")
        return

    # If feature names not given (e.g., pipeline), try generic names
    if feature_names is None or len(feature_names) != len(coef):
        feature_names = [f"x{i}" for i in range(len(coef))]

    idx = np.argsort(np.abs(coef))[::-1][:top_k]
    sel_feats = np.array(feature_names)[idx]
    sel_vals = coef[idx]

    colors = [COLORS["safe"] if v >= 0 else COLORS["danger"] for v in sel_vals]

    plt.figure(figsize=(7, 6))
    y_pos = np.arange(len(sel_feats))
    plt.barh(y_pos, sel_vals, color=colors)
    plt.yticks(y_pos, sel_feats)
    plt.gca().invert_yaxis()
    plt.xlabel("Coefficient (log-odds weight)")
    plt.title("Top Feature Contributions (|coef|)")
    plt.tight_layout()
    ensure_dir(outdir)
    plt.savefig(os.path.join(outdir, "coefficients.png"))
    plt.show()


def evaluate_basic(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "AUROC": roc_auc_score(y_true, y_proba),
        "Recall": recall_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Brier": brier_score_loss(y_true, y_proba),
        "Accuracy": (y_pred == y_true).mean(),
        "Threshold": threshold,
        "ConfusionMatrix": confusion_matrix(y_true, y_pred)
    }
    return metrics, y_pred


def main():
    ap = argparse.ArgumentParser(description="Visualize results for a trained logistic regression model.")
    ap.add_argument("--model", required=True, help="Path to saved model .pkl (e.g., model.pkl)")
    ap.add_argument("--test", required=True, help="Path to test CSV (e.g., preprocessed/test.csv)")
    ap.add_argument("--target", default="liver_cancer", help="Target column name (default: liver_cancer)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for confusion matrix (default: 0.5)")
    ap.add_argument("--outdir", default="figures", help="Directory to save figures (default: figures)")
    args = ap.parse_args()

    # Load model
    model = load_model(args.model)

    # Load data
    test_df = pd.read_csv(args.test)

    # Prepare features/target
    helper = try_import_helpers()
    if helper is not None:
        X_test, y_test = helper(test_df, args.target)
    else:
        X_test, y_test = fallback_prepare_features_and_target(test_df, args.target)

    # Get probabilities
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Some pipelines may require decision_function; map to [0,1] via sigmoid as fallback
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))
        else:
            raise ValueError("Model does not support predict_proba or decision_function.")

    # Evaluate a few basics at the chosen threshold
    metrics, y_pred = evaluate_basic(y_test, y_proba, args.threshold)
    print("[Info] Basic metrics at threshold τ={:.2f}:".format(args.threshold))
    for k, v in metrics.items():
        if k == "ConfusionMatrix":
            cm = v
            print("  Confusion Matrix:\n    TN={} FP={}\n    FN={} TP={}".format(cm[0,0], cm[0,1], cm[1,0], cm[1,1]))
        else:
            print(f"  {k}: {v:.4f}" if isinstance(v, (float, int)) else f"  {k}: {v}")

    # Plots
    plot_roc(y_test, y_proba, args.outdir)
    plot_pr(y_test, y_proba, args.outdir)
    plot_confusion_matrix(y_test, y_pred, args.outdir)
    plot_threshold_sweep(y_test, y_proba, args.threshold, args.outdir)

    # Try to extract feature names if DataFrame
    feat_names = list(X_test.columns) if hasattr(X_test, "columns") else None
    plot_coefficients(model, feat_names, args.outdir)


if __name__ == "__main__":
    main()
