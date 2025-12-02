"""
Random Forest model for liver cancer identification.

Trains a Random Forest classifier on preprocessed training data and evaluates
it using AUROC, Recall, Precision, and Brier Score metrics.

Compared to Logistic Regression, Random Forest can capture non‑linear
interactions between risk factors and provides feature importance scores that
highlight which medical features contribute most to predictions.

Usage:
  python3 random_forest.py --train preprocessed/train.csv --test preprocessed/test.csv
  python3 random_forest.py --train <train_csv> --test <test_csv> [--model-out <model_path>]
"""

import argparse
import os
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    brier_score_loss,
    confusion_matrix,
)

TARGET_COL = "liver_cancer"

# Project Goals (Expected Results)
# For Random Forest, we care more about *high recall* (catching as many
# positives as possible) and accept *lower precision* (more false positives)
# as a trade-off, which is typical for medical screening tools.
PROJECT_GOALS = {
    "AUROC": (0.80, 0.99),
    "Recall": (0.80, 0.99),      # prioritize sensitivity
    # For precision we only care about a lower bound; the upper value is not used
    # in threshold optimization (we're fine if precision is higher).
    "Precision": (0.40, 1.00),
    "Brier Score": 0.20,         # threshold: < 0.20
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and return DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from DataFrame.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str | float | int | None = "sqrt",
    random_state: int = 42,
    class_weight: str | dict | None = "balanced",
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )

    print("Training Random Forest model...")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - max_depth: {max_depth}")
    print(f"  - min_samples_split: {min_samples_split}")
    print(f"  - min_samples_leaf: {min_samples_leaf}")
    print(f"  - max_features: {max_features}")
    print(f"  - class_weight: {class_weight}")

    model.fit(X_train, y_train)
    return model


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    precision_range: Tuple[float, float] = (0.40, 1.00),
    recall_range: Tuple[float, float] = (0.80, 0.99),
) -> Tuple[float, Tuple[float, float] | None]:
    """
    Grid-search classification threshold.

    Design goal here: **maximize recall (sensitivity)** subject to having
    *at least* a minimal usable precision. In other words:
      - primary objective: higher recall
      - secondary: among thresholds with similar recall, prefer higher precision
    """
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    best_metrics: Tuple[float, float] | None = None

    precision_min, _ = precision_range

    for threshold in candidate_thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        # Filter out thresholds with unusably low precision
        if precision < precision_min:
            continue

        # Primary: maximize recall; Secondary: for the same recall, prefer higher precision
        if (recall > best_recall) or (np.isclose(recall, best_recall) and precision > best_precision):
            best_recall = recall
            best_precision = precision
            best_threshold = threshold
            best_metrics = (recall, precision)

    return best_threshold, best_metrics


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluate model and compute all required metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Brier Score": brier_score_loss(y_test, y_pred_proba),
        "Accuracy": (y_pred == y_test).mean(),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Threshold": threshold,
    }
    return metrics, y_pred, y_pred_proba


def print_results(metrics: Dict[str, Any], y_test: pd.Series) -> None:
    """Print evaluation results and compare against project goals."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST MODEL EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nTest Set Size: {len(y_test)}")
    print(f"Positive Cases: {int(y_test.sum())}")
    print(f"Negative Cases: {int((y_test == 0).sum())}")

    threshold_val = metrics.get("Threshold", 0.5)
    if abs(threshold_val - 0.5) < 0.001:
        print(f"\nClassification Threshold: {threshold_val:.4f} (default)")
    else:
        print(
            f"\nClassification Threshold: {threshold_val:.4f} "
            "(optimized, default: 0.5)"
        )

    print("\n" + "-" * 70)
    print("METRICS")
    print("-" * 70)

    for metric_name, goal_value in PROJECT_GOALS.items():
        value = metrics[metric_name]
        if metric_name == "Brier Score":
            meets_goal = value < goal_value
            comparison = "✓" if meets_goal else "✗"
            print(f"{metric_name:15s}: {value:.4f} (goal: < {goal_value:.2f}) {comparison}")
        else:
            min_val, max_val = goal_value
            meets_goal = min_val <= value <= max_val
            comparison = "✓" if meets_goal else "✗"
            if value < min_val:
                status = f"below range (target: {min_val:.2f}-{max_val:.2f})"
            elif value > max_val:
                status = f"above range (target: {min_val:.2f}-{max_val:.2f})"
            else:
                status = f"within range ({min_val:.2f}-{max_val:.2f})"
            print(f"{metric_name:15s}: {value:.4f} ({status}) {comparison}")

    print(f"{'Accuracy':15s}: {metrics['Accuracy']:.4f}")

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    cm = metrics["Confusion Matrix"]
    print("                Predicted")
    print("              Negative  Positive")
    print(f"Actual Negative   {cm[0, 0]:4d}      {cm[0, 1]:4d}")
    print(f"       Positive   {cm[1, 0]:4d}      {cm[1, 1]:4d}")

    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")

    print("\n" + "-" * 70)
    print("GOAL SUMMARY")
    print("-" * 70)

    auroc_min, auroc_max = PROJECT_GOALS["AUROC"]
    recall_min, recall_max = PROJECT_GOALS["Recall"]
    precision_min, precision_max = PROJECT_GOALS["Precision"]

    all_goals_met = all(
        [
            auroc_min <= metrics["AUROC"] <= auroc_max,
            recall_min <= metrics["Recall"] <= recall_max,
            precision_min <= metrics["Precision"] <= precision_max,
            metrics["Brier Score"] < PROJECT_GOALS["Brier Score"],
        ]
    )
    if all_goals_met:
        print("✓ All project goals met!")
    else:
        print("✗ Some project goals not met. See metrics above.")

    print("=" * 70 + "\n")


def save_model(model: RandomForestClassifier, filepath: str) -> None:
    """Save trained model to file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def print_feature_importances(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_k: int = 15,
) -> None:
    """
    Print ranked feature importances.
    """
    if not hasattr(model, "feature_importances_"):
        print("[warn] Model does not expose feature_importances_. Skipping ranking.")
        return

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        print("[warn] Mismatch between feature_importances_ and feature names; skipping ranking.")
        return

    order = np.argsort(importances)[::-1]
    top_k = min(top_k, len(feature_names))
    print("\nTop feature importances (Random Forest):")
    print("-" * 70)
    print(f"{'Rank':<6}{'Feature':<35}{'Importance':>12}")
    print("-" * 70)
    for rank, idx in enumerate(order[:top_k], start=1):
        print(f"{rank:<6}{feature_names[idx]:<35}{importances[idx]:>12.4f}")
    print("-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest model for liver cancer identification."
    )
    parser.add_argument(
        "--train",
        default="preprocessed/train.csv",
        help="Path to training CSV file (default: preprocessed/train.csv)",
    )
    parser.add_argument(
        "--test",
        default="preprocessed/test.csv",
        help="Path to test CSV file (default: preprocessed/test.csv)",
    )
    parser.add_argument(
        "--model-out",
        default=None,
        help="Path to save trained model (optional, as .pkl file)",
    )

    # Random Forest hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees in the forest (default: 500)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (default: None = expand until pure / min_samples)",
    )
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2,
        help="Minimum samples required to split an internal node (default: 2)",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples required to be at a leaf node (default: 1)",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Number of features considered at each split (default: 'sqrt')",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-balanced-classes",
        dest="balanced_classes",
        action="store_false",
        help="Disable class_weight='balanced' (default: balanced enabled)",
    )
    parser.set_defaults(balanced_classes=True)

    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        default=True,
        help="Optimize classification threshold using grid search (default: True)",
    )
    parser.add_argument(
        "--no-optimize-threshold",
        dest="optimize_threshold",
        action="store_false",
        help="Disable threshold optimization (use default 0.5)",
    )

    args = parser.parse_args()

    print("Loading training data...")
    train_df = load_data(args.train)
    print(f"  - Loaded {len(train_df)} training samples")

    print("Loading test data...")
    test_df = load_data(args.test)
    print(f"  - Loaded {len(test_df)} test samples")

    print("\nPreparing features and targets...")
    X_train, y_train = prepare_features_and_target(train_df, TARGET_COL)
    X_test, y_test = prepare_features_and_target(test_df, TARGET_COL)

    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    print(f"  - Feature columns: {list(X_train.columns)}")

    class_weight = "balanced" if args.balanced_classes else None

    print("\n" + "=" * 70)
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        class_weight=class_weight,
    )

    optimal_threshold = 0.5
    if args.optimize_threshold:
        print("\n" + "=" * 70)
        print("Finding optimal classification threshold (Random Forest)...")

        y_train_proba = model.predict_proba(X_train)[:, 1]

        precision_min, precision_max = PROJECT_GOALS["Precision"]
        recall_min, recall_max = PROJECT_GOALS["Recall"]

        optimal_threshold, best_metrics = find_optimal_threshold(
            y_train.values,
            y_train_proba,
            precision_range=(precision_min, precision_max),
            recall_range=(recall_min, recall_max),
        )

        print(f"  - Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
        if best_metrics is not None:
            print(f"  - Expected recall: {best_metrics[0]:.4f}")
            print(f"  - Expected precision: {best_metrics[1]:.4f}")

    print("\n" + "=" * 70)
    print("Evaluating Random Forest model on test set...")
    metrics, y_pred, y_pred_proba = evaluate_model(
        model, X_test, y_test, threshold=optimal_threshold
    )

    print_results(metrics, y_test)

    # Feature importance ranking to show key medical factors
    print_feature_importances(model, list(X_train.columns))

    if args.model_out:
        save_model(model, args.model_out)


if __name__ == "__main__":
    main()


