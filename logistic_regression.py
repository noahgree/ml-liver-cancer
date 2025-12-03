"""
Logistic Regression model for liver cancer identification.

Trains a Logistic Regression model on preprocessed training data and evaluates
it using AUROC, Recall, Precision, and Brier Score metrics.

Expected Results:
  - AUROC: 0.75–0.85
  - Recall: 0.80–0.90
  - Precision: 0.30–0.50
  - Brier Score: < 0.20

Usage:
  python3 logistic_regression.py --train preprocessed/train.csv --test preprocessed/test.csv
  python3 logistic_regression.py --train <train_csv> --test <test_csv>
"""

import argparse
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

TARGET_COL = "liver_cancer"

# Project Goals (Expected Results)
# Ranges are specified as (min, max) tuples
PROJECT_GOALS = {
    "AUROC": (0.75, 0.85),      # Range: 0.75–0.85
    "Recall": (0.80, 0.90),     # Range: 0.80–0.90
    "Precision": (0.30, 0.50),  # Range: 0.30–0.50
    "Brier Score": 0.20         # Threshold: < 0.20
}


def load_data(csv_path: str):
    """Load CSV data and return DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_features_and_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Separate features and target from DataFrame.

    Args:
        df: DataFrame with features and target column
        target_col: Name of the target column

    Returns:
        X: Feature matrix (DataFrame)
        y: Target vector (Series)
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def train_model(X_train, y_train, C=1.0, max_iter=1000, solver='lbfgs', random_state=42):
    """
    Train Logistic Regression model.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Inverse of regularization strength (default: 1.0)
        max_iter: Maximum iterations (default: 1000)
        solver: Solver algorithm (default: 'lbfgs')
        random_state: Random seed for reproducibility

    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'
    )

    print(f"Training Logistic Regression model...")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - C (regularization): {C}")
    print(f"  - Solver: {solver}")

    model.fit(X_train, y_train)

    return model


def find_optimal_threshold(y_true, y_pred_proba, precision_range=(0.30, 0.50),
                           recall_range=(0.80, 0.90)):
    """
    Find optimal classification threshold using grid search.

    This uses grid search because:
    - Threshold is a discrete parameter (not differentiable)
    - We're optimizing discrete metrics (precision/recall)
    - We need to find a value that satisfies constraints

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        precision_range: Target precision range (min, max)
        recall_range: Target recall range (min, max)

    Returns:
        Optimal threshold value that best satisfies the constraints
    """
    # Grid search: try thresholds from 0.1 to 0.9 in steps of 0.01
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5  # Default
    best_score = -1
    best_metrics = None

    precision_min, precision_max = precision_range
    recall_min, recall_max = recall_range

    for threshold in candidate_thresholds:
        # Convert probabilities to binary predictions using this threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Skip if no positive predictions (would cause division by zero)
        if y_pred.sum() == 0:
            continue

        # Calculate metrics
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        # Check if within target ranges
        recall_in_range = recall_min <= recall <= recall_max
        precision_in_range = precision_min <= precision <= precision_max

        # Score: prioritize metrics within target ranges
        if recall_in_range and precision_in_range:
            # Both in range - ideal! Maximize recall (since we want to catch cases)
            score = recall * 1000 + precision  # Large weight on recall
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = (recall, precision)
        elif recall_in_range:
            # Recall in range, precision not - try to get precision closer
            precision_penalty = abs(precision - (precision_min + precision_max) / 2)
            score = recall * 500 - precision_penalty * 100
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = (recall, precision)
        elif precision_in_range:
            # Precision in range, recall not - try to get recall closer
            recall_penalty = abs(recall - (recall_min + recall_max) / 2)
            score = precision * 500 - recall_penalty * 100
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = (recall, precision)
        else:
            # Neither in range - minimize distance from both ranges
            recall_distance = min(abs(recall - recall_min), abs(recall - recall_max))
            if recall < recall_min:
                recall_distance = recall_min - recall
            elif recall > recall_max:
                recall_distance = recall - recall_max

            precision_distance = min(abs(precision - precision_min), abs(precision - precision_max))
            if precision < precision_min:
                precision_distance = precision_min - precision
            elif precision > precision_max:
                precision_distance = precision - precision_max

            # Penalty score (lower is better)
            penalty = recall_distance * 2 + precision_distance  # Weight recall more
            score = -penalty

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = (recall, precision)

    return best_threshold, best_metrics


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model and compute all required metrics.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary containing all metrics
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    # Convert probabilities to binary predictions using threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Brier Score": brier_score_loss(y_test, y_pred_proba)
    }

    # Additional metrics for reporting
    metrics["Accuracy"] = (y_pred == y_test).mean()
    metrics["Confusion Matrix"] = confusion_matrix(y_test, y_pred)
    metrics["Threshold"] = threshold

    return metrics, y_pred, y_pred_proba


def print_results(metrics, y_test, y_pred):
    """Print evaluation results and compare against project goals."""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)

    print(f"\nTest Set Size: {len(y_test)}")
    print(f"Positive Cases: {y_test.sum()}")
    print(f"Negative Cases: {(y_test == 0).sum()}")

    # Print threshold information
    if "Threshold" in metrics:
        threshold_val = metrics['Threshold']
        if abs(threshold_val - 0.5) < 0.001:
            print(f"\nClassification Threshold: {threshold_val:.4f} (default)")
        else:
            print(f"\nClassification Threshold: {threshold_val:.4f} (optimized, default: 0.5)")

    print("\n" + "-"*70)
    print("METRICS")
    print("-"*70)

    # Print each metric with goal comparison
    for metric_name, goal_value in PROJECT_GOALS.items():
        value = metrics[metric_name]
        if metric_name == "Brier Score":
            # Brier Score: lower is better (should be < threshold)
            meets_goal = value < goal_value
            comparison = "✓" if meets_goal else "✗"
            print(f"{metric_name:15s}: {value:.4f} (goal: < {goal_value:.2f}) {comparison}")
        else:
            # Other metrics: check if within range (min, max)
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

    # Confusion Matrix
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    cm = metrics["Confusion Matrix"]
    print(f"                Predicted")
    print(f"              Negative  Positive")
    print(f"Actual Negative   {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"       Positive   {cm[1,0]:4d}      {cm[1,1]:4d}")

    # Calculate derived metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")

    print("\n" + "-"*70)
    print("GOAL SUMMARY")
    print("-"*70)

    # Check if all goals are met
    auroc_min, auroc_max = PROJECT_GOALS["AUROC"]
    recall_min, recall_max = PROJECT_GOALS["Recall"]
    precision_min, precision_max = PROJECT_GOALS["Precision"]

    all_goals_met = all([
        auroc_min <= metrics["AUROC"] <= auroc_max,
        recall_min <= metrics["Recall"] <= recall_max,
        precision_min <= metrics["Precision"] <= precision_max,
        metrics["Brier Score"] < PROJECT_GOALS["Brier Score"]
    ])

    if all_goals_met:
        print("✓ All project goals met!")
    else:
        print("✗ Some project goals not met. See metrics above.")

    print("="*70 + "\n")


def save_model(model, filepath: str = "models/logreg.pkl"):
    """Save trained model to file as a .pkl file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Logistic Regression model for liver cancer identification."
    )
    parser.add_argument(
        "--train",
        default="preprocessed/train.csv",
        help="Path to training CSV file (default: preprocessed/train.csv)"
    )
    parser.add_argument(
        "--test",
        default="preprocessed/test.csv",
        help="Path to test CSV file (default: preprocessed/test.csv)"
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse of regularization strength (default: 1.0)"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for solver (default: 1000)"
    )
    parser.add_argument(
        "--solver",
        default="lbfgs",
        choices=["lbfgs", "liblinear", "saga", "newton-cg"],
        help="Solver algorithm (default: lbfgs)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--optimize-threshold",
        action="store_true",
        default=True,
        help="Optimize classification threshold using grid search (default: True)"
    )
    parser.add_argument(
        "--no-optimize-threshold",
        dest="optimize_threshold",
        action="store_false",
        help="Disable threshold optimization (use default 0.5)"
    )

    args = parser.parse_args()

    # Load data
    print("Loading training data...")
    train_df = load_data(args.train)
    print(f"  - Loaded {len(train_df)} training samples")

    print("Loading test data...")
    test_df = load_data(args.test)
    print(f"  - Loaded {len(test_df)} test samples")

    # Prepare features and targets
    print("\nPreparing features and targets...")
    X_train, y_train = prepare_features_and_target(train_df, TARGET_COL)
    X_test, y_test = prepare_features_and_target(test_df, TARGET_COL)

    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    print(f"  - Feature columns: {list(X_train.columns)}")

    # Train model
    print("\n" + "="*70)
    model = train_model(
        X_train, y_train,
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
        random_state=args.random_state
    )

    # Find optimal threshold if requested
    optimal_threshold = 0.5
    if args.optimize_threshold:
        print("\n" + "="*70)
        print("Finding optimal classification threshold...")

        # Get probabilities on training set to find optimal threshold
        y_train_proba = model.predict_proba(X_train)[:, 1]

        # Get target ranges from PROJECT_GOALS
        precision_min, precision_max = PROJECT_GOALS["Precision"]
        recall_min, recall_max = PROJECT_GOALS["Recall"]

        optimal_threshold, best_metrics = find_optimal_threshold(
            y_train, y_train_proba,
            precision_range=(precision_min, precision_max),
            recall_range=(recall_min, recall_max)
        )

        print(f"  - Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
        if best_metrics:
            print(f"  - Expected recall: {best_metrics[0]:.4f}")
            print(f"  - Expected precision: {best_metrics[1]:.4f}")

    # Evaluate model
    print("\n" + "="*70)
    print("Evaluating model on test set...")
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, threshold=optimal_threshold)

    # Print results
    print_results(metrics, y_test, y_pred)

    save_model(model, "models/logreg.pkl")

    return model, metrics


if __name__ == "__main__":
    main()
