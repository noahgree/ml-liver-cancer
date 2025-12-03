"""
Support Vector Machine model for liver cancer prediction.
By: Benjamin Gunasekera
Referenced logistic_regression.py
"""
from logistic_regression import load_data, prepare_features_and_target, save_model
from sklearn.svm import SVC
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    recall_score,
    precision_score,
    confusion_matrix,
)

# Project Goals (Expected Results)
# Ranges are specified as (min, max) tuples
PROJECT_GOALS = {
    "AUROC": (0.80, 0.99),      # Range: 0.80–0.99
    "Recall": (0.80, 0.90),     # Range: 0.80–0.90
    "Precision": (0.30, 0.50),  # Range: 0.30–0.50
    "Brier Score": (0.00, 0.20)         # Threshold: < 0.20
}

def svm_train_model(X_train, y_train, kernel: str = "rbf", C: int = 1,
                max_iter: int = 1000, random_state: int = None):
    """
    Train SVM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        kernel: Specifies the kernel type to be used in the algorithm.
        C: Inverse of regularization strength
        gamma: Kernel coefficient ('scale', 'auto', or float)
        max_iter: Maximum iterations
        random_state: Random seed for reproducibility
        
    Returns:
        Trained SVC model
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
			    'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
    model = GridSearchCV(SVC(kernel=kernel, probability=True, random_state=random_state, max_iter=max_iter, class_weight='balanced'), param_grid, refit=True, verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def find_optimal_threshold(
    y_true,
    y_pred_proba,
    precision_range=(0.30, 0.50),
    recall_range=(0.80, 0.90),
):
    """
    Grid-search classification threshold.

    Design goal: MAXIMIZE RECALL (minimize false negatives) for medical screening.
    Precision must be above minimum, but exceeding precision/recall upper bounds
    is acceptable if it means catching more cancer cases.
    
    Priority order:
    1. Maximize recall (catch all cancer cases)
    2. Keep precision above minimum threshold
    3. If multiple thresholds have similar recall, prefer higher precision
    """
    import numpy as np
    from sklearn.metrics import recall_score, precision_score

    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    best_threshold = 0.5
    best_recall = -1.0
    best_precision = -1.0
    best_metrics = None

    precision_min, _ = precision_range  # Only use minimum, ignore maximum
    recall_min, _ = recall_range       # Only use minimum, ignore maximum

    for threshold in candidate_thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if y_pred.sum() == 0:
            continue

        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)

        # Must meet minimum precision (avoid too many false alarms)
        if precision < precision_min:
            continue

        # Must meet minimum recall (catch at least 80% of cases)
        if recall < recall_min:
            continue

        # Primary: maximize recall (minimize false negatives)
        # Secondary: for same recall, prefer higher precision
        if (recall > best_recall) or (np.isclose(recall, best_recall) and precision > best_precision):
            best_recall = recall
            best_precision = precision
            best_threshold = threshold
            best_metrics = (recall, precision)

    return best_threshold, best_metrics

def svm_evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Compute metrics
    metrics = {
        "AUROC": roc_auc_score(y_test, y_pred_proba),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Brier Score": brier_score_loss(y_test, y_pred_proba),
        "Accuracy": (y_pred == y_test).mean(),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Threshold": threshold,
    }

    return metrics, y_pred, y_pred_proba

def svm_print_results(metrics, y_test, y_pred, y_pred_proba):
    """Print evaluation results and compare against project goals."""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTest Set Size: {len(y_test)}")
    print(f"Positive Cases: {y_test.sum()}")
    print(f"Negative Cases: {(y_test == 0).sum()}")

    threshold_val = metrics.get("Threshold", 0.5)
    if abs(threshold_val - 0.5) < 0.001:
        print(f"\nClassification Threshold: {threshold_val:.4f} (default)")
    else:
        print(
            f"\nClassification Threshold: {threshold_val:.4f} "
            "(optimized, default: 0.5)"
        )
    
    print("\n" + "-"*70)
    print("METRICS")
    print("-"*70)
    
    # Print each metric with goal comparison
    for metric_name, goal_value in PROJECT_GOALS.items():
        value = metrics[metric_name]
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
    brier_score_min, brier_score_max = PROJECT_GOALS["Brier Score"]
    
    all_goals_met = all([
        auroc_min <= metrics["AUROC"] <= auroc_max,
        recall_min <= metrics["Recall"] <= recall_max,
        precision_min <= metrics["Precision"] <= precision_max,
        brier_score_min <= metrics["Brier Score"] <= brier_score_max,
    ])
    
    if all_goals_met:
        print("✓ All project goals met!")
    else:
        print("✗ Some project goals not met. See metrics above.")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate SVM model for liver cancer identification."
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
        "--model-out",
        default=None,
        help="Path to save trained model (optional, as .pkl file)"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        help="Specifies the kernel type to be used in the algorithm (default: rbf)"
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
        default=-1,
        help="Maximum iterations for solver (default: -1)"
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

    args = parser.parse_args()
    
    # Load data
    print("Loading training data...")
    train_df = load_data(args.train)
    print(f"  - Loaded {len(train_df)} training samples")
    
    print("Loading test data...")
    test_df = load_data(args.test)
    print(f"  - Loaded {len(test_df)} test samples")
    
    # Prepare features and targets
    TARGET_COL = "liver_cancer"
    print("\nPreparing features and targets...")
    X_train, y_train = prepare_features_and_target(train_df, TARGET_COL)
    X_test, y_test = prepare_features_and_target(test_df, TARGET_COL)
    
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    print(f"  - Feature columns: {list(X_train.columns)}")
    
    # Train model
    print("\n" + "="*70)
    model = svm_train_model(
        X_train, y_train,
        kernel=args.kernel,
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    print("\n")
    
    optimal_threshold = 0.5
    if args.optimize_threshold:
        print("\n" + "=" * 70)
        print("Finding optimal classification threshold (SVM)...")

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

    # Evaluate model
    print("\n" + "="*70)
    print("Evaluating model on test set...")
    metrics, y_pred, y_pred_proba = svm_evaluate_model(model, X_test, y_test, threshold=optimal_threshold)
    
    # Print results
    svm_print_results(metrics, y_test, y_pred, y_pred_proba)
    
    # Save model if requested
    if args.model_out:
        save_model(model, args.model_out)
    
    return model, metrics


if __name__ == "__main__":
    main()