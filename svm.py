"""
Support Vector Machine model for liver cancer prediction.
By: Benjamin Gunasekera
"""
from logistic_regression import load_data, prepare_features_and_target, save_model
from sklearn.svm import SVC
import argparse
from sklearn.metrics import (
    recall_score,
    precision_score,
    confusion_matrix,
)

def svm_train_model(X_train, y_train, kernel: str = "rbf", C: int = 1,
                max_iter: int = 1000, random_state: int = None):
    """
    Train Logistic Regression model.
    
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
    model = SVC(C=C, kernel=kernel, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def svm_optimize_model():
    raise NotImplementedError

def svm_evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Accuracy": (y_pred == y_test).mean(),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics, y_pred

def svm_print_results(metrics, y_test, y_pred):
    """Print evaluation results and compare against project goals."""
    """ Modified from logistic_regression.py print results (didn't want to update so we now have some ugly code)"""
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nTest Set Size: {len(y_test)}")
    print(f"Positive Cases: {y_test.sum()}")
    print(f"Negative Cases: {(y_test == 0).sum()}")
    
    print("\n" + "-"*70)
    print("METRICS")
    print("-"*70)


    # Project Goals (Expected Results)
    # Ranges are specified as (min, max) tuples
    PROJECT_GOALS = {
        "Recall": (0.80, 0.90),     # Range: 0.80–0.90
        "Precision": (0.30, 0.50),  # Range: 0.30–0.50
    }
    
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
    recall_min, recall_max = PROJECT_GOALS["Recall"]
    precision_min, precision_max = PROJECT_GOALS["Precision"]
    
    all_goals_met = all([
        recall_min <= metrics["Recall"] <= recall_max,
        precision_min <= metrics["Precision"] <= precision_max,
    ])
    
    if all_goals_met:
        print("✓ All project goals met!")
    else:
        print("✗ Some project goals not met. See metrics above.")
    
    print("="*70 + "\n")


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

    # Evaluate model
    print("\n" + "="*70)
    print("Evaluating model on test set...")
    metrics, y_pred = svm_evaluate_model(model, X_test, y_test)
    
    # Print results
    svm_print_results(metrics, y_test, y_pred)
    
    # Save model if requested
    if args.model_out:
        save_model(model, args.model_out)
    
    return model, metrics


if __name__ == "__main__":
    main()