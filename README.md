# ml-project-group-129
Repository for the Fall 2025 CS 4641 group project for group 129.

## Directory and File Structure

### Root Directory Files

/full_dataset.csv: Raw liver cancer dataset with unprocessed features and labels

/requirements.txt: Python package dependencies required to run the project

/preprocess.py: Data preprocessing script that encodes/normalizes features and performs train-test split. Accepts input CSV and outputs full_processed.csv, train.csv, and test.csv

/feature_selection.py: Feature selection and dimensionality reduction module using SelectKBest, RFE, PCA, and correlation analysis

/logistic_regression.py: Logistic Regression classifier implementation for liver cancer prediction with evaluation metrics (AUROC, Recall, Precision, Brier Score)

/random_forest.py: Random Forest classifier implementation that captures non-linear feature interactions and provides feature importance scores

/svm.py: Support Vector Machine classifier for liver cancer prediction with grid search hyperparameter tuning

/visualize_results.py: Generates publication-quality plots for trained model results including ROC curves, confusion matrices, and performance visualizations

### Directories

/models/: Trained model artifacts and result figures
  /logreg.pkl: Serialized Logistic Regression model
  /randforest.pkl: Serialized Random Forest model
  /svm.pkl: Serialized Support Vector Machine model
  /logreg_figures/: Directory containing Logistic Regression visualization outputs
  /randforest_figures/: Directory containing Random Forest visualization outputs

/preprocessed/: Processed training and testing datasets
  /full_processed.csv: Complete processed dataset with all rows and encoded/normalized columns
  /train.csv: Training subset (default 80% of data) without labels for model training
  /test.csv: Testing subset (default 20% of data) with labels for model evaluation

/__pycache__/: Python bytecode cache directory (auto-generated)

/.git/: Git version control metadata

/.venv/: Python virtual environment (local dependencies)
