"""
Compact feature selection + PCA demo for processed liver dataset.
Usage:
  python3 feature_selection.py --input <input_csv_here> --out <output_directory_here>
"""
import os
import argparse
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_df(path):
    df = pd.read_csv(path)
    return df

def corr_matrix(df, out_path=None):
    corr = df.corr()
    if out_path:
        corr.to_csv(out_path)
    return corr

def compute_vif(X):
    # Requires statsmodels installed. X must be a DataFrame of numerical features.
    vif_df = pd.DataFrame({
        "feature": X.columns,
        "vif": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_df.sort_values("vif", ascending=False)

def select_kbest(X, y, k=5):
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X, y)
    mask = skb.get_support()
    return X.columns[mask].tolist(), skb

def l1_logistic_select(X, y, C=1.0, penalty='l1', solver='saga', max_iter=5000):
    # Use L1 logistic for feature selection. Returns non-zero coef features.
    clf = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter)
    clf.fit(X, y)
    coefs = np.abs(clf.coef_).ravel()
    mask = coefs > 1e-6
    selected = X.columns[mask].tolist()
    return selected, clf

def rfe_select(X, y, n_features_to_select=5, estimator=None):
    if estimator is None:
        estimator = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=2000)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    return selected, rfe

def pca_summary(X, n_components=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    pca.fit(Xs)
    return pca, scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="./feature_selection")
    parser.add_argument("--k", type=int, default=6, help="k for SelectKBest")
    parser.add_argument("--rfe-k", type=int, default=6, help="n features for RFE")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_df(args.input)

    target = "liver_cancer"
    if target not in df.columns:
        raise SystemExit("Target column 'liver_cancer' not found")

    X = df.drop(columns=[target])
    y = df[target]

    # 1) Correlation matrix
    corr = corr_matrix(X, out_path=os.path.join(args.out, "correlation_matrix.csv"))
    print("Saved correlation matrix.")

    # 2) VIF (requires statsmodels)
    try:
        vif = compute_vif(X)
        vif.to_csv(os.path.join(args.out, "vif.csv"), index=False)
        print("Computed VIF; saved to vif.csv")
        print(vif.head())
    except Exception as e:
        print("VIF computation skipped (needs statsmodels). Error:", e)

    # 3) SelectKBest (ANOVA F-test)
    kbest_features, skb = select_kbest(X, y, k=args.k)
    print(f"SelectKBest (k={args.k}) selected: {kbest_features}")

    # 4) L1 Logistic selection (tune C if needed)
    # Scale features before L1 logistic
    scaler = StandardScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    l1_feats, l1_clf = l1_logistic_select(Xs, y, C=0.5)
    print("L1 logistic selected:", l1_feats)

    # 5) RFE
    try:
        rfe_feats, rfe = rfe_select(Xs, y, n_features_to_select=args.rfe_k)
        print(f"RFE selected ({args.rfe_k}):", rfe_feats)
    except Exception as e:
        print("RFE failed:", e)

    # 6) PCA for comparison
    pca, pca_scaler = pca_summary(X, n_components=min(6, X.shape[1]))
    explained = pca.explained_variance_ratio_
    print("PCA explained variance ratios:", explained)

    # Save selected feature lists
    pd.Series(kbest_features).to_csv(os.path.join(args.out, "selectkbest.txt"), index=False, header=False)
    pd.Series(l1_feats).to_csv(os.path.join(args.out, "l1_selected.txt"), index=False, header=False)
    try:
        pd.Series(rfe_feats).to_csv(os.path.join(args.out, "rfe_selected.txt"), index=False, header=False)
    except:
        pass

    # Optionally save reduced datasets (example: keep L1-selected features)
    if len(l1_feats) > 0:
        df[l1_feats + [target]].to_csv(os.path.join(args.out, "data_l1_selected.csv"), index=False)
        print("Saved dataset with L1-selected features.")

if __name__ == "__main__":
    main()
