"""
Preprocess liver cancer dataset:
- Encodes/normalizes features for logistic regression
- Writes:
  1) full_processed.csv (all processed rows/columns)
  2) train.csv (top % without labels for training)
  3) test.csv (bottom % with labels for testing)
- Optional: stratified split via --mode stratified producing train.csv and test.csv

Usage:
  python3 preprocess.py --input <input_csv_here, default="full_dataset.csv"> --out <output_directory_here> --fraction 0.8
  python3 preprocess.py --input <input_csv_here, default="full_dataset.csv"> --out ./out --fraction 0.8 --mode stratified --seed 42
"""

import argparse
import os
from re import split
import sys
import math
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "gender",
    "bmi",
    "alcohol_consumption",
    "smoking_status",
    "hepatitis_b",
    "hepatitis_c",
    "liver_function_score",
    "alpha_fetoprotein_level",
    "cirrhosis_history",
    "family_history_cancer",
    "physical_activity_level",
    "diabetes",
    "liver_cancer",
]

# ---------------------------
# Robust parsers/encoders
# ---------------------------

def _to_str(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s != "" else None
    return str(v).strip()

def parse_bool(v):
    s = _to_str(v)
    if s is None:
        return np.nan
    s_low = s.lower()
    true_set = {"true", "t", "yes", "y", "1", "pos", "positive"}
    false_set = {"false", "f", "no", "n", "0", "neg", "negative"}
    if s_low in true_set:
        return 1
    if s_low in false_set:
        return 0
    # Try numeric
    try:
        num = float(s)
        if math.isnan(num):
            return np.nan
        return 1 if num != 0 else 0
    except Exception:
        return np.nan

def encode_gender(v):
    s = _to_str(v)
    if s is None:
        return np.nan
    s_low = s.lower()
    if s_low in {"female", "f", "woman", "w"}:
        return 0
    if s_low in {"male", "m", "man"}:
        return 1
    # Unknown/other -> NaN (will be imputed if needed)
    return np.nan

def encode_smoking_status(v):
    # Ordinal: never=0, former=1, current=2
    s = _to_str(v)
    if s is None:
        return np.nan
    s_low = s.lower()
    mapping = {
        "never": 0,
        "former": 1,
        "current": 2,
    }
    return mapping.get(s_low, np.nan)

def encode_physical_activity(v):
    # Ordinal: low=0, moderate=1, high=2
    s = _to_str(v)
    if s is None:
        return np.nan
    s_low = s.lower()
    mapping = {
        "low": 0,
        "moderate": 1,
        "high": 2,
    }
    return mapping.get(s_low, np.nan)

def encode_alcohol_consumption(v):
   # Ordinal: never=0, occasional=1, regular=2
   s = _to_str(v)
   if s is None:
      return np.nan
   s_low = s.lower()
   mapping = {
      "never": 0,
      "occasional": 1,
      "regular": 2,
   }
   return mapping.get(s_low, np.nan)

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def zscore(series: pd.Series) -> pd.Series:
    # Population std (ddof=0) to mirror StandardScaler
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std

# ---------------------------
# Processing pipeline
# ---------------------------

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy
    df = df.copy()

    # Normalize column names to exact expected (strip spaces)
    rename_map = {c: c.strip() for c in df.columns}
    df.rename(columns=rename_map, inplace=True)

    # Validate
    validate_columns(df)

    # Encoders / coercions
    df["gender"] = df["gender"].apply(encode_gender)

    df["smoking_status"] = df["smoking_status"].apply(encode_smoking_status)
    df["physical_activity_level"] = df["physical_activity_level"].apply(encode_physical_activity)
    df["alcohol_consumption"] = df["alcohol_consumption"].apply(encode_alcohol_consumption)

    for col in ["hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "diabetes"]:
        df[col] = df[col].apply(parse_bool)

    # Numeric coercion (exclude alcohol_consumption: it's ordinal encoded now)
    for col in ["bmi", "liver_function_score", "alpha_fetoprotein_level"]:
        df[col] = coerce_numeric(df[col])

    # Target
    df["liver_cancer"] = df["liver_cancer"].apply(parse_bool)

    # Impute numeric with median
    numeric_cols = ["gender", "bmi", "alcohol_consumption", "smoking_status",
                    "liver_function_score", "alpha_fetoprotein_level",
                    "physical_activity_level",
                    "hepatitis_b", "hepatitis_c", "cirrhosis_history",
                    "family_history_cancer", "diabetes"]
    target_col = "liver_cancer"

    for col in numeric_cols:
        if col not in df.columns:
            continue
        if df[col].dtype.kind not in "biufc":
            df[col] = coerce_numeric(df[col])
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # Impute target with mode if any missing (rare)
    if df[target_col].isna().any():
        mode = df[target_col].mode(dropna=True)
        if len(mode) > 0:
            df[target_col] = df[target_col].fillna(mode.iloc[0])
        else:
            df[target_col] = df[target_col].fillna(0)

    # Scale selected continuous features (do NOT scale ordinal alcohol_consumption)
    to_scale = ["bmi", "liver_function_score", "alpha_fetoprotein_level"]
    for col in to_scale:
        if col in df.columns:
            df[col] = zscore(df[col])

    # Ensure integer dtype for binary/ordinal where appropriate (includes alcohol_consumption)
    int_like_cols = ["gender", "smoking_status", "physical_activity_level", "alcohol_consumption",
                     "hepatitis_b", "hepatitis_c", "cirrhosis_history",
                     "family_history_cancer", "diabetes", "liver_cancer"]
    for col in int_like_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    # Column order: features then target at end
    feature_cols = [c for c in numeric_cols if c in df.columns]
    ordered_cols = feature_cols + [target_col]
    return df[ordered_cols]

# ---------------------------
# Splitting helpers
# ---------------------------

def stratified_split(df: pd.DataFrame, target_col: str, split_fraction, seed=42):
    rng = np.random.RandomState(seed)
    y = df[target_col].values
    idx = np.arange(len(df))

    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n_test = int(round(len(cls_idx) * (1 - split_fraction)))
        test_idx.extend(cls_idx[:n_test].tolist())
        train_idx.extend(cls_idx[n_test:].tolist())

    train_df = df.loc[train_idx].sort_index()
    test_df = df.loc[test_idx].sort_index()
    return train_df, test_df

def split_data_into_train_and_test(df: pd.DataFrame, split_fraction: float):
    """
    Split by row order:
      - Train = top fraction of rows
      - Test  = bottom fraction of rows
    """
    n = len(df)
    split_idx = int(math.ceil(split_fraction * n))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess liver cancer CSV for logistic regression.")
    parser.add_argument("--input", default="full_dataset.csv", help="Path to input CSV")
    parser.add_argument("--out", default="./preprocessed", help="Directory to write outputs")
    parser.add_argument("--fraction", type=float, default=0.8, help="Percent of rows to use for training.")
    parser.add_argument("--mode", choices=["requested", "stratified"], default="requested",
                        help='requested: normal split based on preordered dataset; '
                             'stratified: reproducible split using a seed.')
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified split")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        raw = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read input CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        processed = preprocess(raw)
    except Exception as e:
        print(f"Preprocessing error: {e}", file=sys.stderr)
        sys.exit(1)

    # 1) Full processed CSV
    full_path = os.path.join(args.out, "full_processed.csv")
    processed.to_csv(full_path, index=False)
    print(f"Wrote: {full_path} (rows={len(processed)}, positives={(processed['liver_cancer']==1).sum()}, negatives={(processed['liver_cancer']==0).sum()})")

    # 2) and 3) Split outputs
    if args.mode == "requested":
        train_df, test_df = split_data_into_train_and_test(processed, args.fraction)
        train_path = os.path.join(args.out, "train.csv")
        test_path = os.path.join(args.out, "test.csv")
    else:
        train_df, test_df = stratified_split(processed, "liver_cancer", args.fraction, seed=args.seed)
        train_path = os.path.join(args.out, "train_stratified_{args.seed}.csv")
        test_path = os.path.join(args.out, "test_stratified_{args.seed}.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Wrote: {train_path} (rows={len(train_df)})")
    print(f"Wrote: {test_path} (rows={len(test_df)})")

if __name__ == "__main__":
    main()
