"""
Data Preprocessing Pipeline.
Handles missing value imputation, encoding, feature scaling, and train-test split.
Returns processed data along with a fitted pipeline for prediction reuse.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str, problem_type: str = None):
    """
    Build and execute the full preprocessing pipeline.

    Steps:
      1. Separate features and target
      2. Impute missing values (mean for numeric, most_frequent for categorical)
      3. Encode categorical features (one-hot)
      4. Scale numeric features (StandardScaler)
      5. Encode target if classification
      6. 80/20 train-test split

    Returns:
        dict with keys:
          X_train, X_test, y_train, y_test,
          feature_names, preprocessing_pipeline, label_encoder (if classification),
          steps_log (list of preprocessing step descriptions)
    """
    steps_log = []

    # ── 1. Separate features and target ──
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    steps_log.append(f"Separated target column '{target_column}' from {X.shape[1]} features.")

    # ── Drop High-Cardinality & Unique Identifier Columns ──
    dropped_cols = []
    for col in X.columns:
        n_unique = X[col].nunique()
        # Stricter memory limit:
        # drop categorical/text columns with > 20 unique values or high unique ratio.
        if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col].dtype):
            if n_unique > 20 or (n_unique / len(X) > 0.05 and len(X) > 50):
                dropped_cols.append(col)
        # Drop numeric integer columns that are purely unique keys (like indices, row IDs)
        elif pd.api.types.is_integer_dtype(X[col].dtype):
            # If it's a primary key/unique ID column (100% unique values)
            # and has 'id' or 'key' in name.
            if n_unique == len(X) and any(kw in col.lower() for kw in ["id", "key", "index", "no"]):
                dropped_cols.append(col)

    if dropped_cols:
        X = X.drop(columns=dropped_cols)
        steps_log.append(
            "Dropped high-cardinality/identifier columns to prevent memory "
            f"overflow: {', '.join(dropped_cols)}"
        )

    # ── 2. Identify column types ──
    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    steps_log.append(
        f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features."
    )

    # ── 3. Handle missing values ──
    missing_before = int(X.isna().sum().sum())
    if missing_before > 0:
        if numeric_cols:
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        if categorical_cols:
            for col in categorical_cols:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
        steps_log.append(
            f"Imputed {missing_before} missing values (mean for numeric, mode for categorical)."
        )
    else:
        steps_log.append("No missing values found in features.")

    # Handle target missing values
    target_missing = int(y.isna().sum())
    if target_missing > 0:
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        steps_log.append(f"Dropped {target_missing} rows with missing target values.")

    # ── 4. Encode categorical features ──
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        steps_log.append(
            f"One-Hot Encoded {len(categorical_cols)} categorical columns → "
            f"{X.shape[1]} total features."
        )

    # ── Protect Against Feature Explosion ──
    if X.shape[1] > 500:
        steps_log.append(
            f"Feature count too high ({X.shape[1]}). "
            "Truncating to 500 to prevent memory exhaustion."
        )
        # Only keep top 500 features. You would ideally use feature selection
        # here, but for preprocessor we just keep the first 500.
        # For simplicity, we just keep the first 500
        X = X.iloc[:, :500]

    # ── 5. Encode target (for classification) ──
    label_encoder = None
    if problem_type in ("binary", "multiclass"):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target_column)
        steps_log.append(f"Label Encoded target column. Classes: {list(label_encoder.classes_)}")
    elif problem_type is None and (
        y.dtype == "object" or pd.api.types.is_categorical_dtype(y.dtype)
    ):
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=target_column)
        steps_log.append(f"Label Encoded target column. Classes: {list(label_encoder.classes_)}")

    # ── 6. Scale numeric features ──
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)
    steps_log.append(f"Applied StandardScaler to all {len(feature_names)} features.")

    # ── 7. Train-Test Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None
    )
    steps_log.append(
        f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples (80/20)."
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "steps_log": steps_log,
        "X_raw": X,  # Before scaling, for EDA
    }


def preprocess_for_prediction(input_data: dict, feature_names: list, scaler, label_encoder=None):
    """
    Preprocess a single data point (dict) for prediction.
    Creates a DataFrame, applies one-hot encoding to match training features, and scales.
    """
    df = pd.DataFrame([input_data])

    # One-hot encode any categorical columns
    df = pd.get_dummies(df)

    # Align columns with training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
    return df_scaled
