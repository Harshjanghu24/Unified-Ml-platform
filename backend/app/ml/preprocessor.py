"""
Data Preprocessing Pipeline.
Handles missing value imputation, duplicate removal, encoding, feature scaling,
and outlier handling in both automatic and manual configurations.
Includes an adaptive encoding system to prevent dimensional explosion and out-of-memory errors.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler


def get_dataset_stats_dict(df: pd.DataFrame) -> dict:
    """Helper to compute stats dictionary for a DataFrame."""
    return {
        "num_rows": len(df),
        "num_cols": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


def is_identifier_column(col_name: str, df: pd.DataFrame) -> bool:
    """Detect likely identifier/index columns based on naming patterns."""
    name_lower = col_name.lower()
    id_keywords = ["id", "uuid", "key", "index", "no", "patient_id", "customer_id", "record_id", "transaction_id", "employee_id", "user_id"]
    
    # Check exact word matches or prefix/suffix patterns
    if (
        name_lower == "id"
        or name_lower == "_id"
        or name_lower.startswith("id_")
        or name_lower.endswith("_id")
        or name_lower.endswith("id")
        or any(kw in name_lower for kw in id_keywords)
    ):
        return True
    return False


def get_cardinality_recommendations(df: pd.DataFrame, target_column: str) -> list[dict]:
    """Analyze categorical columns and recommend encoding strategies based on cardinality."""
    recommendations = []
    for col in df.columns:
        if col == target_column:
            continue
        
        # Check if column is categorical/text
        if (
            df[col].dtype == "object"
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[col].dtype)
        ):
            n_unique = df[col].nunique()
            is_id = is_identifier_column(col, df)
            
            # Categorize cardinality level
            if n_unique <= 20:
                card_level = "Low"
                rec = "onehot"
            elif n_unique <= 1000:
                card_level = "Medium"
                rec = "label"
            else:
                card_level = "High"
                rec = "drop"
                
            # Identifiers should always be dropped
            if is_id:
                rec = "drop"
                
            recommendations.append({
                "column": col,
                "nunique": int(n_unique),
                "is_identifier": is_id,
                "cardinality_level": card_level,
                "recommendation": rec
            })
    return recommendations


def handle_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """IQR method for outlier removal (removes rows)."""
    rows_before = len(df)
    if not cols:
        return df, 0
    
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
    df_clean = df[mask]
    return df_clean, rows_before - len(df_clean)


def handle_outliers_zscore(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Z-score method for outlier removal (removes rows)."""
    rows_before = len(df)
    if not cols:
        return df, 0
        
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                z_scores = (df[col] - mean) / std
                mask &= z_scores.abs() <= 3
                
    df_clean = df[mask]
    return df_clean, rows_before - len(df_clean)


def preprocess_dataset_core(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = None,
    config: dict = None
) -> dict:
    """
    Core engine that applies missing value imputation, duplicate removal, encoding,
    outlier handling, scaling, and feature selection based on a config.
    Implements adaptive encoding to prevent memory issues.
    """
    start_time = time.time()
    
    config = config or {}
    before_stats = get_dataset_stats_dict(df)
    original_preview = df.head(10).astype(object).fillna("").to_dict(orient="records")
    
    working_df = df.copy()
    report = {}
    
    # 1. Drop rows with missing target values
    target_missing = int(working_df[target_column].isna().sum())
    if target_missing > 0:
        working_df = working_df.dropna(subset=[target_column])
        report["target_missing"] = f"Dropped {target_missing} rows with missing target values."
    else:
        report["target_missing"] = "No missing target values found."
        
    # 2. Duplicate Handling
    duplicate_handling = config.get("duplicate_handling", "remove")
    duplicates_before = int(working_df.duplicated().sum())
    duplicates_removed = 0
    if duplicate_handling == "remove":
        working_df = working_df.drop_duplicates()
        duplicates_removed = duplicates_before
        report["duplicates"] = f"Removed {duplicates_removed} duplicate rows."
    else:
        report["duplicates"] = "Duplicates kept as requested."

    # Identify all categorical columns initially present
    feature_cols = [c for c in working_df.columns if c != target_column]
    categorical_cols = []
    numeric_cols = []
    for col in feature_cols:
        if (
            working_df[col].dtype == "object"
            or isinstance(working_df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(working_df[col].dtype)
        ):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(working_df[col].dtype):
            numeric_cols.append(col)

    # 3. Categorical Column Strategy Resolution (Adaptive vs User Configured)
    column_strategies = {}
    encoding_analysis = []
    high_cardinality_dropped_list = []
    
    # Resolve manual overrides if provided
    manual_encodings = config.get("column_encodings") or {}
    
    # Get automatic recommendations
    recs = get_cardinality_recommendations(working_df, target_column)
    recs_map = {r["column"]: r for r in recs}

    for col in categorical_cols:
        rec_info = recs_map.get(col, {"recommendation": "onehot", "nunique": working_df[col].nunique(), "is_identifier": False})
        
        # Resolve strategy
        if col in manual_encodings:
            strategy = manual_encodings[col] # User explicit choice: "onehot", "label", or "drop"
        else:
            # Auto Mode: use recommended strategy
            strategy = rec_info["recommendation"]
            
        column_strategies[col] = strategy
        
        # If auto mode and dropped, add details to report
        if strategy == "drop":
            high_cardinality_dropped_list.append(col)
            
        encoding_analysis.append({
            "column": col,
            "unique_values": int(rec_info["nunique"]),
            "encoding": "One-Hot" if strategy == "onehot" else ("Label Encoding" if strategy == "label" else "Dropped")
        })

    # Drop columns determined to be dropped
    dropped_cols = [col for col, strat in column_strategies.items() if strat == "drop"]
    if dropped_cols:
        working_df = working_df.drop(columns=dropped_cols)
        report["features"] = f"Dropped categorical/identifier columns: {', '.join(dropped_cols)}"
        # Re-evaluate feature and categorical column lists
        feature_cols = [c for c in working_df.columns if c != target_column]
        categorical_cols = [c for c in categorical_cols if c not in dropped_cols]
    else:
        report["features"] = "All available features retained."

    # 4. User Selected Feature Filtering (if manual mode explicitly filtered features)
    selected_features = config.get("selected_features")
    if selected_features is not None:
        cols_to_keep = [col for col in selected_features if col in working_df.columns]
        if target_column not in cols_to_keep:
            cols_to_keep.append(target_column)
        features_removed = (working_df.shape[1] - 1) - (len(cols_to_keep) - 1)
        working_df = working_df[cols_to_keep]
        # Re-evaluate features list
        feature_cols = [c for c in working_df.columns if c != target_column]
        categorical_cols = [c for c in categorical_cols if c in feature_cols]
        numeric_cols = [c for c in numeric_cols if c in feature_cols]

    # 5. Missing Value Imputation
    missing_value_method = config.get("missing_value_method", "median")
    missing_values_fixed = int(working_df[feature_cols].isna().sum().sum())
    
    if missing_values_fixed > 0:
        if missing_value_method == "drop_cols":
            cols_with_missing = [c for c in feature_cols if working_df[c].isna().any()]
            working_df = working_df.drop(columns=cols_with_missing)
            report["missing_values"] = f"Dropped columns with missing values: {cols_with_missing}"
            feature_cols = [c for c in working_df.columns if c != target_column]
            numeric_cols = [c for c in numeric_cols if c in feature_cols]
            categorical_cols = [c for c in categorical_cols if c in feature_cols]
        elif missing_value_method == "drop_rows":
            working_df = working_df.dropna(subset=feature_cols)
            report["missing_values"] = "Dropped all rows containing missing values in feature columns."
        else:
            # Impute numerical features
            if numeric_cols:
                for col in numeric_cols:
                    if working_df[col].isna().any():
                        if missing_value_method == "mean":
                            val = working_df[col].mean()
                        elif missing_value_method == "mode":
                            val = working_df[col].mode()[0] if not working_df[col].mode().empty else 0
                        else:  # median
                            val = working_df[col].median()
                        if pd.isna(val):
                            val = 0
                        working_df[col] = working_df[col].fillna(val)
            
            # Impute categorical features
            if categorical_cols:
                for col in categorical_cols:
                    if working_df[col].isna().any():
                        mode_val = working_df[col].mode()
                        val = mode_val[0] if not mode_val.empty else "Unknown"
                        working_df[col] = working_df[col].fillna(val)
            
            report["missing_values"] = f"Missing values imputed using {missing_value_method.capitalize()} method."
    else:
        report["missing_values"] = "No missing values found in features."

    # 6. Outlier Handling
    outlier_method = config.get("outlier_method", "none")
    outliers_removed = 0
    if outlier_method == "iqr" and numeric_cols:
        working_df, outliers_removed = handle_outliers_iqr(working_df, numeric_cols)
        report["outliers"] = f"Outliers removed using IQR method ({outliers_removed} rows removed)."
    elif outlier_method == "zscore" and numeric_cols:
        working_df, outliers_removed = handle_outliers_zscore(working_df, numeric_cols)
        report["outliers"] = f"Outliers removed using Z-score method ({outliers_removed} rows removed)."
    else:
        report["outliers"] = "No outlier treatment applied."

    # Re-evaluate column categories after outlier removal
    feature_cols = [c for c in working_df.columns if c != target_column]
    numeric_cols = working_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = working_df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    # 7. Dimensional Explosion Protection & Strategy Apply
    # Estimate total columns after encoding
    estimated_features = len(numeric_cols)
    ohe_columns_to_apply = []
    
    for col in categorical_cols:
        strat = column_strategies.get(col, "onehot")
        if strat == "onehot":
            estimated_features += working_df[col].nunique()
            ohe_columns_to_apply.append(col)
        elif strat == "label":
            estimated_features += 1

    # Abort if the total features generated exceeds the safe limit (10,000)
    if estimated_features > 10000:
        raise ValueError(
            f"Encoding would generate {estimated_features} columns.\n"
            f"Recommended actions:\n"
            f"- Remove high-cardinality features\n"
            f"- Use Label Encoding\n"
            f"- Drop identifier columns"
        )

    # Perform Encoding
    categorical_encoders = {}
    features_before_encode = len(working_df.columns)
    
    if categorical_cols:
        # First process Label Encoding columns
        label_cols_to_apply = [col for col in categorical_cols if column_strategies.get(col) == "label"]
        for col in label_cols_to_apply:
            le = LabelEncoder()
            working_df[col] = le.fit_transform(working_df[col].astype(str))
            categorical_encoders[col] = le
            
        # Then process One-Hot Encoding columns using sparse encoding to save memory
        if ohe_columns_to_apply:
            # Generate sparse dummies
            dummies = pd.get_dummies(working_df[ohe_columns_to_apply], sparse=True, dtype="uint8")
            # Convert to dense before concatenation to keep downstream model compatibility
            dummies_dense = dummies.sparse.to_dense()
            # Drop original columns and concatenate dummies
            working_df = working_df.drop(columns=ohe_columns_to_apply)
            working_df = pd.concat([working_df, dummies_dense], axis=1)
            
        report["encoding"] = (
            f"Applied adaptive encoding: One-Hot encoded {len(ohe_columns_to_apply)} columns, "
            f"Label encoded {len(label_cols_to_apply)} columns."
        )
    else:
        report["encoding"] = "No categorical features present to encode."

    features_after_encode = len(working_df.columns)
    features_generated = max(0, features_after_encode - features_before_encode)

    # Protect against feature truncation (safeguard)
    final_feature_cols = [c for c in working_df.columns if c != target_column]
    if len(final_feature_cols) > 500:
        truncated_cols = final_feature_cols[:500]
        working_df = working_df[truncated_cols + [target_column]]
        final_feature_cols = truncated_cols
        report["features"] += " (Truncated features list to 500 to protect performance)."

    # 8. Scaling Numerical Features
    scaling_method = config.get("scaling_method", "standard")
    scaler = None
    if scaling_method != "none" and numeric_cols:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
            
        if scaler:
            existing_numeric_cols = [c for c in numeric_cols if c in working_df.columns]
            if existing_numeric_cols:
                working_df[existing_numeric_cols] = scaler.fit_transform(working_df[existing_numeric_cols])
                report["scaling"] = f"Applied {scaling_method.capitalize()} scaling to numeric features."
    else:
        report["scaling"] = "No numeric scaling applied."

    # 9. Target Encoding
    target_encoder = None
    if problem_type in ("binary", "multiclass") or (problem_type is None and not pd.api.types.is_numeric_dtype(working_df[target_column])):
        target_encoder = LabelEncoder()
        working_df[target_column] = target_encoder.fit_transform(working_df[target_column])
        report["target_encoding"] = f"Target label encoded. Classes: {list(target_encoder.classes_)}"
    else:
        report["target_encoding"] = "Target encoding not required."

    # 10. Compute Stats & Memory Metrics
    after_stats = get_dataset_stats_dict(working_df)
    processed_preview = working_df.head(10).astype(object).fillna("").to_dict(orient="records")
    processing_time = round(time.time() - start_time, 2)
    
    # Estimated Memory Saved Formula (Naive OHE float64 vs Sparse Adaptive size)
    naive_columns_count = len(numeric_cols)
    for col in categorical_cols:
        naive_columns_count += df[col].nunique()
    naive_memory_bytes = naive_columns_count * len(df) * 8 # float64 representation
    actual_memory_bytes = working_df.memory_usage(deep=True).sum()
    saved_bytes = max(0, naive_memory_bytes - actual_memory_bytes)
    estimated_memory_saved_mb = round(saved_bytes / (1024 * 1024), 2)
    
    mem_before = before_stats["memory_usage_mb"]
    mem_after = after_stats["memory_usage_mb"]
    memory_reduction_pct = round((1 - mem_after / mem_before) * 100, 1) if mem_before > 0 else 0.0

    processing_summary = {
        "missing_values_fixed": missing_values_fixed,
        "duplicates_removed": duplicates_removed,
        "features_generated": features_generated,
        "features_removed": len(dropped_cols),
        "memory_reduction_pct": memory_reduction_pct,
        "processing_time": processing_time,
        # Redesigned Memory Metrics:
        "columns_before_encoding": features_before_encode - 1, # minus target
        "columns_after_encoding": features_after_encode - 1,
        "estimated_memory_saved_mb": estimated_memory_saved_mb,
        "high_cardinality_removed_count": len(high_cardinality_dropped_list),
        "encoding_strategy_used": "Manual Custom" if config.get("column_encodings") else "Adaptive (One-Hot <=20, Label <=1000, Drop >1000)"
    }

    return {
        "processed_df": working_df,
        "scaler": scaler,
        "categorical_encoders": categorical_encoders,
        "target_encoder": target_encoder,
        "feature_names": final_feature_cols,
        "before_stats": before_stats,
        "after_stats": after_stats,
        "preprocessing_report": report,
        "processing_summary": processing_summary,
        "encoding_analysis": encoding_analysis,
        "column_strategies": column_strategies,
        "original_preview": original_preview,
        "processed_preview": processed_preview,
    }


def run_auto_preprocessing(df: pd.DataFrame, target_column: str, problem_type: str = None) -> dict:
    """Execute standard automatic preprocessing using core engine."""
    config = {
        "missing_value_method": "median",
        "duplicate_handling": "remove",
        "scaling_method": "standard",
        "outlier_method": "none",
        "selected_features": None
    }
    return preprocess_dataset_core(df, target_column, problem_type, config)


def run_manual_preprocessing(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str = None,
    config: dict = None
) -> dict:
    """Execute manual preprocessing based on user-defined config using core engine."""
    return preprocess_dataset_core(df, target_column, problem_type, config)


# ── Backward Compatibility Wrapper ──
def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str, problem_type: str = None):
    """
    Maintains compatibility with the training and validation orchestrator.
    Runs automatic preprocessing on a copy and returns splits.
    """
    res = run_auto_preprocessing(df, target_column, problem_type)
    processed_df = res["processed_df"]
    
    X = processed_df.drop(columns=[target_column])
    y = processed_df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None
    )
    
    steps_log = []
    for k, v in res["preprocessing_report"].items():
        steps_log.append(f"{k.replace('_', ' ').capitalize()}: {v}")
    steps_log.append(f"Split data: {len(X_train)} training, {len(X_test)} testing samples (80/20).")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": res["feature_names"],
        "scaler": res["scaler"],
        "label_encoder": res["target_encoder"],
        "steps_log": steps_log,
        "X_raw": X,
    }


def preprocess_for_prediction(
    input_data: dict | list | pd.DataFrame,
    feature_names: list,
    scaler,
    label_encoder=None,
    categorical_encoders=None,
    encoding_method="onehot"
):
    """
    Preprocess data (dict, list of dicts, or DataFrame) for prediction.
    Aligns columns with training features, applies fitted encoders and scalers.
    """
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    else:
        df = pd.DataFrame([input_data])

    categorical_encoders = categorical_encoders or {}

    # If encoding_method is a dict, it's the adaptive strategy mapping each column
    if isinstance(encoding_method, dict):
        # Apply Label Encoding first
        for col, method in encoding_method.items():
            if method == "label" and col in df.columns:
                if col in categorical_encoders:
                    le = categorical_encoders[col]
                    val = str(df[col].iloc[0])
                    if val in le.classes_:
                        df[col] = le.transform([val])[0]
                    else:
                        df[col] = 0
                else:
                    df[col] = 0
            elif method == "drop" and col in df.columns:
                df = df.drop(columns=[col])

        # Apply One-Hot Encoding to OHE columns
        ohe_cols = [col for col, method in encoding_method.items() if method == "onehot"]
        if ohe_cols:
            ohe_cols_present = [col for col in ohe_cols if col in df.columns]
            if ohe_cols_present:
                df = pd.get_dummies(df, columns=ohe_cols_present, dtype="uint8")

        # Fill any missing one-hot variables with zero and select exactly feature_names
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]

    else:
        # Backward compatibility for old pipelines (where encoding_method is a string)
        if encoding_method == "onehot":
            df = pd.get_dummies(df)
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_names]
        else:
            # Label encoding
            for col in feature_names:
                if col in categorical_encoders:
                    le = categorical_encoders[col]
                    val = str(df.get(col, [np.nan])[0])
                    if val in le.classes_:
                        df[col] = le.transform([val])[0]
                    else:
                        df[col] = 0
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df = df[feature_names]

    if scaler:
        # Perform scaling alignment
        df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_names)
        return df_scaled
    return df
