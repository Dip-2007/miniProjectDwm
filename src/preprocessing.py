"""
SafeTraffic - Preprocessing Module
Cleans, validates, and prepares network traffic data for K-Means clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_preprocessing_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the raw dataset before preprocessing.
    Returns a dictionary with dataset statistics.
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': int(df.isnull().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
    }


def preprocess_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    remove_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict]:
    """
    Preprocess the dataframe for K-Means clustering.

    Steps:
    1. Convert numeric strings to numbers.
    2. Replace infinite values with NaN.
    3. Drop rows with NaN in selected features.
    4. Remove duplicate rows (optional).
    5. Handle highly skewed features (log1p transform for skewness > 5).
    6. Scale features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    feature_columns : list[str]
        Columns to use as clustering features.
    remove_duplicates : bool
        Whether to remove duplicate rows.

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned dataframe (original scale, with all columns preserved).
    df_scaled : pd.DataFrame
        Scaled feature matrix (only selected features).
    scaler : StandardScaler
        Fitted scaler for inverse transforms.
    report : dict
        Summary of preprocessing actions.
    """
    report = {
        'initial_rows': len(df),
        'initial_columns': len(df.columns),
        'features_selected': len(feature_columns),
        'inf_replaced': 0,
        'nan_dropped': 0,
        'duplicates_removed': 0,
        'skewed_features': [],
        'final_rows': 0,
    }

    # Work on a copy
    df_work = df.copy()

    # Step 1: Convert numeric strings
    for col in feature_columns:
        if df_work[col].dtype == object:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')

    # Step 2: Replace inf with NaN
    inf_mask = df_work[feature_columns].isin([np.inf, -np.inf])
    report['inf_replaced'] = int(inf_mask.sum().sum())
    df_work[feature_columns] = df_work[feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    # Step 3: Drop rows with NaN in feature columns
    before_drop = len(df_work)
    df_work = df_work.dropna(subset=feature_columns).reset_index(drop=True)
    report['nan_dropped'] = before_drop - len(df_work)

    # Step 4: Remove duplicates
    if remove_duplicates:
        before_dedup = len(df_work)
        df_work = df_work.drop_duplicates(subset=feature_columns).reset_index(drop=True)
        report['duplicates_removed'] = before_dedup - len(df_work)

    # Keep the cleaned full dataframe (original scale)
    df_clean = df_work.copy()

    # Step 5: Handle highly skewed features with log1p
    feature_data = df_clean[feature_columns].copy()
    for col in feature_columns:
        skewness = feature_data[col].skew()
        if abs(skewness) > 5 and feature_data[col].min() >= 0:
            feature_data[col] = np.log1p(feature_data[col])
            report['skewed_features'].append(col)

    # Step 6: Scale using StandardScaler
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_data)
    df_scaled = pd.DataFrame(scaled_values, columns=feature_columns)

    report['final_rows'] = len(df_clean)

    return df_clean, df_scaled, scaler, report


def validate_features(df: pd.DataFrame, feature_columns: list[str]) -> tuple[bool, str]:
    """
    Validate that the selected feature columns are suitable for clustering.

    Returns (is_valid, message).
    """
    if not feature_columns:
        return False, "No features selected. Please select at least 2 numeric features."

    if len(feature_columns) < 2:
        return False, "Please select at least 2 features for meaningful clustering."

    # Check that all features exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return False, f"Columns not found in dataset: {missing}"

    # Check that features are numeric or convertible
    for col in feature_columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors='coerce')
            if converted.notna().sum() < 10:
                return False, (
                    f"Column '{col}' cannot be converted to numeric values. "
                    "Please select only numeric features."
                )

    # Check variance
    numeric_df = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    zero_var = [c for c in feature_columns if numeric_df[c].std() == 0]
    if zero_var:
        return False, (
            f"Columns with zero variance (constant values): {zero_var}. "
            "These provide no information for clustering."
        )

    return True, "Features are valid for clustering."
