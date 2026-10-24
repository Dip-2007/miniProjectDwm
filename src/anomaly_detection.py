"""
SafeTraffic - Anomaly Detection Module
Flags potentially suspicious network traffic records using transparent,
rule-based heuristics applied AFTER K-Means clustering.

IMPORTANT: These rules do NOT make K-Means a classifier. They are simple
statistical thresholds that help analysts focus their investigation.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

DEFAULT_ZSCORE_THRESHOLD = 2.5  # Standard deviations above mean


def flag_suspicious_records(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """
    Flag records whose feature values are statistically unusual.

    Method
    ------
    For each selected feature, compute the z-score across the full dataset.
    A record is flagged if ANY feature's absolute z-score exceeds the threshold.

    This is a transparent, configurable heuristic — not a ground-truth label.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataframe (original scale).
    labels : np.ndarray
        Cluster assignments.
    feature_columns : list[str]
        Features used for clustering.
    zscore_threshold : float
        Number of standard deviations beyond which a value is unusual.

    Returns
    -------
    result : pd.DataFrame
        Original data augmented with:
          - Cluster : cluster ID
          - Suspicious : bool flag
          - Suspicion_Reasons : human-readable explanation
          - Investigation_Status : severity label
    """
    result = df_clean.copy()
    result['Cluster'] = labels

    # Compute z-scores for each feature
    reasons_list = []
    suspicious_flags = np.zeros(len(result), dtype=bool)
    reason_counts = np.zeros(len(result), dtype=int)

    for col in feature_columns:
        mean_val = result[col].mean()
        std_val = result[col].std()
        if std_val == 0:
            continue
        zscores = (result[col] - mean_val) / std_val
        is_unusual = zscores.abs() > zscore_threshold

        suspicious_flags |= is_unusual.values
        reason_counts += is_unusual.values.astype(int)

    # Build per-row reason strings only for flagged records (massive speedup on large datasets)
    reasons_list = [""] * len(result)
    flagged_indices = np.where(suspicious_flags)[0]
    
    if len(flagged_indices) > 0:
        # Precompute means and stds as numpy arrays
        means = np.array([result[col].mean() for col in feature_columns])
        stds = np.array([result[col].std() for col in feature_columns])
        valid_mask = stds > 0
        feat_names = np.array(feature_columns)
        
        flagged_vals = result[feature_columns].iloc[flagged_indices].to_numpy()
        
        for i, orig_idx in enumerate(flagged_indices):
            row_vals = flagged_vals[i]
            zs = np.zeros_like(row_vals)
            zs[valid_mask] = (row_vals[valid_mask] - means[valid_mask]) / stds[valid_mask]
            
            unusual_indices = np.where(np.abs(zs) > zscore_threshold)[0]
            if len(unusual_indices) > 0:
                parts = [
                    f"{feat_names[u]}: unusually {'high' if zs[u] > 0 else 'low'} (val={row_vals[u]:.1f}, z={zs[u]:.1f})"
                    for u in unusual_indices
                ]
                reasons_list[orig_idx] = "; ".join(parts)

    result['Suspicious'] = suspicious_flags
    result['Suspicion_Reasons'] = reasons_list

    # Assign investigation status
    result['Investigation_Status'] = 'Normal'
    result.loc[
        result['Suspicious'] & (reason_counts <= 2),
        'Investigation_Status'
    ] = 'Needs Investigation'
    result.loc[
        result['Suspicious'] & (reason_counts > 2),
        'Investigation_Status'
    ] = 'High Priority'

    return result


def flag_suspicious_clusters(
    df_flagged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise suspicion rates per cluster.

    Returns a DataFrame with one row per cluster and columns:
      Cluster, Size, Suspicious_Count, Suspicious_Pct
    """
    summary = df_flagged.groupby('Cluster').agg(
        Size=('Cluster', 'count'),
        Suspicious_Count=('Suspicious', 'sum'),
    ).reset_index()
    summary['Suspicious_Pct'] = np.round(
        summary['Suspicious_Count'] / summary['Size'] * 100, 2
    )
    return summary


def get_suspicious_records(df_flagged: pd.DataFrame) -> pd.DataFrame:
    """Return only the flagged records."""
    return df_flagged[df_flagged['Suspicious']].copy()
