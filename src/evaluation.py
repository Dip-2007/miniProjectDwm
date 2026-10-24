"""
SafeTraffic - Evaluation Module
Compares K-Means cluster assignments against ground-truth labels when available.

KEY ACADEMIC NOTE:
  K-Means is an UNSUPERVISED algorithm. Cluster IDs are arbitrary integers
  and do NOT directly correspond to attack classes. We use external clustering
  metrics (ARI, NMI) which are invariant to label permutation.
  We do NOT report "accuracy" as if K-Means were a supervised classifier.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder


def evaluate_with_labels(
    true_labels: pd.Series,
    cluster_labels: np.ndarray,
) -> dict:
    """
    Evaluate clustering against ground-truth labels using permutation-
    invariant metrics.

    Parameters
    ----------
    true_labels : pd.Series
        Ground-truth class labels (e.g., 'BENIGN', 'DDoS').
    cluster_labels : np.ndarray
        Cluster IDs from K-Means.

    Returns
    -------
    metrics : dict
        - adjusted_rand_index : float (-1 to 1, higher = better agreement)
        - normalized_mutual_info : float (0 to 1, higher = better)
        - label_distribution : dict mapping label -> count
        - cluster_label_crosstab : pd.DataFrame
    """
    # Encode string labels to integers for sklearn metrics
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels.astype(str))

    ari = adjusted_rand_score(true_encoded, cluster_labels)
    nmi = normalized_mutual_info_score(
        true_encoded, cluster_labels, average_method='arithmetic'
    )

    # Cross-tabulation: rows = true labels, columns = cluster IDs
    crosstab = pd.crosstab(
        true_labels.astype(str),
        cluster_labels,
        rownames=['True Label'],
        colnames=['Cluster'],
    )

    label_dist = true_labels.value_counts().to_dict()

    return {
        'adjusted_rand_index': round(float(ari), 4),
        'normalized_mutual_info': round(float(nmi), 4),
        'label_distribution': label_dist,
        'cluster_label_crosstab': crosstab,
    }


def get_cluster_label_mapping(
    true_labels: pd.Series,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    For each cluster, show the dominant true label and its proportion.
    Useful for understanding which traffic types ended up together.
    """
    df_temp = pd.DataFrame({
        'True_Label': true_labels.astype(str).values,
        'Cluster': cluster_labels,
    })

    rows = []
    for cid in sorted(df_temp['Cluster'].unique()):
        subset = df_temp[df_temp['Cluster'] == cid]
        counts = subset['True_Label'].value_counts()
        dominant = counts.index[0]
        dominant_pct = round(counts.iloc[0] / len(subset) * 100, 2)
        rows.append({
            'Cluster': cid,
            'Size': len(subset),
            'Dominant_Label': dominant,
            'Dominant_Label_Pct': dominant_pct,
            'Unique_Labels': len(counts),
        })

    return pd.DataFrame(rows)


def unsupervised_evaluation_summary(
    inertia: float,
    silhouette: float | None,
    k: int,
    n_samples: int,
) -> dict:
    """
    Produce an evaluation summary when ground-truth labels are NOT available.
    """
    summary = {
        'k': k,
        'n_samples': n_samples,
        'inertia': round(float(inertia), 2),
        'silhouette_score': round(float(silhouette), 4) if silhouette is not None else None,
        'note': (
            "Without ground-truth labels, we cannot measure attack-detection "
            "performance. Inertia and Silhouette Score measure clustering "
            "quality only — lower inertia and higher silhouette suggest "
            "tighter, better-separated clusters."
        ),
    }
    return summary
