"""
SafeTraffic - Clustering Module
Implements K-Means clustering, Elbow Method, and Silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# K-Means helpers
# ---------------------------------------------------------------------------

def run_kmeans(
    df_scaled: pd.DataFrame,
    k: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit a K-Means model and return it along with cluster labels.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled feature matrix.
    k : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of K-Means initialisations.

    Returns
    -------
    model : KMeans
        Fitted KMeans model.
    labels : np.ndarray
        Cluster labels for each row.
    """
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
        max_iter=300,
    )
    labels = model.fit_predict(df_scaled.values)
    return model, labels


def compute_elbow_data(
    df_scaled: pd.DataFrame,
    k_range: range,
    random_state: int = 42,
    n_init: int = 3,
    max_samples: int = 3000,
) -> dict[int, float]:
    """
    Compute inertia for a range of K values (for the Elbow Method).
    Samples data if rows exceed max_samples for fast computation.

    Returns a dict mapping K -> inertia.
    """
    data = df_scaled.values
    if len(data) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_samples, replace=False)
        data = data[idx]

    inertias = {}
    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=100,
        )
        model.fit(data)
        inertias[k] = model.inertia_
    return inertias


def compute_silhouette(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    max_sample: int = 2000,
    random_state: int = 42,
) -> float | None:
    """
    Compute the Silhouette Score for the given clustering.

    Returns None if the score cannot be computed (e.g., only one cluster,
    fewer than 2 samples, or all points in one cluster).
    """
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return None

    data = df_scaled.values
    if len(data) > max_sample:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_sample, replace=False)
        data = data[idx]
        labels = labels[idx]

    try:
        return float(silhouette_score(data, labels))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------------------------

def cluster_statistics(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Compute per-cluster statistics (mean, std, min, max) for each feature
    in the original (unscaled) feature space.
    """
    df_temp = df_clean[feature_columns].copy()
    df_temp['Cluster'] = labels

    stats = df_temp.groupby('Cluster').agg(['mean', 'std', 'min', 'max', 'count'])
    return stats


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with cluster sizes and proportions."""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    df = pd.DataFrame({
        'Cluster': unique,
        'Size': counts,
        'Percentage': np.round(counts / total * 100, 2),
    })
    return df


def get_centroids_original_scale(
    model: KMeans,
    scaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert cluster centroids from scaled space back to original feature units.
    This gives faculty an interpretable view of what each cluster centre looks like.
    """
    centroids_scaled = model.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    return pd.DataFrame(centroids_original, columns=feature_columns)
