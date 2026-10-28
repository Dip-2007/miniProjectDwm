"""
SafeTraffic - Clustering Module
Implements K-Means clustering, Elbow Method, and Silhouette analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import LabelEncoder


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


def find_best_k(
    df_scaled: pd.DataFrame,
    k_range: range = range(2, 9),
    random_state: int = 42,
    max_samples: int = 3000,
) -> tuple[int, dict]:
    """
    Automatically determine the best K using the Elbow (knee-point) method
    and validating with the Silhouette Score.

    Returns
    -------
    best_k : int
        Recommended optimal number of clusters.
    details : dict
        Dictionary containing inertias, elbow distances, and silhouette scores.
    """
    inertias = compute_elbow_data(
        df_scaled, k_range=k_range, random_state=random_state, max_samples=max_samples
    )
    ks = sorted(inertias.keys())
    if len(ks) < 3:
        default_k = ks[0] if ks else 4
        return default_k, {'inertias': inertias, 'best_k': default_k}

    ys = [inertias[k] for k in ks]
    ks_arr = np.array(ks, dtype=float)
    ys_arr = np.array(ys, dtype=float)

    # Normalize coordinates to [0, 1]
    x_norm = (ks_arr - ks_arr[0]) / (ks_arr[-1] - ks_arr[0])
    y_range = ys_arr[0] - ys_arr[-1]
    if y_range == 0:
        y_norm = np.zeros_like(ys_arr)
    else:
        y_norm = (ys_arr - ys_arr[-1]) / y_range

    # Distance to the chord line connecting (ks[0], y_max) to (ks[-1], y_min)
    # Line equation: x_norm + y_norm = 1 => deviation below line = 1 - (x_norm + y_norm)
    elbow_distances = 1.0 - (x_norm + y_norm)
    best_idx = int(np.argmax(elbow_distances))
    elbow_best_k = ks[best_idx]

    # Validate with silhouette score for elbow_best_k
    sil_score = None
    try:
        _, sample_labels = run_kmeans(df_scaled, k=elbow_best_k, random_state=random_state)
        sil_score = compute_silhouette(df_scaled, sample_labels, random_state=random_state)
    except Exception:
        pass

    details = {
        'inertias': inertias,
        'elbow_best_k': elbow_best_k,
        'elbow_distances': dict(zip(ks, elbow_distances.tolist())),
        'silhouette_score': sil_score,
    }
    return elbow_best_k, details


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


# ---------------------------------------------------------------------------
# DBSCAN helpers & Algorithm Comparison
# ---------------------------------------------------------------------------

def run_dbscan(
    df_scaled: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 5,
    n_jobs: int = -1,
) -> tuple[DBSCAN, np.ndarray]:
    """
    Fit a DBSCAN model and return it along with cluster labels.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Scaled feature matrix.
    eps : float
        Maximum distance between two samples to be considered in the same neighborhood.
    min_samples : int
        Number of samples in a neighborhood for a point to be a core point.
    n_jobs : int
        Number of parallel jobs (-1 uses all available processors).

    Returns
    -------
    model : DBSCAN
        Fitted DBSCAN model.
    labels : np.ndarray
        Cluster labels (-1 represents noise / anomaly points).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs)
    labels = model.fit_predict(df_scaled.values)
    return model, labels


def estimate_dbscan_eps(
    df_scaled: pd.DataFrame,
    min_samples: int = 5,
    max_samples: int = 3000,
    random_state: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Automatically estimate an optimal eps parameter for DBSCAN using
    the knee-point of the k-distance graph.

    Computes the distance to the min_samples-th nearest neighbor for each sample,
    sorts them in ascending order, and locates the elbow/knee point where distance
    spikes sharply.

    Returns
    -------
    recommended_eps : float
        Estimated optimal eps parameter.
    k_distances : np.ndarray
        Sorted k-nearest neighbor distances for plotting the k-distance curve.
    """
    data = df_scaled.values
    if len(data) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_samples, replace=False)
        data = data[idx]

    k = max(2, min_samples)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)

    k_distances = np.sort(distances[:, -1])
    n_pts = len(k_distances)

    if n_pts < 3:
        return 0.5, k_distances

    x_norm = np.linspace(0.0, 1.0, n_pts)
    d_min, d_max = float(k_distances[0]), float(k_distances[-1])
    denom = d_max - d_min if d_max > d_min else 1.0
    y_norm = (k_distances - d_min) / denom

    # Maximum vertical distance between diagonal chord line (y=x) and the curve
    knee_distances = x_norm - y_norm
    best_idx = int(np.argmax(knee_distances))
    recommended_eps = float(k_distances[best_idx])

    # Fallback guard if knee is at the boundary
    if recommended_eps <= 0.05 or best_idx <= 5 or best_idx >= n_pts - 5:
        recommended_eps = float(np.percentile(k_distances, 85))

    recommended_eps = round(max(0.1, recommended_eps), 3)
    return recommended_eps, k_distances


def dbscan_cluster_summary(labels: np.ndarray) -> dict:
    """
    Summarise DBSCAN clustering results:
    - Number of clusters (excluding noise)
    - Noise points count and percentage
    - Per-cluster sizes DataFrame
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    noise_count = int(np.sum(labels == -1))
    total_points = len(labels)
    noise_pct = round((noise_count / total_points) * 100, 2) if total_points > 0 else 0.0

    n_clusters = len(set(labels) - {-1})

    rows = []
    for lbl, count in zip(unique_labels, counts):
        pct = round(count / total_points * 100, 2)
        if lbl == -1:
            name = "Noise / Outliers (Anomalies)"
            type_str = "Anomaly"
        else:
            name = f"Cluster {lbl}"
            type_str = "Dense Cluster"
        rows.append({
            'Cluster ID': lbl,
            'Name': name,
            'Type': type_str,
            'Size': count,
            'Percentage': pct,
        })
    df_sizes = pd.DataFrame(rows)

    return {
        'n_clusters': n_clusters,
        'noise_count': noise_count,
        'noise_pct': noise_pct,
        'sizes_df': df_sizes,
    }


def compare_clustering_algorithms(
    df_scaled: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    kmeans_outliers_count: int,
    true_labels: pd.Series | None = None,
    max_samples_metrics: int = 3000,
    random_state: int = 42,
) -> dict:
    """
    Perform a rigorous academic comparison between K-Means and DBSCAN clustering.
    Returns comparison metrics, tables, and conclusion takeaways for faculty presentation.
    """
    n_total = len(df_scaled)
    data = df_scaled.values

    # Subsample if large for fast silhouette computation
    if n_total > max_samples_metrics:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n_total, max_samples_metrics, replace=False)
        sub_data = data[sample_idx]
        sub_km_labels = kmeans_labels[sample_idx]
        sub_db_labels = dbscan_labels[sample_idx]
    else:
        sub_data = data
        sub_km_labels = kmeans_labels
        sub_db_labels = dbscan_labels

    # 1. Clusters count
    km_k = len(np.unique(kmeans_labels))
    db_clusters = len(set(dbscan_labels) - {-1})

    # 2. Anomalies / Outliers detected
    km_outliers_cnt = kmeans_outliers_count
    km_outliers_pct = round(km_outliers_cnt / n_total * 100, 2) if n_total > 0 else 0.0

    db_noise_cnt = int(np.sum(dbscan_labels == -1))
    db_noise_pct = round(db_noise_cnt / n_total * 100, 2) if n_total > 0 else 0.0

    # 3. Silhouette Score
    km_sil = None
    try:
        if len(set(sub_km_labels)) >= 2:
            km_sil = round(float(silhouette_score(sub_data, sub_km_labels)), 4)
    except Exception:
        pass

    db_sil = None
    try:
        non_noise_mask = (sub_db_labels != -1)
        if len(set(sub_db_labels[non_noise_mask])) >= 2:
            db_sil = round(float(silhouette_score(sub_data[non_noise_mask], sub_db_labels[non_noise_mask])), 4)
        elif len(set(sub_db_labels)) >= 2:
            db_sil = round(float(silhouette_score(sub_data, sub_db_labels)), 4)
    except Exception:
        pass

    # 4. Davies-Bouldin Index (lower is better)
    km_db = None
    try:
        if len(set(sub_km_labels)) >= 2:
            km_db = round(float(davies_bouldin_score(sub_data, sub_km_labels)), 4)
    except Exception:
        pass

    db_db = None
    try:
        non_noise_mask = (sub_db_labels != -1)
        if len(set(sub_db_labels[non_noise_mask])) >= 2:
            db_db = round(float(davies_bouldin_score(sub_data[non_noise_mask], sub_db_labels[non_noise_mask])), 4)
        elif len(set(sub_db_labels)) >= 2:
            db_db = round(float(davies_bouldin_score(sub_data, sub_db_labels)), 4)
    except Exception:
        pass

    # 5. External Ground Truth Metrics (if true_labels available)
    km_ari, km_nmi = None, None
    db_ari, db_nmi = None, None
    if true_labels is not None:
        try:
            le = LabelEncoder()
            true_encoded = le.fit_transform(true_labels.astype(str))
            km_ari = round(float(adjusted_rand_score(true_encoded, kmeans_labels)), 4)
            km_nmi = round(float(normalized_mutual_info_score(true_encoded, kmeans_labels, average_method='arithmetic')), 4)
            db_ari = round(float(adjusted_rand_score(true_encoded, dbscan_labels)), 4)
            db_nmi = round(float(normalized_mutual_info_score(true_encoded, dbscan_labels, average_method='arithmetic')), 4)
        except Exception:
            pass

    # 6. Build Academic Comparison Table
    table_rows = [
        {
            'Criterion / Metric': 'Clustering Paradigm',
            'K-Means': 'Centroid-based (Partitioning)',
            'DBSCAN': 'Density-based (Spatial Density)',
            'Academic Takeaway': 'Different mathematical foundations',
        },
        {
            'Criterion / Metric': 'Clusters Discovered',
            'K-Means': f'K = {km_k} (Pre-specified parameter)',
            'DBSCAN': f'{db_clusters} dense clusters (Data-driven)',
            'Academic Takeaway': 'DBSCAN does not require knowing K in advance',
        },
        {
            'Criterion / Metric': 'Anomalies / Noise Detected',
            'K-Means': f'{km_outliers_cnt:,} ({km_outliers_pct}%) [Post-hoc z-score]',
            'DBSCAN': f'{db_noise_cnt:,} ({db_noise_pct}%) [Intrinsic Noise (Label -1)]',
            'Academic Takeaway': 'DBSCAN natively separates noise without heuristics',
        },
        {
            'Criterion / Metric': 'Cluster Geometry Assumption',
            'K-Means': 'Spherical / Convex clusters only',
            'DBSCAN': 'Arbitrary non-linear shapes & manifolds',
            'Academic Takeaway': 'DBSCAN captures complex attack patterns better',
        },
        {
            'Criterion / Metric': 'Silhouette Score (Higher is better)',
            'K-Means': f'{km_sil}' if km_sil is not None else 'N/A',
            'DBSCAN': f'{db_sil}' if db_sil is not None else 'N/A',
            'Academic Takeaway': 'K-Means usually optimizes compactness; DBSCAN isolates density',
        },
        {
            'Criterion / Metric': 'Davies-Bouldin Index (Lower is better)',
            'K-Means': f'{km_db}' if km_db is not None else 'N/A',
            'DBSCAN': f'{db_db}' if db_db is not None else 'N/A',
            'Academic Takeaway': 'Evaluates separation between cluster boundaries',
        },
    ]

    if true_labels is not None and km_ari is not None:
        table_rows.append({
            'Criterion / Metric': 'Adjusted Rand Index (ARI)',
            'K-Means': f'{km_ari:.4f}',
            'DBSCAN': f'{db_ari:.4f}' if db_ari is not None else 'N/A',
            'Academic Takeaway': 'Agreement with true ground-truth attack classes',
        })
        table_rows.append({
            'Criterion / Metric': 'Normalized Mutual Info (NMI)',
            'K-Means': f'{km_nmi:.4f}',
            'DBSCAN': f'{db_nmi:.4f}' if db_nmi is not None else 'N/A',
            'Academic Takeaway': 'Mutual information shared with true labels',
        })

    table_rows.extend([
        {
            'Criterion / Metric': 'Time Complexity',
            'K-Means': 'O(k · n · i) — Linear & fast',
            'DBSCAN': 'O(n²) or O(n log n) with spatial index',
            'Academic Takeaway': 'K-Means scales better to massive real-time streams',
        },
        {
            'Criterion / Metric': 'Interpretability',
            'K-Means': 'Explicit centroids represent average profiles',
            'DBSCAN': 'Identifies Core, Border, and Outlier points',
            'Academic Takeaway': 'Centroids are easier for tabular rules; DBSCAN better for pure detection',
        },
    ])

    comparison_df = pd.DataFrame(table_rows)

    return {
        'km_k': km_k,
        'db_clusters': db_clusters,
        'km_outliers_cnt': km_outliers_cnt,
        'km_outliers_pct': km_outliers_pct,
        'db_noise_cnt': db_noise_cnt,
        'db_noise_pct': db_noise_pct,
        'km_sil': km_sil,
        'db_sil': db_sil,
        'km_db': km_db,
        'db_db': db_db,
        'km_ari': km_ari,
        'db_ari': db_ari,
        'km_nmi': km_nmi,
        'db_nmi': db_nmi,
        'comparison_df': comparison_df,
    }
