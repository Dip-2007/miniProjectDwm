"""
SafeTraffic - Visualizations Module
Creates interactive Plotly charts and Matplotlib plots for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Color palette (cybersecurity theme)
# ---------------------------------------------------------------------------
CLUSTER_COLORS = px.colors.qualitative.Set2
STATUS_COLORS = {
    'Normal': '#00c853',          # Green
    'Needs Investigation': '#ffab00',  # Amber
    'High Priority': '#ff1744',        # Red
}


def plot_elbow_curve(
    inertias: dict[int, float],
    best_k: int | None = None,
) -> go.Figure:
    """Interactive Elbow Method chart with optional optimal K highlighted."""
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=vals,
        mode='lines+markers',
        marker=dict(size=10, color='#00bcd4'),
        line=dict(width=3, color='#00bcd4'),
        name='Inertia',
    ))

    if best_k is not None and best_k in inertias:
        fig.add_trace(go.Scatter(
            x=[best_k], y=[inertias[best_k]],
            mode='markers+text',
            marker=dict(size=16, color='#ffab00', symbol='star'),
            text=[f"Recommended (K={best_k})"],
            textposition='top right',
            name=f'Optimal K={best_k}',
        ))
        fig.add_vline(
            x=best_k,
            line_dash='dash',
            line_color='#ffab00',
            opacity=0.7,
            annotation_text=f"Elbow (K={best_k})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title='Elbow Method — Inertia vs. Number of Clusters (K)',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Inertia (Within-Cluster Sum of Squares)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig


def plot_cluster_scatter_2d(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    max_points: int = 5000,
    random_state: int = 42,
) -> go.Figure:
    """
    2D scatter plot of clustered data.
    If more than 2 features, PCA is used to reduce to 2 dimensions.
    """
    data = df_scaled.values
    cluster_labels = labels.copy()
    use_pca = len(feature_columns) > 2

    # Sample for large datasets
    if len(data) > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_points, replace=False)
        data = data[idx]
        cluster_labels = cluster_labels[idx]

    if use_pca:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(data)
        x_label = f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
        y_label = f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
        title_suffix = ' (PCA Projection)'
    else:
        coords = data[:, :2]
        x_label = feature_columns[0]
        y_label = feature_columns[1]
        title_suffix = ''

    plot_df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Cluster': cluster_labels.astype(str),
    })

    fig = px.scatter(
        plot_df, x='X', y='Y', color='Cluster',
        title=f'K-Means Cluster Visualization{title_suffix}',
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.6,
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=500,
    )
    return fig, use_pca


def plot_cluster_sizes(size_df: pd.DataFrame) -> go.Figure:
    """Bar chart of cluster sizes."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=size_df['Cluster'].astype(str),
        y=size_df['Size'],
        text=size_df['Percentage'].apply(lambda p: f'{p}%'),
        textposition='outside',
        marker_color='#00bcd4',
    ))
    fig.update_layout(
        title='Cluster Size Distribution',
        xaxis_title='Cluster',
        yaxis_title='Number of Records',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig


def plot_feature_distributions(
    df_clean: pd.DataFrame,
    feature_columns: list[str],
    labels: np.ndarray,
    max_features: int = 6,
) -> go.Figure:
    """Box plots comparing feature distributions across clusters."""
    cols_to_plot = feature_columns[:max_features]
    df_temp = df_clean[cols_to_plot].copy()
    df_temp['Cluster'] = labels.astype(str)

    # Melt for faceted plot
    df_melted = df_temp.melt(
        id_vars='Cluster', var_name='Feature', value_name='Value'
    )

    fig = px.box(
        df_melted, x='Cluster', y='Value', color='Cluster',
        facet_col='Feature', facet_col_wrap=3,
        color_discrete_sequence=CLUSTER_COLORS,
        title='Feature Distributions by Cluster',
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=500,
        showlegend=False,
    )
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def plot_investigation_status(df_flagged: pd.DataFrame) -> go.Figure:
    """Pie chart of investigation statuses."""
    counts = df_flagged['Investigation_Status'].value_counts()

    colors = [STATUS_COLORS.get(label, '#9e9e9e') for label in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
    )])
    fig.update_layout(
        title='Traffic Investigation Status',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig


def plot_label_cluster_heatmap(crosstab: pd.DataFrame) -> go.Figure:
    """Heatmap showing true labels vs cluster assignments."""
    fig = go.Figure(data=go.Heatmap(
        z=crosstab.values,
        x=[f'Cluster {c}' for c in crosstab.columns],
        y=crosstab.index.tolist(),
        colorscale='Teal',
        text=crosstab.values,
        texttemplate='%{text}',
        hovertemplate='Label: %{y}<br>Cluster: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title='True Labels vs. Cluster Assignments',
        xaxis_title='Cluster',
        yaxis_title='True Label',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=max(400, 50 * len(crosstab)),
    )
    return fig


def plot_traffic_distribution(
    df_clean: pd.DataFrame,
    feature: str,
    labels: np.ndarray,
) -> go.Figure:
    """Histogram of a single feature coloured by cluster."""
    df_temp = df_clean[[feature]].copy()
    df_temp['Cluster'] = labels.astype(str)

    fig = px.histogram(
        df_temp, x=feature, color='Cluster',
        marginal='box', barmode='overlay',
        color_discrete_sequence=CLUSTER_COLORS,
        title=f'Distribution of {feature} by Cluster',
        opacity=0.7,
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# DBSCAN and Comparative Plots
# ---------------------------------------------------------------------------

def plot_k_distance_graph(
    k_distances: np.ndarray,
    recommended_eps: float | None = None,
) -> go.Figure:
    """
    Plot sorted k-nearest neighbor distances for selecting the DBSCAN eps parameter.
    The knee/elbow point indicates the optimal threshold between cluster cores and noise.
    """
    x_vals = np.arange(len(k_distances))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=k_distances,
        mode='lines',
        line=dict(color='#00e5ff', width=2.5),
        name='k-distance',
    ))

    if recommended_eps is not None:
        fig.add_hline(
            y=recommended_eps,
            line_dash='dash',
            line_color='#ffab00',
            annotation_text=f"Estimated optimal ε = {recommended_eps:.3f}",
            annotation_position="top left",
        )

    fig.update_layout(
        title='DBSCAN k-Distance Graph (Knee Method for ε Estimation)',
        xaxis_title='Points Sorted by Distance to k-th Nearest Neighbor',
        yaxis_title='Distance to k-th Neighbor (ε candidate)',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig


def plot_dbscan_scatter_2d(
    df_scaled: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
    max_points: int = 5000,
    random_state: int = 42,
) -> tuple[go.Figure, bool]:
    """
    2D scatter plot of DBSCAN clusters.
    Noise / anomaly points (label = -1) are clearly distinguished in red.
    """
    data = df_scaled.values
    db_labels = labels.copy()
    use_pca = len(feature_columns) > 2

    if len(data) > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_points, replace=False)
        data = data[idx]
        db_labels = db_labels[idx]

    if use_pca:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(data)
        x_label = f'PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
        y_label = f'PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
        title_suffix = ' (PCA Projection)'
    else:
        coords = data[:, :2]
        x_label = feature_columns[0]
        y_label = feature_columns[1]
        title_suffix = ''

    plot_df = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'Cluster_Raw': db_labels,
    })
    plot_df['Cluster'] = plot_df['Cluster_Raw'].apply(
        lambda c: 'Noise / Outlier (Anomaly)' if c == -1 else f'Cluster {c}'
    )

    # Custom color map ensuring noise is red
    unique_clusters = sorted(plot_df['Cluster'].unique())
    color_map = {}
    dense_idx = 0
    for cl in unique_clusters:
        if 'Noise' in cl:
            color_map[cl] = '#ff1744'  # Red for noise
        else:
            color_map[cl] = CLUSTER_COLORS[dense_idx % len(CLUSTER_COLORS)]
            dense_idx += 1

    fig = px.scatter(
        plot_df, x='x', y='y', color='Cluster',
        color_discrete_map=color_map,
        title=f'DBSCAN Clustering Visualization{title_suffix}',
        labels={'x': x_label, 'y': y_label},
        opacity=0.75,
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=500,
    )
    return fig, use_pca


def plot_algorithm_comparison_scatter(
    df_scaled: pd.DataFrame,
    kmeans_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    feature_columns: list[str],
    max_points: int = 3000,
    random_state: int = 42,
) -> go.Figure:
    """
    Side-by-side scatter plots comparing K-Means and DBSCAN on identical 2D/PCA projection.
    """
    data = df_scaled.values
    km_lbl = kmeans_labels.copy()
    db_lbl = dbscan_labels.copy()

    if len(data) > max_points:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(data), max_points, replace=False)
        data = data[idx]
        km_lbl = km_lbl[idx]
        db_lbl = db_lbl[idx]

    if len(feature_columns) > 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(data)
        x_label = f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'
        y_label = f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)'
    else:
        coords = data[:, :2]
        x_label = feature_columns[0]
        y_label = feature_columns[1]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"K-Means (K={len(np.unique(km_lbl))}) — Partitioning",
            f"DBSCAN ({len(set(db_lbl) - {-1})} Clusters + Noise) — Density-Based"
        ),
        shared_yaxes=True,
    )

    # 1. K-Means scatter
    for k_val in sorted(np.unique(km_lbl)):
        mask = (km_lbl == k_val)
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode='markers',
                marker=dict(size=5, opacity=0.7),
                name=f'KM Cluster {k_val}',
                legendgroup='kmeans',
            ),
            row=1, col=1,
        )

    # 2. DBSCAN scatter
    for d_val in sorted(np.unique(db_lbl)):
        mask = (db_lbl == d_val)
        if d_val == -1:
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode='markers',
                    marker=dict(size=6, color='#ff1744', symbol='x', opacity=0.85),
                    name='DBSCAN Noise (Anomalies)',
                    legendgroup='dbscan',
                ),
                row=1, col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode='markers',
                    marker=dict(size=5, opacity=0.7),
                    name=f'DB Cluster {d_val}',
                    legendgroup='dbscan',
                ),
                row=1, col=2,
            )

    fig.update_xaxes(title_text=x_label, row=1, col=1)
    fig.update_xaxes(title_text=x_label, row=1, col=2)
    fig.update_yaxes(title_text=y_label, row=1, col=1)

    fig.update_layout(
        title='Direct Comparison: K-Means vs. DBSCAN Clustering Structure',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=500,
    )
    return fig


def plot_comparison_metrics_bar(comp_res: dict) -> go.Figure:
    """
    Grouped bar chart comparing key quantitative metrics between K-Means and DBSCAN.
    """
    categories = ['Anomaly / Outlier Rate (%)']
    km_vals = [comp_res['km_outliers_pct']]
    db_vals = [comp_res['db_noise_pct']]

    if comp_res.get('km_sil') is not None and comp_res.get('db_sil') is not None:
        categories.append('Silhouette Score')
        km_vals.append(comp_res['km_sil'])
        db_vals.append(comp_res['db_sil'])

    if comp_res.get('km_ari') is not None and comp_res.get('db_ari') is not None:
        categories.append('Adjusted Rand Index (ARI)')
        km_vals.append(comp_res['km_ari'])
        db_vals.append(comp_res['db_ari'])

    if comp_res.get('km_nmi') is not None and comp_res.get('db_nmi') is not None:
        categories.append('Normalized Mutual Info (NMI)')
        km_vals.append(comp_res['km_nmi'])
        db_vals.append(comp_res['db_nmi'])

    fig = go.Figure(data=[
        go.Bar(name='K-Means', x=categories, y=km_vals, marker_color='#00e5ff'),
        go.Bar(name='DBSCAN', x=categories, y=db_vals, marker_color='#ffab00'),
    ])

    fig.update_layout(
        barmode='group',
        title='Quantitative Metrics Comparison: K-Means vs. DBSCAN',
        yaxis_title='Metric Value',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        height=400,
    )
    return fig
