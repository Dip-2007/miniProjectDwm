"""
SafeTraffic - Visualizations Module
Creates interactive Plotly charts and Matplotlib plots for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def plot_elbow_curve(inertias: dict[int, float]) -> go.Figure:
    """Interactive Elbow Method chart."""
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
