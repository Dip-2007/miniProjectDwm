"""
SafeTraffic — Website Traffic Anomaly Detection Using K-Means Clustering
=========================================================================
Main Streamlit application.

Run with:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# ----- Project modules -----
from src.data_loader import (
    generate_demo_dataset, load_csv_file, detect_label_column,
    get_numeric_feature_columns, list_local_datasets, load_csv_path,
)
from src.preprocessing import (
    preprocess_dataframe, validate_features, get_preprocessing_summary,
)
from src.clustering import (
    run_kmeans, compute_elbow_data, compute_silhouette, find_best_k,
    cluster_sizes, cluster_statistics, get_centroids_original_scale,
    run_dbscan, estimate_dbscan_eps, dbscan_cluster_summary,
    compare_clustering_algorithms,
)
from src.anomaly_detection import (
    flag_suspicious_records, flag_suspicious_clusters, get_suspicious_records,
    DEFAULT_ZSCORE_THRESHOLD,
)
from src.evaluation import (
    evaluate_with_labels, get_cluster_label_mapping,
    unsupervised_evaluation_summary,
)
from src.visualizations import (
    plot_elbow_curve, plot_cluster_scatter_2d, plot_cluster_sizes,
    plot_feature_distributions, plot_investigation_status,
    plot_label_cluster_heatmap, plot_traffic_distribution,
    plot_k_distance_graph, plot_dbscan_scatter_2d,
    plot_algorithm_comparison_scatter, plot_comparison_metrics_bar,
)

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    page_title="SafeTraffic — Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS — Dark Cybersecurity Theme
# =====================================================================
st.markdown("""
<style>
    /* --- Global --- */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1628 100%);
        color: #e0e0e0;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* --- KPI Cards --- */
    .kpi-card {
        background: linear-gradient(135deg, #122240 0%, #1a3355 100%);
        border: 1px solid #1e4976;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,188,212,0.15);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #90a4ae;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .kpi-green { color: #00e676; }
    .kpi-amber { color: #ffab00; }
    .kpi-red   { color: #ff1744; }
    .kpi-cyan  { color: #00e5ff; }

    /* --- Section headers --- */
    .section-header {
        background: linear-gradient(90deg, #1e3a5f 0%, transparent 100%);
        padding: 10px 16px;
        border-left: 4px solid #00bcd4;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* --- Info boxes --- */
    .info-box {
        background: rgba(0,188,212,0.08);
        border: 1px solid rgba(0,188,212,0.25);
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* --- Badge --- */
    .badge-demo {
        display: inline-block;
        background: #1a237e;
        color: #82b1ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-real {
        display: inline-block;
        background: #1b5e20;
        color: #69f0ae;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* --- Status badges --- */
    .status-normal { color: #00e676; font-weight: 600; }
    .status-investigate { color: #ffab00; font-weight: 600; }
    .status-high { color: #ff1744; font-weight: 600; }

    /* --- Streamlit overrides --- */
    .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
        color: #b0bec5 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(18,34,64,0.6);
        border: 1px solid #1e4976;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Helper functions
# =====================================================================

def kpi_card(label: str, value, color_class: str = "kpi-cyan"):
    """Render a styled KPI card."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {color_class}">{value}</div>
    </div>
    """


def section_header(title: str):
    """Render a styled section header."""
    st.markdown(f'<div class="section-header">🔹 {title}</div>', unsafe_allow_html=True)


def info_box(text: str):
    """Render an informational callout box."""
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


# =====================================================================
# Sidebar
# =====================================================================

with st.sidebar:
    st.markdown("# 🛡️ SafeTraffic")
    st.markdown("**Website Traffic Anomaly Detection**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard Overview",
            "📁 Dataset Explorer",
            "🔬 Clustering Analysis",
            "🚨 Suspicious Traffic",
            "📈 Model Evaluation",
            "⚖️ Algorithm Comparison (K-Means vs DBSCAN)",
        ],
        index=0,
    )

    st.markdown("---")

    # ----- Detect Local CSV Files -----
    local_data_files = [os.path.join("data", f) for f in list_local_datasets("data")]
    local_root_files = [
        f for f in os.listdir(".")
        if f.lower().endswith(".csv") and not f.startswith("safetraffic_analysis")
    ]
    all_local_csvs = local_data_files + local_root_files

    # ----- Data Source -----
    st.markdown("### Data Source")
    
    source_options = []
    if all_local_csvs:
        source_options.append("📁 Local Dataset (Folder)")
    source_options.extend(["🧪 Demo Mode", "📂 Upload CSV"])
    if not all_local_csvs:
        source_options.insert(1, "📁 Local Dataset (Folder)")

    data_mode = st.radio(
        "Select mode:",
        source_options,
        index=0,
        help="Local Dataset reads CSVs placed in the project/data folder directly from disk without upload delay.",
    )

    selected_local_file = None
    uploaded_file = None

    if data_mode == "📁 Local Dataset (Folder)":
        if all_local_csvs:
            file_display = {
                f: f"{os.path.basename(f)} ({os.path.getsize(f) / (1024*1024):.1f} MB)"
                for f in all_local_csvs
            }
            selected_local_file = st.selectbox(
                "Choose local dataset:",
                options=all_local_csvs,
                format_func=lambda x: file_display.get(x, x),
                help="Select any dataset found in d:\\DWM-MiniProject\\data\\ or project folder."
            )
        else:
            st.warning("No CSV files found in `data/` or project folder.")
            st.info("Copy your CIC-IDS2017 CSV file into `d:\\DWM-MiniProject\\data\\`.")

    elif data_mode == "📂 Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CIC-IDS2017 CSV",
            type=['csv'],
            help="Upload a CSV file from the CIC-IDS2017 dataset.",
        )

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#546e7a;font-size:0.75rem;">'
        'SafeTraffic v1.0<br>DWM Mini-Project<br>K-Means Clustering'
        '</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# Load Data
# =====================================================================

@st.cache_data
def load_demo():
    return generate_demo_dataset(n_samples=2000, random_state=42)

@st.cache_data
def load_disk_csv_cached(filepath: str) -> pd.DataFrame:
    return load_csv_path(filepath)

@st.cache_data
def load_uploaded_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    import io
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding='latin-1', low_memory=False)
    df.columns = df.columns.str.strip()
    return df

is_demo = (data_mode == "🧪 Demo Mode")

if is_demo:
    df_raw = load_demo()
    data_badge = '<span class="badge-demo">🧪 DEMO DATA — Synthetic traffic patterns for demonstration</span>'
elif data_mode == "📁 Local Dataset (Folder)":
    if selected_local_file is not None and os.path.exists(selected_local_file):
        try:
            with st.spinner(f"Loading {os.path.basename(selected_local_file)} directly from disk..."):
                df_raw = load_disk_csv_cached(selected_local_file)
            data_badge = f'<span class="badge-real">📁 LOCAL DATASET — {os.path.basename(selected_local_file)} ({len(df_raw):,} records)</span>'
        except Exception as e:
            st.error(f"Failed to read local CSV file: {e}")
            st.stop()
    else:
        st.info("👈 Please place a CSV file in `data/` or switch to Demo Mode.")
        st.stop()
else:
    if uploaded_file is not None:
        try:
            with st.spinner("Loading uploaded dataset..."):
                file_bytes = uploaded_file.getvalue()
                df_raw = load_uploaded_csv_cached(file_bytes)
            data_badge = f'<span class="badge-real">📂 REAL DATASET — {uploaded_file.name} ({len(df_raw):,} records)</span>'
        except Exception as e:
            st.error(f"Failed to read CSV file: {e}")
            st.stop()
    else:
        st.info("👈 Upload a CSV file from the sidebar, or switch to Demo Mode.")
        st.stop()

# Detect label column
label_col = detect_label_column(df_raw)
has_labels = label_col is not None

# Get numeric feature columns
available_features = get_numeric_feature_columns(df_raw)

if len(available_features) < 2:
    st.error(
        "❌ The uploaded dataset does not have enough numeric feature columns "
        "for clustering. At least 2 numeric columns (excluding IDs and labels) "
        f"are required. Found columns: {list(df_raw.columns)}"
    )
    st.stop()


# =====================================================================
# Feature Selection & Preprocessing (shared state)
# =====================================================================

# Let the user pick features in the sidebar
with st.sidebar:
    st.markdown("### Feature Selection")

    # Default features: pick the first 6 or all if fewer
    default_count = min(6, len(available_features))
    selected_features = st.multiselect(
        "Select clustering features:",
        options=available_features,
        default=available_features[:default_count],
        help="Choose numeric traffic features. Identifiers and labels are excluded automatically.",
    )

    # Performance sampling for large datasets (e.g. CIC-IDS2017 files with 100k+ rows)
    if len(df_raw) > 10000:
        st.markdown("### Performance Sampling")
        sample_choice = st.selectbox(
            "Records to analyze:",
            options=[
                "10,000 records (Instant / Demo Recommended)",
                "25,000 records (Balanced)",
                "50,000 records (Deep Analysis)",
                "All records (Full Dataset)"
            ],
            index=0,
            help="10,000 records preserves the exact statistical traffic distribution while making all charts and clustering run in under 1 second."
        )
        if "10,000" in sample_choice:
            df_to_analyze = df_raw.sample(n=10000, random_state=42).reset_index(drop=True)
        elif "25,000" in sample_choice:
            df_to_analyze = df_raw.sample(n=min(25000, len(df_raw)), random_state=42).reset_index(drop=True)
        elif "50,000" in sample_choice:
            df_to_analyze = df_raw.sample(n=min(50000, len(df_raw)), random_state=42).reset_index(drop=True)
        else:
            df_to_analyze = df_raw
    else:
        sample_choice = "all"
        df_to_analyze = df_raw

    st.markdown("### 🔹 K-Means Settings")

    max_k_possible = max(2, min(15, len(df_raw) - 1))
    if "k_slider" not in st.session_state:
        st.session_state["k_slider"] = 4
    elif st.session_state["k_slider"] > max_k_possible:
        st.session_state["k_slider"] = min(4, max_k_possible)

    if st.button("⚡ Auto-Detect Best K", use_container_width=True, help="Automatically determine optimal K via Elbow knee-point analysis"):
        if not selected_features:
            st.error("Select features first.")
        else:
            with st.spinner("Finding optimal K..."):
                df_c_temp, df_s_temp, _, _ = preprocess_dataframe(df_to_analyze, selected_features)
                max_k_search = min(10, len(df_c_temp) - 1)
                best_k_found, k_details = find_best_k(df_s_temp, k_range=range(2, max_k_search + 1))
                st.session_state["k_slider"] = int(best_k_found)
                st.session_state["auto_k_detected"] = int(best_k_found)
                sil_str = f" (Silhouette = {k_details['silhouette_score']:.3f})" if k_details.get('silhouette_score') is not None else ""
                st.session_state["auto_k_msg"] = f"✅ Auto-selected **K = {best_k_found}**{sil_str}!"
                st.rerun()

    if st.session_state.get("auto_k_msg"):
        st.caption(st.session_state["auto_k_msg"])

    k_value = st.slider(
        "Number of clusters (K):",
        min_value=2,
        max_value=max_k_possible,
        key="k_slider",
        help="Choose K for K-Means. Use '⚡ Auto-Detect Best K' or the Elbow chart.",
    )

    st.markdown("### 🔸 DBSCAN Settings")
    default_min_samples = max(3, len(selected_features) * 2) if selected_features else 5
    if "dbscan_eps_slider" not in st.session_state:
        st.session_state["dbscan_eps_slider"] = 0.8
    if "dbscan_min_samples_slider" not in st.session_state:
        st.session_state["dbscan_min_samples_slider"] = min(15, default_min_samples)

    if st.button("⚡ Auto-Detect Best ε", use_container_width=True, help="Automatically estimate optimal epsilon via k-distance graph"):
        if not selected_features:
            st.error("Select features first.")
        else:
            with st.spinner("Estimating optimal ε..."):
                df_c_temp, df_s_temp, _, _ = preprocess_dataframe(df_to_analyze, selected_features)
                cur_min_s = st.session_state.get("dbscan_min_samples_slider", 5)
                rec_eps, _ = estimate_dbscan_eps(df_s_temp, min_samples=cur_min_s)
                st.session_state["dbscan_eps_slider"] = float(rec_eps)
                st.session_state["auto_eps_msg"] = f"✅ Auto-selected **ε = {rec_eps:.3f}**!"
                st.rerun()

    if st.session_state.get("auto_eps_msg"):
        st.caption(st.session_state["auto_eps_msg"])

    dbscan_eps = st.slider(
        "Epsilon radius (ε):",
        min_value=0.1,
        max_value=5.0,
        step=0.05,
        key="dbscan_eps_slider",
        help="Max distance for two points to be considered in the same neighborhood.",
    )

    dbscan_min_samples = st.slider(
        "Min samples (MinPts):",
        min_value=2,
        max_value=30,
        key="dbscan_min_samples_slider",
        help="Minimum number of samples in an ε-neighborhood for a point to be a core point.",
    )

    st.markdown("### ⚠️ Heuristic Anomaly Filter")
    zscore_thresh = st.slider(
        "K-Means suspicion threshold (z-score):",
        min_value=1.5, max_value=5.0, value=DEFAULT_ZSCORE_THRESHOLD, step=0.1,
        help="Records with any feature z-score above this threshold are flagged for investigation.",
    )

# Validate features
is_valid, validation_msg = validate_features(df_to_analyze, selected_features)
if not is_valid:
    st.warning(f"⚠️ {validation_msg}")
    st.stop()

# Pipeline Caching Key (so switching tabs/pages takes 0.00 seconds)
pipeline_cache_key = (
    data_mode,
    selected_local_file if data_mode == "📁 Local Dataset (Folder)" else "",
    tuple(selected_features),
    k_value,
    dbscan_eps,
    dbscan_min_samples,
    zscore_thresh,
    sample_choice,
    len(df_to_analyze),
)

if st.session_state.get('pipeline_key') != pipeline_cache_key:
    # Preprocess
    with st.spinner("Preprocessing selected features..."):
        df_clean, df_scaled, scaler, prep_report = preprocess_dataframe(
            df_to_analyze, selected_features
        )

    if len(df_clean) < k_value:
        st.error(
            f"After preprocessing, only {len(df_clean)} records remain, "
            f"which is fewer than K={k_value}. Reduce K or use a larger dataset."
        )
        st.stop()

    # Run K-Means
    with st.spinner("Fitting K-Means model..."):
        model, labels = run_kmeans(df_scaled, k=k_value)

    # Run DBSCAN
    with st.spinner("Fitting DBSCAN model..."):
        dbscan_model, dbscan_labels = run_dbscan(
            df_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples
        )
        dbscan_summary = dbscan_cluster_summary(dbscan_labels)

    # Flag suspicious records (K-Means heuristic)
    with st.spinner("Analyzing anomalies..."):
        df_flagged = flag_suspicious_records(df_clean, labels, selected_features, zscore_thresh)

    # Cluster sizes
    sizes_df = cluster_sizes(labels)

    # Suspicious summary
    suspicious_summary = flag_suspicious_clusters(df_flagged)
    total_suspicious = int(df_flagged['Suspicious'].sum())
    suspicious_pct = round(total_suspicious / len(df_flagged) * 100, 2) if len(df_flagged) > 0 else 0

    # Save to session_state
    st.session_state['pipeline_key'] = pipeline_cache_key
    st.session_state['pipeline_results'] = {
        'df_clean': df_clean,
        'df_scaled': df_scaled,
        'scaler': scaler,
        'prep_report': prep_report,
        'model': model,
        'labels': labels,
        'dbscan_model': dbscan_model,
        'dbscan_labels': dbscan_labels,
        'dbscan_summary': dbscan_summary,
        'dbscan_eps': dbscan_eps,
        'dbscan_min_samples': dbscan_min_samples,
        'df_flagged': df_flagged,
        'sizes_df': sizes_df,
        'suspicious_summary': suspicious_summary,
        'total_suspicious': total_suspicious,
        'suspicious_pct': suspicious_pct,
    }
else:
    # Retrieve instantly from cache
    res = st.session_state['pipeline_results']
    df_clean = res['df_clean']
    df_scaled = res['df_scaled']
    scaler = res['scaler']
    prep_report = res['prep_report']
    model = res['model']
    labels = res['labels']
    dbscan_model = res['dbscan_model']
    dbscan_labels = res['dbscan_labels']
    dbscan_summary = res['dbscan_summary']
    dbscan_eps = res['dbscan_eps']
    dbscan_min_samples = res['dbscan_min_samples']
    df_flagged = res['df_flagged']
    sizes_df = res['sizes_df']
    suspicious_summary = res['suspicious_summary']
    total_suspicious = res['total_suspicious']
    suspicious_pct = res['suspicious_pct']


# =====================================================================
# PAGE: Dashboard Overview
# =====================================================================

if page == "📊 Dashboard Overview":
    st.markdown("# 📊 Dashboard Overview")
    st.markdown(data_badge, unsafe_allow_html=True)

    if is_demo:
        info_box(
            "⚠️ <strong>Demo Mode Active:</strong> You are viewing results from a "
            "synthetic dataset generated for demonstration purposes. These are "
            "<em>simulated traffic patterns</em>, not real network data or cyberattacks."
        )

    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(kpi_card("Total Records Analyzed", f"{len(df_flagged):,}", "kpi-cyan"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Clusters Formed", k_value, "kpi-green"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Flagged for Investigation", f"{total_suspicious:,}", "kpi-amber"), unsafe_allow_html=True)
    with col4:
        st.markdown(kpi_card("Flagged Percentage", f"{suspicious_pct}%", "kpi-red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Traffic Investigation Status")
        fig_status = plot_investigation_status(df_flagged)
        st.plotly_chart(fig_status, use_container_width=True)

    with col_right:
        section_header("Cluster Size Distribution")
        fig_sizes = plot_cluster_sizes(sizes_df)
        st.plotly_chart(fig_sizes, use_container_width=True)

    # Feature distribution
    section_header("Traffic Feature Distribution")
    if selected_features:
        feat_to_show = st.selectbox(
            "Select feature to inspect:", selected_features, key="dash_feat"
        )
        fig_dist = plot_traffic_distribution(df_clean, feat_to_show, labels)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Algorithm explanation
    section_header("How It Works")
    info_box(
        "<strong>K-Means Clustering</strong> groups records that have similar "
        "numerical characteristics. Records in the same cluster share similar "
        "traffic feature patterns (e.g., packet sizes, flow rates, durations).<br><br>"
        "After clustering, we apply <em>statistical thresholds</em> (z-scores) to "
        "identify records with unusually extreme feature values. These are flagged "
        "as <strong>Potentially Suspicious</strong> — they need human investigation "
        "and are NOT automatically confirmed as attacks.<br><br>"
        "K-Means does <em>not</em> know which clusters are malicious. The flagging "
        "is a separate, transparent step applied after clustering."
    )


# =====================================================================
# PAGE: Dataset Explorer
# =====================================================================

elif page == "📁 Dataset Explorer":
    st.markdown("# 📁 Dataset Explorer")
    st.markdown(data_badge, unsafe_allow_html=True)

    if is_demo:
        info_box(
            "⚠️ <strong>Demo Data:</strong> This is a synthetic dataset with "
            "simulated traffic patterns. Switch to Upload Mode for real data."
        )

    # Raw dataset summary
    section_header("Raw Dataset Summary")
    raw_summary = get_preprocessing_summary(df_raw)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{raw_summary['rows']:,}")
    with col2:
        st.metric("Columns", raw_summary['columns'])
    with col3:
        st.metric("Duplicates", f"{raw_summary['duplicates']:,}")

    # Data types
    section_header("Column Information")
    col_info = pd.DataFrame({
        'Column': raw_summary['column_names'],
        'Data Type': [raw_summary['dtypes'][c] for c in raw_summary['column_names']],
        'Missing Values': [raw_summary['missing_values'][c] for c in raw_summary['column_names']],
    })
    st.dataframe(col_info, use_container_width=True, height=300)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**Numeric columns:** {len(raw_summary['numeric_columns'])}")
        st.code(", ".join(raw_summary['numeric_columns'][:20]))
    with col_right:
        st.markdown(f"**Categorical columns:** {len(raw_summary['categorical_columns'])}")
        st.code(", ".join(raw_summary['categorical_columns']))

    if has_labels:
        section_header("Ground-Truth Labels")
        st.markdown(f"Label column detected: **`{label_col}`**")
        label_counts = df_raw[label_col].value_counts()
        st.dataframe(label_counts.reset_index().rename(
            columns={'index': 'Label', label_col: 'Label', 'count': 'Count'}
        ), use_container_width=True)

    # Preprocessing report
    section_header("Preprocessing Summary (Before → After)")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("**Before Preprocessing**")
        st.write(f"- Rows: **{prep_report['initial_rows']:,}**")
        st.write(f"- Columns: **{prep_report['initial_columns']}**")
    with p_col2:
        st.markdown("**After Preprocessing**")
        st.write(f"- Rows: **{prep_report['final_rows']:,}**")
        st.write(f"- Features selected: **{prep_report['features_selected']}**")
        st.write(f"- Infinite values replaced: **{prep_report['inf_replaced']}**")
        st.write(f"- NaN rows dropped: **{prep_report['nan_dropped']}**")
        st.write(f"- Duplicates removed: **{prep_report['duplicates_removed']}**")
        if prep_report['skewed_features']:
            st.write(f"- Log-transformed (high skew): **{len(prep_report['skewed_features'])}** features")

    # Data preview
    section_header("Data Preview (First 100 Rows)")
    st.dataframe(df_raw.head(100), use_container_width=True, height=300)

    # Feature statistics
    section_header("Selected Feature Statistics")
    st.dataframe(
        df_clean[selected_features].describe().T.round(2),
        use_container_width=True,
    )


# =====================================================================
# PAGE: Clustering Analysis
# =====================================================================

elif page == "🔬 Clustering Analysis":
    st.markdown("# 🔬 Clustering Analysis")
    st.markdown(data_badge, unsafe_allow_html=True)

    tab_km, tab_db = st.tabs(["🔹 K-Means Clustering", "🔸 DBSCAN Density Clustering"])

    # -----------------------------------------------------------------
    # TAB 1: K-Means Clustering
    # -----------------------------------------------------------------
    with tab_km:
        info_box(
            "<strong>K-Means Algorithm:</strong> K-Means partitions the data into K clusters "
            "by minimizing the within-cluster sum of squared distances (inertia). Each record "
            "is assigned to the nearest cluster centroid. The algorithm iterates until centroids "
            "stabilize.<br><br>"
            "<strong>Selected features:</strong> " + ", ".join(selected_features) +
            f"<br><strong>K = {k_value}</strong> clusters"
        )

        # Elbow Method
        section_header("Elbow Method")
        info_box(
            "The <strong>Elbow Method</strong> plots inertia (within-cluster sum of squares) "
            "for different values of K. The 'elbow' — where adding more clusters gives "
            "diminishing returns — suggests a good K. There may not be a clear elbow in "
            "all datasets."
        )

        elbow_cache_key = (pipeline_cache_key, "elbow")
        if st.session_state.get('elbow_key') != elbow_cache_key:
            with st.spinner("Computing Elbow curve..."):
                max_k = min(10, len(df_clean) - 1)
                elbow_data = compute_elbow_data(df_scaled, k_range=range(2, max_k + 1))
                best_k_calc, _ = find_best_k(df_scaled, k_range=range(2, max_k + 1))
                st.session_state['elbow_data'] = elbow_data
                st.session_state['elbow_best_k'] = best_k_calc
                st.session_state['elbow_key'] = elbow_cache_key
        else:
            elbow_data = st.session_state['elbow_data']
            best_k_calc = st.session_state.get('elbow_best_k')

        fig_elbow = plot_elbow_curve(elbow_data, best_k=best_k_calc)
        st.plotly_chart(fig_elbow, use_container_width=True)

        if best_k_calc is not None:
            col_rec_1, col_rec_2 = st.columns([1.2, 3])
            with col_rec_1:
                if st.button("⚡ Apply Recommended K", key="apply_opt_k_btn", use_container_width=True):
                    st.session_state["k_slider"] = int(best_k_calc)
                    st.session_state["auto_k_msg"] = f"✅ Applied optimal **K = {best_k_calc}**!"
                    st.rerun()
            with col_rec_2:
                st.markdown(
                    f"🌟 **Optimal Knee Point:** The Elbow Method recommends **K = {best_k_calc}** "
                    f"(marked with a gold star on the chart)."
                )

        # Silhouette Score
        section_header("Silhouette Score")
        if st.session_state.get('sil_key') != pipeline_cache_key:
            sil_score = compute_silhouette(df_scaled, labels)
            st.session_state['sil_score'] = sil_score
            st.session_state['sil_key'] = pipeline_cache_key
        else:
            sil_score = st.session_state.get('sil_score')
        if sil_score is not None:
            st.metric("Silhouette Score", f"{sil_score:.4f}")
            info_box(
                f"The <strong>Silhouette Score</strong> measures how similar each record is "
                f"to its own cluster compared to other clusters. Range: -1 to 1.<br>"
                f"• Values near <strong>1</strong> = well-clustered<br>"
                f"• Values near <strong>0</strong> = overlapping clusters<br>"
                f"• Values near <strong>-1</strong> = possible misassignment<br><br>"
                f"Current score: <strong>{sil_score:.4f}</strong> for K={k_value}."
            )
        else:
            st.warning(
                "Silhouette Score could not be computed. This happens when there is "
                "only 1 effective cluster or too few data points."
            )

        # Cluster Visualization
        section_header("Cluster Visualization (2D)")
        fig_scatter, used_pca = plot_cluster_scatter_2d(
            df_scaled, labels, selected_features
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        if used_pca:
            info_box(
                "📐 <strong>PCA Projection:</strong> Since you selected more than 2 features, "
                "PCA (Principal Component Analysis) was used to reduce the data to 2 dimensions "
                "for visualization only. The K-Means model was trained on ALL selected features "
                "in the original high-dimensional space. PCA does NOT replace K-Means — it is "
                "only used here to create a 2D view."
            )

        # Cluster Sizes
        section_header("Cluster Sizes")
        fig_sizes = plot_cluster_sizes(sizes_df)
        st.plotly_chart(fig_sizes, use_container_width=True)
        st.dataframe(sizes_df, use_container_width=True)

        # Cluster Centroids
        section_header("Cluster Centroids (Original Feature Scale)")
        centroids_df = get_centroids_original_scale(model, scaler, selected_features)
        centroids_df.index.name = 'Cluster'
        st.dataframe(centroids_df.round(2), use_container_width=True)
        info_box(
            "Centroids are the 'average point' of each cluster, shown in the original "
            "feature units (not scaled). Compare centroid values across clusters to "
            "understand what makes each cluster different."
        )

        # Feature Distributions by Cluster
        section_header("Feature Distributions by Cluster")
        fig_box = plot_feature_distributions(df_clean, selected_features, labels)
        st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # TAB 2: DBSCAN Density Clustering
    # -----------------------------------------------------------------
    with tab_db:
        info_box(
            "<strong>DBSCAN (Density-Based Spatial Clustering of Applications with Noise):</strong><br>"
            "Unlike K-Means, DBSCAN does <strong>NOT</strong> require specifying the number of clusters in advance. "
            "It groups samples that are closely packed together (within distance <strong>ε = "
            f"{dbscan_eps}</strong> and having at least <strong>MinPts = {dbscan_min_samples}</strong> neighbors).<br><br>"
            "Crucially for network cybersecurity, records that reside in sparse, low-density regions "
            "are intrinsically marked as <strong>Noise / Anomalies (Cluster -1)</strong> without needing post-hoc thresholds!"
        )

        # DBSCAN KPIs
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            st.markdown(kpi_card("Dense Clusters", dbscan_summary['n_clusters'], "kpi-green"), unsafe_allow_html=True)
        with d_col2:
            st.markdown(kpi_card("Noise / Anomalies", f"{dbscan_summary['noise_count']:,}", "kpi-red"), unsafe_allow_html=True)
        with d_col3:
            st.markdown(kpi_card("Noise Rate", f"{dbscan_summary['noise_pct']}%", "kpi-amber"), unsafe_allow_html=True)
        with d_col4:
            st.markdown(kpi_card("Epsilon (ε)", f"{dbscan_eps}", "kpi-cyan"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # k-distance graph
        section_header("Parameter Selection: k-Distance Graph")
        k_dist_cache_key = (pipeline_cache_key, "k_distance", dbscan_min_samples)
        if st.session_state.get('k_dist_key') != k_dist_cache_key:
            with st.spinner("Computing k-distance graph..."):
                rec_eps_val, k_dist_arr = estimate_dbscan_eps(df_scaled, min_samples=dbscan_min_samples)
                st.session_state['k_dist_arr'] = k_dist_arr
                st.session_state['rec_eps_val'] = rec_eps_val
                st.session_state['k_dist_key'] = k_dist_cache_key
        else:
            k_dist_arr = st.session_state['k_dist_arr']
            rec_eps_val = st.session_state['rec_eps_val']

        fig_kdist = plot_k_distance_graph(k_dist_arr, recommended_eps=rec_eps_val)
        st.plotly_chart(fig_kdist, use_container_width=True)

        col_eps_1, col_eps_2 = st.columns([1.2, 3])
        with col_eps_1:
            if st.button("⚡ Apply Recommended ε", key="apply_opt_eps_btn", use_container_width=True):
                st.session_state["dbscan_eps_slider"] = float(rec_eps_val)
                st.session_state["auto_eps_msg"] = f"✅ Applied optimal **ε = {rec_eps_val:.3f}**!"
                st.rerun()
        with col_eps_2:
            st.markdown(
                f"🌟 **Optimal Knee Point:** The k-distance curve indicates a sharp inflection at **ε = {rec_eps_val:.3f}**. "
                "Points beyond this threshold are low-density outliers."
            )

        # DBSCAN Silhouette Score
        section_header("DBSCAN Silhouette Evaluation")
        db_non_noise = (dbscan_labels != -1)
        if len(set(dbscan_labels[db_non_noise])) >= 2:
            db_sil = compute_silhouette(df_scaled[db_non_noise], dbscan_labels[db_non_noise])
            if db_sil is not None:
                st.metric("Silhouette Score (Dense Clusters)", f"{db_sil:.4f}")
                info_box(f"Silhouette score for the {dbscan_summary['n_clusters']} dense clusters (excluding noise): <strong>{db_sil:.4f}</strong>.")
        else:
            st.info("Silhouette score requires at least 2 dense clusters to compute. Adjust ε or MinPts if you wish to separate more dense clusters.")

        # DBSCAN 2D Scatter Plot
        section_header("DBSCAN Cluster & Noise Visualization (2D)")
        fig_db_scatter, db_used_pca = plot_dbscan_scatter_2d(
            df_scaled, dbscan_labels, selected_features
        )
        st.plotly_chart(fig_db_scatter, use_container_width=True)
        if db_used_pca:
            info_box(
                "📐 <strong>PCA Projection:</strong> PCA was used to project the multi-dimensional feature space "
                "to 2D for visualization. The bright red 'x' marks indicate true density-based <strong>Noise / Anomalies</strong>."
            )

        # DBSCAN Cluster Sizes
        section_header("DBSCAN Cluster Sizes & Outliers")
        st.dataframe(dbscan_summary['sizes_df'], use_container_width=True)


# =====================================================================
# PAGE: Suspicious Traffic Explorer
# =====================================================================

elif page == "🚨 Suspicious Traffic":
    st.markdown("# 🚨 Suspicious Traffic Explorer")
    st.markdown(data_badge, unsafe_allow_html=True)

    info_box(
        f"Records are flagged when <strong>any feature's z-score exceeds "
        f"±{zscore_thresh:.1f}</strong> standard deviations from the dataset mean. "
        f"This is a transparent statistical heuristic, NOT a ground-truth attack label. "
        f"Flagged records need <em>human investigation</em>."
    )

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(kpi_card("Total Records", f"{len(df_flagged):,}", "kpi-cyan"), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_card("Flagged Records", f"{total_suspicious:,}", "kpi-amber"), unsafe_allow_html=True)
    with col3:
        st.markdown(kpi_card("Flagged %", f"{suspicious_pct}%", "kpi-red"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Per-cluster suspicion summary
    section_header("Suspicion Rate by Cluster")
    st.dataframe(suspicious_summary, use_container_width=True)

    # Investigation status pie
    section_header("Investigation Status Breakdown")
    fig_status = plot_investigation_status(df_flagged)
    st.plotly_chart(fig_status, use_container_width=True)

    # Filterable table
    section_header("Flagged Records — Detailed View")

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        status_filter = st.multiselect(
            "Filter by status:",
            options=['Normal', 'Needs Investigation', 'High Priority'],
            default=['Needs Investigation', 'High Priority'],
        )
    with filter_col2:
        cluster_filter = st.multiselect(
            "Filter by cluster:",
            options=sorted(df_flagged['Cluster'].unique().tolist()),
            default=sorted(df_flagged['Cluster'].unique().tolist()),
        )

    # Build display columns
    display_cols = ['Cluster', 'Investigation_Status'] + selected_features
    if 'Suspicion_Reasons' in df_flagged.columns:
        display_cols.append('Suspicion_Reasons')
    if has_labels and label_col in df_flagged.columns:
        display_cols.append(label_col)

    filtered = df_flagged[
        (df_flagged['Investigation_Status'].isin(status_filter)) &
        (df_flagged['Cluster'].isin(cluster_filter))
    ][display_cols]

    st.markdown(f"Showing **{len(filtered):,}** of **{len(df_flagged):,}** records")
    st.dataframe(filtered.head(500), use_container_width=True, height=400)

    if len(filtered) > 500:
        st.info("Displaying first 500 rows. Download the full results below.")

    # Download button
    section_header("Download Results")
    csv_download = df_flagged[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis Results (CSV)",
        data=csv_download,
        file_name="safetraffic_analysis_results.csv",
        mime="text/csv",
    )


# =====================================================================
# PAGE: Model Evaluation
# =====================================================================

elif page == "📈 Model Evaluation":
    st.markdown("# 📈 Model Evaluation")
    st.markdown(data_badge, unsafe_allow_html=True)

    info_box(
        "<strong>Important:</strong> K-Means is an unsupervised algorithm. Cluster IDs "
        "are arbitrary integers and do NOT directly correspond to attack classes. "
        "We use <em>permutation-invariant</em> metrics (ARI, NMI) that correctly compare "
        "cluster assignments against true labels regardless of label numbering."
    )

    # Unsupervised metrics (always available)
    if st.session_state.get('sil_key') != pipeline_cache_key:
        sil_score = compute_silhouette(df_scaled, labels)
        st.session_state['sil_score'] = sil_score
        st.session_state['sil_key'] = pipeline_cache_key
    else:
        sil_score = st.session_state.get('sil_score')

    unsup_summary = unsupervised_evaluation_summary(
        inertia=model.inertia_,
        silhouette=sil_score,
        k=k_value,
        n_samples=len(df_scaled),
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("K (Clusters)", unsup_summary['k'])
    with m_col2:
        st.metric("Inertia", f"{unsup_summary['inertia']:,.0f}")
    with m_col3:
        sil_display = f"{unsup_summary['silhouette_score']:.4f}" if unsup_summary['silhouette_score'] is not None else "N/A"
        st.metric("Silhouette Score", sil_display)

    st.info(unsup_summary['note'])

    # Supervised evaluation (only when labels exist)
    if has_labels:
        section_header("Supervised Evaluation (Ground-Truth Comparison)")

        true_labels = df_clean[label_col]
        metrics = evaluate_with_labels(true_labels, labels)

        st.markdown("#### External Clustering Metrics")
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.metric(
                "Adjusted Rand Index (ARI)",
                f"{metrics['adjusted_rand_index']:.4f}",
                help="Measures agreement between clusters and true labels, adjusted for chance. Range: -1 to 1. Higher = better.",
            )
        with e_col2:
            st.metric(
                "Normalized Mutual Info (NMI)",
                f"{metrics['normalized_mutual_info']:.4f}",
                help="Measures mutual information between clusters and true labels, normalized. Range: 0 to 1. Higher = better.",
            )

        info_box(
            "<strong>ARI</strong> (Adjusted Rand Index): Measures how well cluster "
            "assignments agree with true labels, adjusted for random chance. "
            "A score of 1 = perfect agreement; 0 = random; negative = worse than random.<br><br>"
            "<strong>NMI</strong> (Normalized Mutual Information): Measures the mutual "
            "dependence between cluster IDs and true labels. 1 = perfect correlation; 0 = independent."
        )

        # Cluster-label mapping
        section_header("Cluster ↔ Label Mapping")
        mapping_df = get_cluster_label_mapping(true_labels, labels)
        st.dataframe(mapping_df, use_container_width=True)
        info_box(
            "This table shows the <em>dominant</em> true label in each cluster. "
            "Since cluster IDs are arbitrary, this mapping helps interpret what "
            "type of traffic ended up in each cluster."
        )

        # Label distribution
        section_header("Label Distribution in Dataset")
        label_dist = pd.DataFrame(
            list(metrics['label_distribution'].items()),
            columns=['Label', 'Count']
        ).sort_values('Count', ascending=False)
        st.dataframe(label_dist, use_container_width=True)

        # Heatmap
        section_header("True Labels vs. Cluster Assignments (Heatmap)")
        fig_heatmap = plot_label_cluster_heatmap(metrics['cluster_label_crosstab'])
        st.plotly_chart(fig_heatmap, use_container_width=True)
        info_box(
            "This heatmap shows how many records of each true label fall into each "
            "cluster. Ideally, each cluster would contain primarily one label. "
            "In practice, K-Means may mix labels because it groups by feature similarity, "
            "not by attack type."
        )
    else:
        section_header("Supervised Evaluation")
        st.warning(
            "⚠️ No ground-truth label column was detected in this dataset. "
            "Supervised evaluation metrics (ARI, NMI) require a label column. "
            "Without labels, attack-detection performance cannot be measured — "
            "only unsupervised clustering quality (inertia, silhouette) is available."
        )
        if is_demo:
            st.info(
                "💡 The demo dataset includes a 'Simulated Label' column. "
                "These are synthetic pattern labels used for demonstration."
            )


# =====================================================================
# PAGE: Algorithm Comparison (K-Means vs DBSCAN)
# =====================================================================

elif page == "⚖️ Algorithm Comparison (K-Means vs DBSCAN)":
    st.markdown("# ⚖️ Algorithm Comparison: K-Means vs. DBSCAN")
    st.markdown(data_badge, unsafe_allow_html=True)

    info_box(
        "<strong>Academic Comparison (DWM Mini-Project):</strong><br>"
        "In Data Warehousing & Mining (DWM), evaluating multiple clustering paradigms is essential. "
        "Here we compare a <strong>Centroid-Based Partitioning Algorithm (K-Means)</strong> against a "
        "<strong>Density-Based Spatial Clustering Algorithm (DBSCAN)</strong> on the exact same network traffic dataset."
    )

    # Compute comparative metrics
    true_labels_series = df_clean[label_col] if has_labels else None
    comp_res = compare_clustering_algorithms(
        df_scaled=df_scaled,
        kmeans_labels=labels,
        dbscan_labels=dbscan_labels,
        kmeans_outliers_count=total_suspicious,
        true_labels=true_labels_series,
    )

    # Top KPI comparison row
    section_header("Head-to-Head Performance Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Clusters Formed</div>
                <div class="kpi-value kpi-cyan" style="font-size:1.6rem;">
                    KM: {k_value} <span style="color:#78909c;font-size:1rem;">vs</span> DB: {comp_res['db_clusters']}
                </div>
                <div style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">Fixed K vs. Density-Discovered</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Anomalies Detected</div>
                <div class="kpi-value kpi-red" style="font-size:1.6rem;">
                    {comp_res['km_outliers_pct']}% <span style="color:#78909c;font-size:1rem;">vs</span> {comp_res['db_noise_pct']}%
                </div>
                <div style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">KM: {comp_res['km_outliers_cnt']:,} | DB: {comp_res['db_noise_cnt']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        km_sil_disp = f"{comp_res['km_sil']:.3f}" if comp_res['km_sil'] is not None else "N/A"
        db_sil_disp = f"{comp_res['db_sil']:.3f}" if comp_res['db_sil'] is not None else "N/A"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Silhouette Score</div>
                <div class="kpi-value kpi-green" style="font-size:1.6rem;">
                    {km_sil_disp} <span style="color:#78909c;font-size:1rem;">vs</span> {db_sil_disp}
                </div>
                <div style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">Compactness & Separation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        if has_labels and comp_res['km_ari'] is not None:
            db_ari_disp = f"{comp_res['db_ari']:.3f}" if comp_res['db_ari'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Ground-Truth ARI</div>
                    <div class="kpi-value kpi-amber" style="font-size:1.6rem;">
                        {comp_res['km_ari']:.3f} <span style="color:#78909c;font-size:1rem;">vs</span> {db_ari_disp}
                    </div>
                    <div style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">Agreement with Attack Labels</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            db_disp = f"{comp_res['km_db']:.2f}" if comp_res['km_db'] is not None else "N/A"
            db_db_disp = f"{comp_res['db_db']:.2f}" if comp_res['db_db'] is not None else "N/A"
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Davies-Bouldin Index</div>
                    <div class="kpi-value kpi-amber" style="font-size:1.6rem;">
                        {db_disp} <span style="color:#78909c;font-size:1rem;">vs</span> {db_db_disp}
                    </div>
                    <div style="font-size:0.75rem;color:#90a4ae;margin-top:4px;">Lower = Better Separation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Side-by-side scatter plot
    section_header("Side-by-Side Structural Comparison (2D PCA Projection)")
    fig_comp_scatter = plot_algorithm_comparison_scatter(
        df_scaled, labels, dbscan_labels, selected_features
    )
    st.plotly_chart(fig_comp_scatter, use_container_width=True)

    info_box(
        "🔍 <strong>Key Visual Takeaways for Faculty Presentation:</strong><br>"
        "• <strong>Left (K-Means):</strong> Notice how K-Means partitions the feature space into spherical voronoi cells. "
        "It forces every point — even distant outliers — to belong to one of the K clusters.<br>"
        "• <strong>Right (DBSCAN):</strong> Notice how DBSCAN groups dense traffic cores while leaving sparse, isolated records "
        "as <strong>Noise / Anomalies (red crosses)</strong> without forcing them into a normal cluster!"
    )

    # Bar chart of quantitative metrics
    section_header("Quantitative Metrics Comparison")
    fig_metrics = plot_comparison_metrics_bar(comp_res)
    st.plotly_chart(fig_metrics, use_container_width=True)

    # Full Comparison Matrix Table
    section_header("Comprehensive Algorithmic Evaluation Matrix")
    st.dataframe(comp_res['comparison_df'], use_container_width=True, height=360)

    # Comparative Analysis Summary & Key Takeaways
    section_header("Comparative Analysis Summary & Key Takeaways")
    info_box(
        f"<strong>Key Takeaways from the Algorithmic Comparison:</strong><br><br>"
        f"• <strong>Clustering Paradigms:</strong> K-Means is a partition-based algorithm that assumes convex, spherical clusters and requires specifying <em>K</em> in advance. "
        f"DBSCAN is a density-based algorithm that discovers clusters of arbitrary shapes based on local spatial density (ε={dbscan_eps}, MinPts={dbscan_min_samples}).<br><br>"
        f"• <strong>Anomaly Detection Purity:</strong> K-Means assigns all records to centroids, requiring post-hoc z-score thresholding to flag suspicious traffic ({comp_res['km_outliers_cnt']:,} records, {comp_res['km_outliers_pct']}%). "
        f"DBSCAN natively separates sparse, low-density records as intrinsic <strong>Noise (Cluster -1)</strong> ({comp_res['db_noise_cnt']:,} records, {comp_res['db_noise_pct']}%) during clustering without heuristic rules.<br><br>"
        f"• <strong>Computational Trade-offs:</strong> K-Means provides <code>O(k · n · i)</code> linear time complexity and explicit centroid profiles, making it well-suited for high-throughput baseline traffic segmentation. "
        f"DBSCAN provides higher anomaly purity for non-spherical attack distributions at the cost of <code>O(n²)</code> computational complexity.<br><br>"
        f"• <strong>Architectural Recommendation:</strong> In production Intrusion Detection Systems (IDS), a <em>hybrid architecture</em> is recommended: "
        f"utilize K-Means for high-throughput line-rate baseline segmentation, while routing boundary or flagged traffic to DBSCAN for density-based anomaly confirmation."
    )

