"""
Tests for SafeTraffic — preprocessing, clustering, and evaluation modules.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

# Add project root to path so imports work when running from project root
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import generate_demo_dataset, detect_label_column, get_numeric_feature_columns
from src.preprocessing import preprocess_dataframe, validate_features, get_preprocessing_summary
from src.clustering import run_kmeans, compute_elbow_data, compute_silhouette, cluster_sizes
from src.anomaly_detection import flag_suspicious_records, flag_suspicious_clusters
from src.evaluation import evaluate_with_labels, unsupervised_evaluation_summary


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def demo_df():
    """Small demo dataset for testing."""
    return generate_demo_dataset(n_samples=500, random_state=42)


@pytest.fixture
def feature_cols(demo_df):
    """Numeric feature columns from the demo dataset."""
    return get_numeric_feature_columns(demo_df)


@pytest.fixture
def preprocessed(demo_df, feature_cols):
    """Preprocessed demo data."""
    df_clean, df_scaled, scaler, report = preprocess_dataframe(
        demo_df, feature_cols
    )
    return df_clean, df_scaled, scaler, report


# =====================================================================
# Data Loader Tests
# =====================================================================

class TestDataLoader:
    def test_demo_dataset_shape(self, demo_df):
        assert len(demo_df) == 500
        assert len(demo_df.columns) >= 10

    def test_demo_dataset_reproducible(self):
        df1 = generate_demo_dataset(n_samples=100, random_state=42)
        df2 = generate_demo_dataset(n_samples=100, random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_label_column_detected(self, demo_df):
        label_col = detect_label_column(demo_df)
        assert label_col == 'Simulated Label'

    def test_numeric_features_exclude_labels(self, demo_df, feature_cols):
        for col in feature_cols:
            assert col.strip().lower() not in ('label', 'simulated label')

    def test_numeric_features_not_empty(self, feature_cols):
        assert len(feature_cols) >= 2


# =====================================================================
# Preprocessing Tests
# =====================================================================

class TestPreprocessing:
    def test_preprocessing_produces_scaled_data(self, preprocessed, feature_cols):
        df_clean, df_scaled, scaler, report = preprocessed
        assert df_scaled.shape[1] == len(feature_cols)
        assert len(df_clean) == len(df_scaled)

    def test_no_nan_after_preprocessing(self, preprocessed):
        df_clean, df_scaled, scaler, report = preprocessed
        assert df_scaled.isnull().sum().sum() == 0

    def test_no_inf_after_preprocessing(self, preprocessed, feature_cols):
        df_clean, df_scaled, scaler, report = preprocessed
        assert not np.isinf(df_clean[feature_cols].values).any()

    def test_report_contains_required_keys(self, preprocessed):
        _, _, _, report = preprocessed
        required_keys = [
            'initial_rows', 'initial_columns', 'features_selected',
            'inf_replaced', 'nan_dropped', 'duplicates_removed',
            'final_rows',
        ]
        for key in required_keys:
            assert key in report

    def test_validation_rejects_empty_features(self, demo_df):
        valid, msg = validate_features(demo_df, [])
        assert not valid

    def test_validation_rejects_single_feature(self, demo_df, feature_cols):
        valid, msg = validate_features(demo_df, feature_cols[:1])
        assert not valid

    def test_validation_accepts_valid_features(self, demo_df, feature_cols):
        valid, msg = validate_features(demo_df, feature_cols[:4])
        assert valid

    def test_summary_structure(self, demo_df):
        summary = get_preprocessing_summary(demo_df)
        assert summary['rows'] == 500
        assert 'missing_values' in summary

    def test_handles_inf_values(self, feature_cols):
        """Inject inf values and ensure preprocessing removes them."""
        df = generate_demo_dataset(n_samples=100, random_state=99)
        # Inject some infinities
        df.iloc[0, 0] = np.inf
        df.iloc[1, 1] = -np.inf
        df_clean, df_scaled, _, report = preprocess_dataframe(df, feature_cols)
        assert not np.isinf(df_clean[feature_cols].values).any()
        assert report['inf_replaced'] >= 2


# =====================================================================
# Clustering Tests
# =====================================================================

class TestClustering:
    def test_kmeans_produces_correct_labels(self, preprocessed):
        _, df_scaled, _, _ = preprocessed
        model, labels = run_kmeans(df_scaled, k=3)
        assert len(labels) == len(df_scaled)
        assert set(labels) == {0, 1, 2}

    def test_elbow_data_monotonic(self, preprocessed):
        _, df_scaled, _, _ = preprocessed
        inertias = compute_elbow_data(df_scaled, k_range=range(2, 6))
        vals = [inertias[k] for k in sorted(inertias.keys())]
        # Inertia should generally decrease (allow small tolerance)
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i-1] + 1e-6

    def test_silhouette_score_valid(self, preprocessed):
        _, df_scaled, _, _ = preprocessed
        _, labels = run_kmeans(df_scaled, k=3)
        score = compute_silhouette(df_scaled, labels)
        assert score is not None
        assert -1 <= score <= 1

    def test_silhouette_none_for_single_cluster(self, preprocessed):
        _, df_scaled, _, _ = preprocessed
        labels = np.zeros(len(df_scaled), dtype=int)
        score = compute_silhouette(df_scaled, labels)
        assert score is None

    def test_cluster_sizes_sum(self, preprocessed):
        _, df_scaled, _, _ = preprocessed
        _, labels = run_kmeans(df_scaled, k=4)
        sizes = cluster_sizes(labels)
        assert sizes['Size'].sum() == len(df_scaled)
        assert abs(sizes['Percentage'].sum() - 100) < 0.1


# =====================================================================
# Anomaly Detection Tests
# =====================================================================

class TestAnomalyDetection:
    def test_flagging_adds_columns(self, preprocessed, feature_cols):
        df_clean, df_scaled, _, _ = preprocessed
        _, labels = run_kmeans(df_scaled, k=3)
        result = flag_suspicious_records(df_clean, labels, feature_cols)
        assert 'Suspicious' in result.columns
        assert 'Investigation_Status' in result.columns
        assert 'Suspicion_Reasons' in result.columns

    def test_no_division_by_zero(self):
        """Constant-feature dataset should not crash."""
        df = pd.DataFrame({
            'A': [1.0] * 100,
            'B': np.random.randn(100),
        })
        labels = np.zeros(100, dtype=int)
        result = flag_suspicious_records(df, labels, ['A', 'B'])
        assert len(result) == 100

    def test_cluster_summary(self, preprocessed, feature_cols):
        df_clean, df_scaled, _, _ = preprocessed
        _, labels = run_kmeans(df_scaled, k=3)
        flagged = flag_suspicious_records(df_clean, labels, feature_cols)
        summary = flag_suspicious_clusters(flagged)
        assert 'Suspicious_Pct' in summary.columns
        assert len(summary) == 3


# =====================================================================
# Evaluation Tests
# =====================================================================

class TestEvaluation:
    def test_evaluation_with_labels(self, preprocessed, demo_df, feature_cols):
        df_clean, df_scaled, _, _ = preprocessed
        _, labels = run_kmeans(df_scaled, k=4)
        label_col = detect_label_column(demo_df)

        # Align labels with cleaned data
        true_labels = df_clean[label_col]
        metrics = evaluate_with_labels(true_labels, labels)

        assert 'adjusted_rand_index' in metrics
        assert 'normalized_mutual_info' in metrics
        assert -1 <= metrics['adjusted_rand_index'] <= 1
        assert 0 <= metrics['normalized_mutual_info'] <= 1

    def test_unsupervised_summary(self):
        summary = unsupervised_evaluation_summary(
            inertia=12345.67, silhouette=0.45, k=3, n_samples=500
        )
        assert summary['k'] == 3
        assert summary['silhouette_score'] == 0.45
        assert 'note' in summary
