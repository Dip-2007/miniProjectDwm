"""
SafeTraffic - Data Loader Module
Handles loading CSV files and generating synthetic demo datasets.
"""

import pandas as pd
import numpy as np
import streamlit as st


def generate_demo_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a small synthetic dataset that simulates different network traffic
    patterns for demonstration purposes.

    Traffic Patterns Simulated (NOT real attacks):
    1. Normal Browsing  - typical web browsing traffic
    2. High Volume      - bulk data transfer / streaming
    3. Repeated Connects - many short connections (e.g., scanning-like)
    4. Unusual Scanning  - slow, varied probing-like traffic

    Returns a DataFrame with columns similar to CIC-IDS2017.
    """
    rng = np.random.RandomState(random_state)

    # Proportions: ~60% normal, ~20% high-volume, ~12% repeated, ~8% scanning
    n_normal = int(n_samples * 0.60)
    n_high = int(n_samples * 0.20)
    n_repeat = int(n_samples * 0.12)
    n_scan = n_samples - n_normal - n_high - n_repeat

    records = []

    # --- Pattern 1: Normal Browsing ---
    records.append(pd.DataFrame({
        'Flow Duration': rng.exponential(50000, n_normal).astype(int),
        'Total Fwd Packets': rng.poisson(8, n_normal),
        'Total Backward Packets': rng.poisson(10, n_normal),
        'Total Length of Fwd Packets': rng.exponential(1500, n_normal).astype(int),
        'Total Length of Bwd Packets': rng.exponential(5000, n_normal).astype(int),
        'Flow Bytes/s': rng.exponential(50000, n_normal),
        'Flow Packets/s': rng.exponential(200, n_normal),
        'Flow IAT Mean': rng.exponential(8000, n_normal),
        'Fwd Packet Length Mean': rng.exponential(200, n_normal),
        'Bwd Packet Length Mean': rng.exponential(500, n_normal),
        'Packet Length Variance': rng.exponential(30000, n_normal),
        'Average Packet Size': rng.exponential(350, n_normal),
        'Simulated Label': 'Normal Browsing'
    }))

    # --- Pattern 2: High Volume Transfer ---
    records.append(pd.DataFrame({
        'Flow Duration': rng.exponential(500000, n_high).astype(int),
        'Total Fwd Packets': rng.poisson(200, n_high),
        'Total Backward Packets': rng.poisson(250, n_high),
        'Total Length of Fwd Packets': rng.exponential(100000, n_high).astype(int),
        'Total Length of Bwd Packets': rng.exponential(500000, n_high).astype(int),
        'Flow Bytes/s': rng.exponential(800000, n_high),
        'Flow Packets/s': rng.exponential(600, n_high),
        'Flow IAT Mean': rng.exponential(2000, n_high),
        'Fwd Packet Length Mean': rng.exponential(800, n_high),
        'Bwd Packet Length Mean': rng.exponential(1400, n_high),
        'Packet Length Variance': rng.exponential(200000, n_high),
        'Average Packet Size': rng.exponential(1000, n_high),
        'Simulated Label': 'High Volume Transfer'
    }))

    # --- Pattern 3: Repeated Short Connections (Scanning-like) ---
    records.append(pd.DataFrame({
        'Flow Duration': rng.exponential(500, n_repeat).astype(int),
        'Total Fwd Packets': rng.poisson(2, n_repeat) + 1,
        'Total Backward Packets': rng.poisson(1, n_repeat),
        'Total Length of Fwd Packets': rng.exponential(100, n_repeat).astype(int),
        'Total Length of Bwd Packets': rng.exponential(50, n_repeat).astype(int),
        'Flow Bytes/s': rng.exponential(200000, n_repeat),
        'Flow Packets/s': rng.exponential(3000, n_repeat),
        'Fwd Packet Length Mean': rng.exponential(50, n_repeat),
        'Bwd Packet Length Mean': rng.exponential(30, n_repeat),
        'Flow IAT Mean': rng.exponential(200, n_repeat),
        'Packet Length Variance': rng.exponential(5000, n_repeat),
        'Average Packet Size': rng.exponential(60, n_repeat),
        'Simulated Label': 'Repeated Connections'
    }))

    # --- Pattern 4: Unusual Probing / Slow Scan ---
    records.append(pd.DataFrame({
        'Flow Duration': rng.exponential(2000000, n_scan).astype(int),
        'Total Fwd Packets': rng.poisson(3, n_scan) + 1,
        'Total Backward Packets': rng.poisson(1, n_scan),
        'Total Length of Fwd Packets': rng.exponential(300, n_scan).astype(int),
        'Total Length of Bwd Packets': rng.exponential(100, n_scan).astype(int),
        'Flow Bytes/s': rng.exponential(500, n_scan),
        'Flow Packets/s': rng.exponential(5, n_scan),
        'Flow IAT Mean': rng.exponential(500000, n_scan),
        'Fwd Packet Length Mean': rng.exponential(80, n_scan),
        'Bwd Packet Length Mean': rng.exponential(60, n_scan),
        'Packet Length Variance': rng.exponential(8000, n_scan),
        'Average Packet Size': rng.exponential(70, n_scan),
        'Simulated Label': 'Unusual Scanning'
    }))

    df = pd.concat(records, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


# Columns that should NOT be used as clustering features
EXCLUDE_COLUMNS = {
    'flow id', 'source ip', 'src ip', 'source port', 'src port',
    'destination ip', 'dst ip', 'destination port', 'dst port',
    'protocol', 'timestamp', 'label', 'simulated label',
    'external ip',
}


import os

def list_local_datasets(data_dir: str = "data") -> list[str]:
    """
    List all CSV files present in the local data directory.
    """
    if not os.path.exists(data_dir):
        return []
    files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith('.csv')
    ]
    return sorted(files)


def load_csv_path(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file directly from local disk path.
    Handles encoding and strips column names.
    """
    try:
        df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)

    df.columns = df.columns.str.strip()
    return df


def load_csv_file(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV file uploaded through Streamlit's file uploader.
    Handles common CIC-IDS2017 issues:
      - BOM markers
      - Leading/trailing spaces in column names
      - Mixed encodings
    """
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    return df


def detect_label_column(df: pd.DataFrame) -> str | None:
    """
    Detect the ground-truth label column if it exists.
    CIC-IDS2017 uses ' Label' or 'Label'.
    Demo data uses 'Simulated Label'.
    """
    for col in df.columns:
        if col.strip().lower() in ('label', 'simulated label'):
            return col
    return None


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a list of numeric columns suitable for clustering,
    excluding identifiers and labels.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    suitable = [
        c for c in numeric_cols
        if c.strip().lower() not in EXCLUDE_COLUMNS
    ]
    return suitable
