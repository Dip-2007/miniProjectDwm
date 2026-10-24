# 🛡️ SafeTraffic — Website Traffic Anomaly Detection Using K-Means Clustering

## 📋 Table of Contents

1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Scope and Limitations](#scope-and-limitations)
5. [Dataset Description](#dataset-description)
6. [Technologies Used](#technologies-used)
7. [System Architecture](#system-architecture)
8. [Data Preprocessing](#data-preprocessing)
9. [K-Means Algorithm](#k-means-algorithm)
10. [Elbow Method](#elbow-method)
11. [Silhouette Score](#silhouette-score)
12. [Suspicious Cluster Identification](#suspicious-cluster-identification)
13. [Evaluation Methodology](#evaluation-methodology)
14. [Installation Instructions](#installation-instructions)
15. [Running the Application](#running-the-application)
16. [Demonstration Workflow](#demonstration-workflow)
17. [Limitations and Future Scope](#limitations-and-future-scope)

---

## Abstract

**SafeTraffic** is an interactive web application that analyzes network traffic records and identifies potentially suspicious traffic patterns using **K-Means clustering**, an unsupervised data mining algorithm. The application processes network flow data (such as the CIC-IDS2017 dataset), groups similar traffic patterns into clusters, and flags records with statistically unusual feature values for investigation. Built with Python, Streamlit, and Scikit-learn, SafeTraffic provides a beginner-friendly dashboard with interactive visualizations, making it suitable for academic demonstration and understanding of clustering concepts in the context of cybersecurity.

---

## Problem Statement

Modern computer networks generate enormous amounts of traffic data. Manually inspecting every network flow record to identify potentially malicious activity is impractical. Traditional rule-based intrusion detection systems rely on known attack signatures and may miss novel threats. There is a need for data-driven approaches that can automatically group similar traffic patterns and highlight unusual behavior for further investigation.

This project addresses the question: **Can unsupervised clustering (K-Means) identify groups of network traffic with unusual characteristics that may warrant security investigation?**

---

## Objectives

1. Implement a complete **K-Means clustering pipeline** for network traffic analysis.
2. Preprocess real-world network traffic data (CIC-IDS2017) to handle missing values, infinite values, duplicates, and feature scaling.
3. Apply the **Elbow Method** and **Silhouette Score** to evaluate clustering quality.
4. Flag traffic records with statistically unusual feature values using **transparent, explainable rules** (z-score thresholds).
5. Provide an interactive dashboard for exploring clusters, features, and suspicious records.
6. Compare clustering results against ground-truth labels (when available) using **permutation-invariant metrics** (ARI, NMI).
7. Include a demo mode with synthetic data for immediate demonstration.

---

## Scope and Limitations

### In Scope
- Offline analysis of pre-collected network traffic CSV files.
- K-Means clustering with configurable parameters.
- Statistical anomaly flagging using z-score thresholds.
- Interactive visualization and exploration.
- Academic demonstration and explanation of DWM concepts.

### Limitations
- **K-Means is not a classifier.** It does not know which clusters are attacks. Flagging is a separate heuristic step.
- **K-Means assumes spherical, equally-sized clusters.** Real network traffic may not follow this assumption.
- **Not real-time.** The application analyzes uploaded files, not live network traffic.
- **Z-score flagging is a heuristic.** It can produce false positives and false negatives.
- **The number of clusters (K) must be chosen by the user.** There is no universally "correct" K.
- **Scalability.** Very large datasets (millions of rows) may be slow on a student laptop.

---

## Dataset Description

### CIC-IDS2017

- **Source:** [Canadian Institute for Cybersecurity — IDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Description:** Network traffic captured over 5 days (Monday–Friday), containing both benign traffic and common cyberattacks (DDoS, Port Scan, Brute Force, Web Attacks, etc.).
- **Format:** CSV files with ~80 network flow features extracted using CICFlowMeter.
- **Key Features Used:**
  - `Flow Duration` — Duration of the network flow in microseconds.
  - `Total Fwd Packets` / `Total Backward Packets` — Packet counts in each direction.
  - `Total Length of Fwd Packets` / `Total Length of Bwd Packets` — Total bytes in each direction.
  - `Flow Bytes/s` — Data transfer rate.
  - `Flow Packets/s` — Packet transfer rate.
  - `Flow IAT Mean` — Mean inter-arrival time between packets.
  - `Fwd Packet Length Mean` / `Bwd Packet Length Mean` — Average packet size.
  - `Packet Length Variance` — Variance in packet lengths.
  - `Average Packet Size` — Overall average packet size.

### Demo Dataset

A synthetic dataset with 2,000 records simulating four traffic patterns:
1. **Normal Browsing** — Typical web browsing flows (60%).
2. **High Volume Transfer** — Large file transfers or streaming (20%).
3. **Repeated Connections** — Many short-lived connections (12%).
4. **Unusual Scanning** — Slow, varied probing patterns (8%).

> ⚠️ The demo dataset is **simulated** and does NOT represent real network attacks.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Programming language |
| Streamlit | Web-based dashboard interface |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Scikit-learn | K-Means clustering, StandardScaler, evaluation metrics |
| Plotly | Interactive charts and visualizations |
| Matplotlib | Static plots (supporting role) |
| Joblib | Model serialization (optional) |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    USER INTERFACE (Streamlit)             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│  │Dashboard │  │ Dataset  │  │Clustering│  │Suspicious││
│  │ Overview │  │ Explorer │  │ Analysis │  │ Traffic  ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│       │              │              │              │      │
│  ┌────┴──────────────┴──────────────┴──────────────┴────┐│
│  │              VISUALIZATION MODULE (Plotly)            ││
│  └──────────────────────┬───────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴───────────────────────────────┐│
│  │           ANOMALY DETECTION (z-score rules)          ││
│  └──────────────────────┬───────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴───────────────────────────────┐│
│  │            K-MEANS CLUSTERING (Scikit-learn)         ││
│  └──────────────────────┬───────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴───────────────────────────────┐│
│  │         PREPROCESSING (clean, scale, validate)       ││
│  └──────────────────────┬───────────────────────────────┘│
│                         │                                │
│  ┌──────────────────────┴───────────────────────────────┐│
│  │          DATA LOADER (CSV / Demo Generator)          ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

The preprocessing pipeline performs these steps:

1. **Load Data** — Read CSV files, handle encoding issues, strip whitespace from column names.
2. **Inspect** — Report rows, columns, data types, missing values, and duplicates.
3. **Convert Types** — Coerce numeric strings into numeric values.
4. **Handle Infinities** — Replace `inf` and `-inf` values with `NaN`.
5. **Handle Missing Values** — Drop rows with `NaN` in the selected feature columns.
6. **Remove Duplicates** — Remove duplicate rows (reporting how many were removed).
7. **Handle Skewness** — Apply `log1p` transformation to features with extreme skewness (|skewness| > 5) and non-negative values.
8. **Feature Scaling** — Apply `StandardScaler` to normalize features to zero mean and unit variance.

> **Why scale?** K-Means uses Euclidean distance. Without scaling, features with larger magnitudes would dominate the distance calculation.

> **Data Leakage Prevention:** Attack labels (e.g., the `Label` column) are NEVER used as clustering inputs. They are preserved separately for evaluation only.

---

## K-Means Algorithm

### How K-Means Works (Simple Explanation)

1. **Choose K** — Decide how many groups (clusters) you want.
2. **Initialize** — Place K random "centre points" (centroids) in the feature space.
3. **Assign** — Assign each data point to the nearest centroid.
4. **Update** — Move each centroid to the average position of all points assigned to it.
5. **Repeat** — Continue steps 3-4 until centroids stop moving (converge).

### Example

Imagine plotting network flows by "packet count" and "flow duration." Flows with similar packet counts and durations will naturally cluster together. Flows that are far from any cluster centre may represent unusual traffic worth investigating.

### Parameters Used

| Parameter | Value | Reason |
|---|---|---|
| `n_clusters` | User-selected (2–15) | Number of clusters |
| `random_state` | 42 | Reproducible results |
| `n_init` | 10 | Run K-Means 10 times with different seeds, keep the best result |
| `max_iter` | 300 | Maximum iterations per run |

---

## Elbow Method

The Elbow Method helps choose a good value of K:

1. Run K-Means for K = 2, 3, 4, ..., 12.
2. For each K, record the **inertia** (total within-cluster sum of squared distances).
3. Plot K vs. inertia.
4. Look for the "elbow" — the point where adding more clusters stops significantly reducing inertia.

> The elbow is a **guideline**, not a definitive answer. Some datasets may not show a clear elbow.

---

## Silhouette Score

The Silhouette Score measures clustering quality:

- For each point, compute:
  - **a** = average distance to other points in the same cluster.
  - **b** = average distance to points in the nearest other cluster.
  - **silhouette** = (b - a) / max(a, b)

- **Score range:** -1 to 1
  - **Near 1:** Point is well inside its cluster.
  - **Near 0:** Point is on the boundary between two clusters.
  - **Near -1:** Point may be assigned to the wrong cluster.

> For large datasets, the Silhouette Score is computed on a random sample (up to 10,000 points) for efficiency.

---

## Suspicious Cluster Identification

After clustering, records are flagged using **z-score thresholds**:

1. For each selected feature, compute the **z-score** of every record: `z = (value - mean) / std_dev`.
2. If **any** feature has `|z| > threshold` (default: 2.5), the record is flagged.
3. Flagged records are categorized:
   - **Needs Investigation** — 1–2 features are unusual.
   - **High Priority** — 3+ features are unusual.

### Transparency

- Each flagged record shows **which features** triggered the flag and by how much.
- The threshold is **configurable** by the user.
- Flagging does NOT mean the traffic is malicious — it means the record is statistically unusual and warrants human investigation.

---

## Evaluation Methodology

### When Ground-Truth Labels Are Available

| Metric | Description | Range |
|---|---|---|
| **Adjusted Rand Index (ARI)** | Agreement between cluster IDs and true labels, adjusted for chance | -1 to 1 |
| **Normalized Mutual Information (NMI)** | Mutual dependence between clusters and labels | 0 to 1 |

> We do NOT report "accuracy" because K-Means is unsupervised and cluster IDs are arbitrary.

### When Labels Are Not Available

Only unsupervised metrics are reported:
- **Inertia** — Lower = tighter clusters.
- **Silhouette Score** — Higher = better-separated clusters.

> Without labels, we cannot evaluate attack-detection performance.

---

## Installation Instructions

### Prerequisites

- **Python 3.9+** installed on your system.
- **pip** package manager.

### Steps (Windows PowerShell)

```powershell
# 1. Navigate to the project directory
cd D:\DWM-MiniProject

# 2. Create a virtual environment (recommended)
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```powershell
# Make sure the virtual environment is activated
.\venv\Scripts\Activate.ps1

# Run the Streamlit app
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## Demonstration Workflow

Follow these steps for a faculty demo:

### Step 1: Launch in Demo Mode
1. Run `streamlit run app.py`.
2. The app starts in **Demo Mode** by default — synthetic data is loaded automatically.
3. A banner clearly indicates that demo data is being used.

### Step 2: Dashboard Overview
- Show the KPI cards: total records, clusters, flagged count, flagged percentage.
- Show the traffic status pie chart and cluster size chart.
- Explain: "K-Means grouped 2,000 traffic records into clusters based on their feature similarity."

### Step 3: Dataset Explorer
- Show the raw data preview, column types, and missing-value summary.
- Show the preprocessing report (before vs. after).
- Explain: "We cleaned the data by removing missing values and scaling the features."

### Step 4: Clustering Analysis
- Show the **Elbow Method** chart and explain the concept.
- Show the **Silhouette Score** and its interpretation.
- Show the **2D cluster scatter plot** (PCA projection).
- Show **cluster centroids** and explain what each cluster looks like.
- Explain: "Cluster 0 has small packet counts and short durations — normal browsing. Cluster 2 has very high byte counts — bulk transfers."

### Step 5: Suspicious Traffic
- Show the flagged records table.
- Filter by "High Priority" and show the suspicion reasons.
- Explain: "These records have feature values far outside the normal range. They need human investigation."
- Download the results as CSV.

### Step 6: Model Evaluation
- Show ARI and NMI scores.
- Show the heatmap of true labels vs. clusters.
- Explain: "The ARI of X shows moderate/good agreement between our clusters and the known traffic types."

### Step 7: Upload Real Data (Optional)
- Switch to Upload Mode and upload a CIC-IDS2017 CSV file.
- Show that the entire pipeline runs on real data.

---

## Limitations and Future Scope

### Current Limitations
1. K-Means assumes spherical clusters — density-based methods (DBSCAN) might better capture irregular attack patterns.
2. The z-score flagging threshold is a simple heuristic.
3. No real-time monitoring capability.
4. K must be manually selected.
5. Very large files may be slow.

### Future Scope
1. Implement **DBSCAN** as an alternative clustering algorithm.
2. Add **automated K selection** using Gap Statistic or other methods.
3. Integrate with **live network monitoring** tools (Zeek, Suricata).
4. Add **time-series analysis** for detecting temporal attack patterns.
5. Implement **ensemble methods** combining multiple clustering algorithms.
6. Add **user authentication** for multi-user deployment.

---

## Project Structure

```
SafeTraffic/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── PROJECT_REPORT.md         # Detailed academic report
├── data/                     # Data directory (for uploaded files)
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and demo generation
│   ├── preprocessing.py      # Data cleaning and scaling
│   ├── clustering.py         # K-Means implementation
│   ├── anomaly_detection.py  # Suspicious traffic flagging
│   ├── evaluation.py         # Clustering evaluation metrics
│   └── visualizations.py     # Plotly chart generators
└── tests/                    # Unit tests
    └── test_safetraffic.py   # Tests for all modules
```

---

*SafeTraffic — A DWM Mini-Project for academic demonstration of K-Means clustering in network traffic analysis.*
