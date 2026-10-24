# SafeTraffic — Project Report

## Website Traffic Anomaly Detection Using K-Means Clustering

**Subject:** Data Warehousing and Mining (DWM)  
**Project Type:** Mini-Project  
**Algorithm:** K-Means Clustering (Unsupervised Learning)

---

## 1. Abstract

SafeTraffic is a web-based application that analyzes network traffic records to identify potentially suspicious activity using K-Means clustering. The system processes network flow data, groups similar traffic patterns into clusters based on numerical features (such as packet counts, byte counts, flow durations, and transfer rates), and flags records with statistically unusual characteristics for further investigation. The application provides an interactive dashboard built with Streamlit and Python, offering visualizations, explainable flagging, and evaluation metrics. It supports both a synthetic demo dataset for immediate demonstration and real-world CIC-IDS2017 network intrusion detection data.

---

## 2. Problem Statement

Network security teams face the challenge of analyzing large volumes of network traffic to detect potentially malicious activity. Manual inspection is impractical for modern networks that generate millions of flow records daily. Traditional signature-based intrusion detection systems can only detect known attack patterns and may miss novel threats.

**Research Question:** Can unsupervised K-Means clustering effectively group network traffic into meaningful patterns and help identify traffic records with unusual characteristics that warrant security investigation?

---

## 3. Objectives

1. Design and implement a data preprocessing pipeline for network traffic datasets.
2. Apply K-Means clustering to group network flows based on numerical traffic features.
3. Use the Elbow Method and Silhouette Score to evaluate and select the number of clusters.
4. Develop a transparent, rule-based method for flagging statistically unusual traffic records.
5. Compare cluster assignments with ground-truth attack labels (when available) using appropriate external clustering metrics.
6. Build an interactive web dashboard for visualization and exploration of results.

---

## 4. Methodology

### 4.1 Dataset

**Primary Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- Contains network traffic from 5 days of operation.
- Includes both benign traffic and multiple attack types.
- Approximately 80 network flow features per record.
- Source: https://www.unb.ca/cic/datasets/ids-2017.html

**Demo Dataset:** Synthetic dataset with 2,000 records simulating four traffic patterns (normal browsing, high-volume transfer, repeated connections, unusual scanning).

### 4.2 Data Preprocessing

| Step | Action | Purpose |
|------|--------|---------|
| 1 | Strip column name whitespace | Fix CIC-IDS2017 formatting issues |
| 2 | Convert numeric strings | Ensure all features are numeric |
| 3 | Replace inf values with NaN | Remove infinite values |
| 4 | Drop NaN rows | Ensure complete records for clustering |
| 5 | Remove duplicates | Avoid biasing clusters |
| 6 | Log-transform skewed features | Reduce impact of extreme skewness |
| 7 | StandardScaler normalization | Ensure equal feature contribution to distance |

### 4.3 K-Means Clustering Algorithm

**Algorithm Steps:**
1. Select K (number of clusters).
2. Initialize K centroids randomly.
3. Assign each record to the nearest centroid (Euclidean distance).
4. Recompute centroids as the mean of assigned records.
5. Repeat steps 3–4 until convergence.

**Parameters:**
- `n_clusters`: User-configurable (2–15)
- `random_state`: 42 (for reproducibility)
- `n_init`: 10 (multiple random starts)
- `max_iter`: 300

### 4.4 Cluster Quality Evaluation

**Elbow Method:**
- Plot inertia (WCSS) for K = 2 to 12.
- Select K at the "elbow" point.

**Silhouette Score:**
- Measures how well each record fits within its cluster.
- Range: -1 (poor) to +1 (excellent).
- Computed on a sample for large datasets.

### 4.5 Suspicious Traffic Identification

**Method:** Z-score thresholding (post-clustering)
- Compute z-score for each feature across all records.
- Flag records where |z| > threshold (default: 2.5).
- Categorize as "Needs Investigation" (1–2 flagged features) or "High Priority" (3+ flagged features).

**Key Principle:** K-Means does NOT determine which clusters are attacks. The flagging is a separate, transparent, and configurable heuristic.

### 4.6 Evaluation Against Ground Truth

When true labels (e.g., "BENIGN," "DDoS") are available:
- **Adjusted Rand Index (ARI):** Measures agreement, adjusted for chance (-1 to 1).
- **Normalized Mutual Information (NMI):** Measures mutual dependence (0 to 1).
- **Cross-tabulation:** Cluster vs. true label frequency table.

> Classification accuracy is NOT reported because K-Means is unsupervised and cluster IDs are arbitrary.

---

## 5. System Architecture

```
User Interface (Streamlit Dashboard)
         │
    Visualization Module (Plotly)
         │
    Anomaly Detection (Z-score rules)
         │
    K-Means Clustering (Scikit-learn)
         │
    Preprocessing (StandardScaler)
         │
    Data Loader (CSV / Demo Generator)
```

**Technology Stack:**
- Python 3, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib

---

## 6. Results

> **Note:** This section should be filled with actual experimental outputs after running the application. Screenshots and metric values should be captured during the demonstration.

### 6.1 Demo Dataset Results

| Metric | Value |
|--------|-------|
| Total Records | *(fill after running)* |
| Number of Clusters (K) | *(fill after running)* |
| Silhouette Score | *(fill after running)* |
| Records Flagged | *(fill after running)* |
| Flagged Percentage | *(fill after running)* |
| ARI (vs. Simulated Labels) | *(fill after running)* |
| NMI (vs. Simulated Labels) | *(fill after running)* |

### 6.2 Elbow Method Plot

*(Insert screenshot of Elbow Method chart from the application)*

### 6.3 Cluster Visualization

*(Insert screenshot of 2D cluster scatter plot)*

### 6.4 Suspicious Traffic Summary

| Investigation Status | Count |
|---------------------|-------|
| Normal | *(fill after running)* |
| Needs Investigation | *(fill after running)* |
| High Priority | *(fill after running)* |

### 6.5 CIC-IDS2017 Results (if tested)

*(Insert results from real dataset analysis)*

---

## 7. Conclusion

SafeTraffic demonstrates that K-Means clustering can effectively group network traffic records into meaningful clusters based on their numerical characteristics. The Elbow Method and Silhouette Score provide guidance for selecting the number of clusters. Post-clustering z-score analysis identifies records with statistically unusual feature values that may indicate suspicious activity.

However, K-Means has limitations: it assumes spherical clusters, requires manual K selection, and cannot inherently distinguish attacks from benign traffic. The flagging heuristic is a starting point for investigation, not a definitive security verdict.

The project successfully illustrates the application of unsupervised data mining techniques to a real-world cybersecurity problem, making it suitable for academic demonstration of DWM concepts.

---

## 8. Future Scope

1. **DBSCAN Clustering** — Density-based clustering for irregularly shaped attack patterns.
2. **Automated K Selection** — Gap Statistic or information-theoretic methods.
3. **Real-Time Monitoring** — Integration with live network capture tools.
4. **Temporal Analysis** — Time-series clustering for detecting attack sequences.
5. **Ensemble Methods** — Combining multiple clustering algorithms for robustness.
6. **Feature Engineering** — Automated feature selection and dimensionality reduction.
7. **Deep Learning** — Autoencoders for representation learning before clustering.

---

## 9. References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP*.
2. CIC-IDS2017 Dataset — https://www.unb.ca/cic/datasets/ids-2017.html
3. Scikit-learn Documentation — https://scikit-learn.org/stable/modules/clustering.html#k-means
4. Streamlit Documentation — https://docs.streamlit.io/
5. MacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations. *Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability*.

---

*SafeTraffic — Data Warehousing and Mining Mini-Project*
