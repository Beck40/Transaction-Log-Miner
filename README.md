# Transaction Log Miner

## Objective

Payment-operation failures are often buried in free-text complaints and incident logs. This project converts narratives into semantic representations and clusters recurring exception patterns, creating an analytical view of root causes, ISO 20022 remediation opportunities and customer impact.

## Architecture

```text
Transaction-Log-Miner/
├── assets/
│   └── charts/                         # Cluster and embedding diagnostics
├── data/
│   └── CFPB_complaints_2026_02_15.csv
├── notebooks/
│   └── Transaction_log_mining.ipynb
├── src/
│   ├── log_preprocessor.py             # Narrative preparation
│   ├── embeddings.py                   # Semantic representation
│   ├── clustering.py                   # K-Means and projection
│   └── generate_charts.py              # Cluster diagnostics
└── tests/
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.generate_charts
```

![Time-series anomaly spike and root-cause detection](assets/charts/01_cluster_distribution.png)

The incident-response view shows baseline log volume, HDBSCAN failure clusters and the automated alert trigger around a sudden issuer-timeout and HSM-tokenisation spike.

![Cluster stability and embedding projection diagnostics](assets/charts/02_embedding_projection.png)

The two-panel diagnostic combines cluster stability by operational severity with a labelled embedding projection, while muting noise points for clearer incident review.
