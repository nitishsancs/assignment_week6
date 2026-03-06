# Kursor — Week 7 Dashboard

## CVE Data Distribution Analysis & Model Metrics Impact

**[Live Demo on Streamlit Cloud](https://week6assignment-nitish.streamlit.app/)**

An interactive **Streamlit** dashboard that demonstrates how **changing data distribution** affects **machine learning model metrics** on real-world **NVD (National Vulnerability Database) CVE data**. The dashboard implements a full **Muller Loop** pipeline — training four classification algorithms and visualizing how upsampling/downsampling different features impacts model performance.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [How the Muller Loop Works](#how-the-muller-loop-works)
- [Distribution Manipulation](#distribution-manipulation)
- [Hosting on Streamlit Cloud](#hosting-on-streamlit-cloud)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project is part of **Week 6** of the Sexy Securities vulnerability analysis project. The core objective is to:

1. **Load real CVE data** from the NVD (National Vulnerability Database) JSON feeds
2. **Engineer features** from CVSS v3.1 metrics, weakness types, reference counts, and more
3. **Train 4 classification models** (XGBoost, MLP, Random Forest, SVM) to predict CVE severity
4. **Allow interactive distribution modification** — upsample/downsample features or target classes
5. **Re-run the Muller Loop** with the modified distribution and **compare metrics** against baseline
6. **Visualize outcomes** with confusion matrices, specificity vs sensitivity plots, radar charts, and feature importance

The dashboard answers the question: **"How do changes in data distribution affect model scoring?"**

---

## Features

### 🎛️ Interactive Controls
- **Feature selector** — Choose which feature to manipulate (severity class balance, attack vector, exploit status, CWE type, vendor count, exploitability score)
- **Distribution slider** — Continuously adjust from full downsample (-1.0) to full upsample (+1.0)
- **SMOTE toggle** — Use synthetic minority oversampling (SMOTE) instead of random oversampling
- **Algorithm selector** — Pick which of the 4 algorithms to include in each run
- **Max training samples** — Cap dataset size for faster iteration (default: 10,000)

### 📊 Visualization Tabs
| Tab | Description |
|-----|-------------|
| **Metrics** | Summary table with F1, Accuracy, Precision, Recall, ROC AUC. Delta indicators show change from baseline. Per-class expandable reports. |
| **Confusion Matrices** | Normalized heatmaps for all 4 algorithms (2×2 grid) + individual matrices. Toggle normalization on/off. |
| **Specificity vs Sensitivity** | Scatter plot showing each algorithm's per-class specificity (TNR) vs sensitivity (TPR). Includes data table. |
| **Distribution** | Side-by-side bar charts showing original vs modified class distribution. Distribution change summary table. Feature-level histograms. |
| **Feature Importance** | Horizontal bar charts for Gini (MDI), Permutation, and SHAP importance methods. Top-20 features table. |
| **Radar Chart** | Spider/radar chart comparing all algorithms across 5 metrics simultaneously. |

### 🤖 Algorithms (Muller Loop)
- **XGBoost** — Gradient boosted decision trees
- **MLP** — Multi-layer perceptron neural network (128→64 hidden layers)
- **Random Forest** — Ensemble of 100 decision trees
- **SVM** — Support vector machine with RBF kernel

### 📈 Metrics Tracked
- F1 Score (Macro & Weighted)
- Accuracy
- Precision (Macro & Per-class)
- Recall / Sensitivity (Macro & Per-class)
- Specificity / True Negative Rate (Per-class)
- ROC AUC (One-vs-Rest)
- Confusion Matrix
- Training Time

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ Controls │ │  Metrics │ │  Charts  │ │ Tables │  │
│  └────┬─────┘ └────▲─────┘ └────▲─────┘ └───▲────┘  │
│       │             │            │            │     │
│  ┌────▼─────────────┴────────────┴────────────┘     │
│  │              Session State Cache                 │
│  └────┬──────────────────────────────────────┘      │
├───────┼─────────────────────────────────────────────┤
│       ▼          Backend Pipeline                   │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐      │
│  │  Data    │→ │ Feature   │→ │ Distribution │      │
│  │  Loader  │  │ Engineer  │  │ Modifier     │      │
│  └──────────┘  └───────────┘  └──────┬───────┘      │
│                                       ▼             │
│                              ┌──────────────┐       │
│                              │ Muller Loop  │       │
│                              │ (4 Algos)    │       │
│                              └──────┬───────┘       │
│                                     ▼               │
│                            ┌────────────────┐       │
│                            │ Visualizations │       │
│                            └────────────────┘       │
└─────────────────────────────────────────────────────┘
```

---

## Data Source

### NVD CVE JSON Files

The dashboard uses **real CVE data** from the [National Vulnerability Database (NVD)](https://nvd.nist.gov/). Each CVE is stored as an individual JSON file following the NVD API 2.0 schema.

**Data repository:** [nvdcve-master](https://github.com/fkie-cad/nvd-json-data-feeds) — contains 270,000+ CVE records as individual JSON files.

### Extracted Features (per CVE)

| Feature | Type | Description |
|---------|------|-------------|
| `cvss31_score` | float | CVSS v3.1 base score (0-10) |
| `cvss2_score` | float | CVSS v2.0 base score (0-10) |
| `attack_vector` | categorical | NETWORK, ADJACENT_NETWORK, LOCAL, PHYSICAL |
| `attack_complexity` | categorical | LOW, HIGH |
| `privileges_required` | categorical | NONE, LOW, HIGH |
| `user_interaction` | categorical | NONE, REQUIRED |
| `scope` | categorical | UNCHANGED, CHANGED |
| `confidentiality` | categorical | NONE, LOW, HIGH |
| `integrity` | categorical | NONE, LOW, HIGH |
| `availability` | categorical | NONE, LOW, HIGH |
| `exploitability_score` | float | CVSS v3.1 exploitability sub-score |
| `impact_score` | float | CVSS v3.1 impact sub-score |
| `has_exploit` | binary | Whether a public exploit exists (from NVD references) |
| `num_vendors` | int | Number of affected vendors (from CPE configurations) |
| `num_weaknesses` | int | Number of CWE weakness entries |
| `primary_cwe` | categorical | Primary CWE ID (e.g., CWE-79, CWE-89) |
| `num_references` | int | Number of reference URLs |
| `desc_length` | int | Length of English description text |
| `year_published` | int | Year the CVE was published |
| `vuln_status` | categorical | NVD vulnerability status |

### Target Variable

**Severity** (4 classes) — derived from CVSS v3.1 base score:
- **Critical**: score ≥ 9.0
- **High**: score ≥ 7.0
- **Medium**: score ≥ 4.0
- **Low**: score < 4.0

---

## Installation

### Prerequisites

- **Python 3.10+**
- **pip** package manager
- **NVD CVE data** — Download the [nvdcve-master](https://github.com/fkie-cad/nvd-json-data-feeds) repository

### Step 1: Clone this repository

```bash
git clone https://github.com/<your-username>/Sexy-Securities.git
cd Sexy-Securities/week6_dashboard
```

### Step 2: Download NVD CVE data

```bash
# Clone the NVD data feeds repository
git clone https://github.com/fkie-cad/nvd-json-data-feeds.git nvdcve-master
```

Or download and extract the ZIP from the repository.

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
streamlit>=1.30.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
pandas>=2.1.0
numpy<2
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
```

### Step 4: Configure data path (if needed)

By default, the app looks for NVD data at `C:\Users\Nitish\Downloads\nvdcve-master\nvdcve`. To override:

```bash
# Linux/macOS
export NVD_DATA_DIR="/path/to/nvdcve-master/nvdcve"

# Windows PowerShell
$env:NVD_DATA_DIR = "C:\path\to\nvdcve-master\nvdcve"
```

### Step 5: Run the dashboard

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**.

**First run note:** The initial load parses all CVE JSON files (can take 1-2 minutes for 100k+ files) and caches the result to `cve_dataset.csv`. Subsequent runs load instantly from cache.

---

## Usage

### Basic Workflow

1. **Launch the dashboard** — `streamlit run app.py`
2. **View baseline metrics** — The dashboard immediately trains all 4 models on the original data distribution and displays results
3. **Select a feature** — Use the sidebar dropdown to choose which feature's distribution to manipulate
4. **Adjust the slider** — Move left to downsample, right to upsample
5. **Click "Re-run Muller Loop"** — The pipeline re-trains all selected models on the modified distribution
6. **Compare results** — Metrics tab shows delta indicators; distribution tab shows before/after charts
7. **Explore tabs** — Navigate through confusion matrices, specificity plots, feature importance, and radar charts

### Example Experiments

| Experiment | Feature | Slider | Expected Outcome |
|-----------|---------|--------|-----------------|
| Balance classes | Severity Class Balance | +0.8 | Minority classes (Low) get upsampled; F1 macro may improve |
| Downsample majority | Severity Class Balance | -0.8 | All classes shrink toward minority count; precision may change |
| Upsample exploits | Has Public Exploit | +0.5 | More exploit-bearing CVEs in training; model may better identify exploitable vulns |
| SMOTE oversampling | Severity Class Balance + SMOTE | +0.5 | Synthetic samples created for minority classes |
| Reduce network CVEs | Attack Vector | -0.5 | Fewer NETWORK-vector CVEs; changes attack vector feature distribution |

---

## Project Structure

```
week6_dashboard/
├── .gitignore              # Git ignore rules
├── .streamlit/
│   └── config.toml         # Streamlit theme and server config
├── app.py                  # Main Streamlit application
├── data_loader.py          # NVD JSON parser and CSV caching
├── feature_engineering.py  # Feature matrix preparation and encoding
├── feature_importance.py   # Gini, Permutation, SHAP importance
├── muller_loop.py          # Model training pipeline (4 algorithms)
├── distribution.py         # Up/downsampling and SMOTE logic
├── visualizations.py       # All matplotlib/seaborn plot functions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Module Documentation

### `data_loader.py`
Parses individual NVD CVE JSON files into a flat pandas DataFrame. Extracts CVSS v3.1 metrics, weaknesses, CPE vendor counts, reference metadata, and description length. Caches results to CSV for fast reloads.

**Key functions:**
- `load_data()` — Main entry point; loads from cache or parses JSON files
- `parse_single_cve(filepath)` — Parses one CVE JSON into a feature dict
- `load_from_nvd_files()` — Bulk loader with year filtering and progress callbacks

### `feature_engineering.py`
Transforms the raw DataFrame into an ML-ready feature matrix. One-hot encodes categorical CVSS components and CWE types, scales numeric features with StandardScaler.

**Key functions:**
- `prepare_features(df)` — Returns (X, y, feature_names) tuple
- `get_manipulable_features()` — Returns list of features available for distribution manipulation

### `muller_loop.py`
Trains all 4 classification algorithms and computes comprehensive metrics. Uses LabelEncoder internally to handle string labels (required for XGBoost).

**Key functions:**
- `run_muller_loop(X, y)` — Trains all models, returns dict of ModelResult dataclasses
- `results_to_summary_df(results)` — Converts results to a display-friendly DataFrame

### `distribution.py`
Handles all distribution modification logic — binary feature up/downsampling, continuous feature rebalancing, categorical redistribution, target class rebalancing, and SMOTE/RandomUnderSampler integration.

**Key functions:**
- `modify_distribution_raw(df, feature, type, slider)` — Modifies raw DataFrame distribution
- `modify_distribution_smote(X, y, slider)` — Applies SMOTE or undersampling on encoded features
- `get_distribution_stats(y_orig, y_mod)` — Computes before/after comparison table

### `visualizations.py`
All plotting functions returning matplotlib Figure objects (compatible with `st.pyplot(fig)`).

**Key functions:**
- `plot_all_confusion_matrices()` — 2×2 grid of confusion matrices
- `plot_specificity_vs_sensitivity()` — Scatter plot of spec vs sens per class
- `plot_metrics_comparison()` — Side-by-side bar chart (baseline vs current)
- `plot_distribution_comparison()` — Before/after class distribution bars
- `plot_feature_importance()` — Horizontal bar chart of top features
- `plot_radar_chart()` — Spider chart comparing algorithms

### `feature_importance.py`
Computes feature importance using three methods: Gini (MDI) from Random Forest, Permutation importance, and SHAP values.

---

## How the Muller Loop Works

The **Muller Loop** is the core model training pipeline:

```
Input: Feature matrix X, Target labels y
  │
  ├── Encode labels (LabelEncoder for XGBoost compatibility)
  ├── Stratified train/test split (80/20)
  │
  ├── For each algorithm in [XGBoost, MLP, RF, SVM]:
  │     ├── Train model on X_train, y_train
  │     ├── Predict on X_test
  │     ├── Compute: F1, Accuracy, Precision, Recall, ROC AUC
  │     ├── Compute: Confusion Matrix, Classification Report
  │     ├── Compute: Per-class Specificity (TNR)
  │     └── Store ModelResult dataclass
  │
  └── Return: Dict[algorithm_name → ModelResult]
```

The loop runs **twice** — once for the baseline (original distribution) and once after each distribution modification — enabling direct comparison.

---

## Distribution Manipulation

### Methods

| Method | Slider Direction | Technique |
|--------|-----------------|-----------|
| **Random Upsample** | Positive (+) | Sample with replacement from minority group |
| **Random Downsample** | Negative (-) | Sample without replacement from majority group |
| **SMOTE** | Positive (+) | Generate synthetic minority samples using k-nearest neighbors |
| **RandomUnderSampler** | Negative (-) | Randomly remove majority samples using imbalanced-learn |

### Feature Types

- **Target (severity)** — Directly modifies class balance
- **Binary (has_exploit)** — Up/downsamples the positive class
- **Categorical (attack_vector, primary_cwe)** — Rebalances majority vs minority categories
- **Continuous (num_vendors, exploitability_score)** — Up/downsamples records above/below median

---

## Hosting on Streamlit Cloud

### Option 1: Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository → `week6_dashboard/app.py`
5. **Important:** You'll need to upload `cve_dataset.csv` or pre-generate it, since Streamlit Cloud won't have the raw NVD JSON files

### Pre-generating the cache for deployment

```bash
# Run the data loader locally to generate cve_dataset.csv
cd week6_dashboard
python data_loader.py

# Then include cve_dataset.csv in your deployment
# (remove it from .gitignore if deploying with the CSV)
```

### Option 2: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVD_DATA_DIR` | `C:\Users\Nitish\Downloads\nvdcve-master\nvdcve` | Path to NVD CVE JSON files |

### Streamlit Config (`.streamlit/config.toml`)

The theme uses a dark navy + red accent color scheme matching the Sexy Securities brand.

### Training Parameters

Defaults can be modified in each module:
- **Max training samples**: 10,000 (configurable in sidebar)
- **Test split**: 20%
- **Random state**: 42 (reproducible)
- **Year filter**: CVEs from 2017 onward

---

## Troubleshooting

### "No CVE JSON files found"
→ Set the `NVD_DATA_DIR` environment variable to point to the directory containing `CVE-*.json` files.

### SVM training is slow
→ Reduce "Max Training Samples" in the sidebar (default: 10,000). SVM has O(n²) complexity.

### "numpy.core.multiarray failed to import"
→ NumPy version conflict. Run: `pip install "numpy<2" --force-reinstall`

### XGBoost not found
→ `pip install xgboost`. Falls back to sklearn's GradientBoostingClassifier if not available.

### High model accuracy (>99%)
→ Expected. CVSS vector components (attack_vector, attack_complexity, etc.) are inputs to the CVSS score formula, which directly determines severity. The models learn this mapping easily. Distribution changes still produce observable metric shifts, especially for minority classes.

### First load is slow
→ The initial JSON parsing of 100k+ files takes 1-2 minutes. The result is cached to `cve_dataset.csv` — subsequent runs load instantly.

---

## License

MIT License — Part of the Sexy Securities project.

---

*Built with Streamlit, scikit-learn, XGBoost, and real NVD CVE data.*
