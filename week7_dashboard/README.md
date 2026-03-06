# 🎯 Sexy Securities — Week 7 Dashboard

## Finding Optimal Data Distributions via AutoML-Inspired Optimization

An interactive **Streamlit** dashboard that extends Week 6's manual distribution exploration into an **automated optimization pipeline**. The Muller-AutoML loop searches through distribution ranges to find the **"Goldilocks" distribution** — the optimal data balance that maximizes model performance while avoiding overfitting.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Assignment Checklist](#assignment-checklist)
- [How the Muller-AutoML Loop Works](#how-the-muller-automl-loop-works)
- [Overfitting vs Underfitting Analysis](#overfitting-vs-underfitting-analysis)
- [Data Narrative Structure](#data-narrative-structure)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project addresses **Week 7** of the Sexy Securities vulnerability analysis course. Building on Week 6's interactive distribution manipulation, we now:

1. **Manually scope** the search range using interactive sliders to observe where F1 plateaus or declines
2. **Automate the search** with the Muller-AutoML loop — iterating through distribution increments with explicit exit conditions
3. **Find the optimal distribution** — the precise slider value that maximizes model F1 for each feature-algorithm pair
4. **Validate robustness** — learning curves, cross-validation, and train/val gap analysis to detect overfitting vs underfitting
5. **Generate a data narrative** — explaining *why* each feature was chosen, *how* the distribution changed, and *what* the model learned

---

## Key Features

### 🎚️ Tab 1: Manual Scoping
- Interactive slider to probe individual distribution values
- Immediate train/val comparison at each probe point
- Side-by-side baseline vs probed metrics
- Guides setting the min/max search boundaries

### 🤖 Tab 2: AutoML Optimization
- **Automated sweep** from search_min to search_max in configurable increments
- **Exit conditions:**
  - Derivative of F1 turns negative → peak detected
  - Improvement < Δ threshold → plateau detected
  - All steps exhausted → complete sweep
- **Trajectory visualization** showing F1 vs slider value for all algorithms
- **Delta plot** showing step-to-step improvement
- Full metrics at the optimal point with learning curves

### ⚖️ Tab 3: Overfitting Analysis
- **Train vs Validation F1** bar chart with gap annotations
- **Fit diagnosis** per algorithm: Overfit / Underfit / Optimal / Acceptable
- **Cross-validation distributions** (box plots)
- **Learning curves** showing how train/val scores converge with data size
- **Robustness checklist** per assignment requirements

### 📖 Tab 4: Data Narrative
- Feature importance rationale (why this feature was chosen)
- Before/after distribution charts
- Before/after confusion matrix comparison
- Per-algorithm overfitting diagnosis with explanations
- Quantified improvement conclusions

### 📊 Tab 5–7: Baseline Metrics, Confusion Matrices, Feature Importance
- Full baseline model evaluation with cross-validation
- All Week 6 visualization capabilities carried forward

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                          │
│  ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Manual     │ │  AutoML    │ │ Overfit  │ │  Narrative │  │
│  │  Scoping    │ │  Optimizer │ │ Analysis │ │  Writeup   │  │
│  └─────┬──────┘ └─────┬──────┘ └────┬─────┘ └─────┬──────┘  │
│        │               │             │              │         │
│  ┌─────▼───────────────▼─────────────▼──────────────▼──────┐ │
│  │                  Session State Cache                      │ │
│  └─────────────────────────┬────────────────────────────────┘ │
├────────────────────────────┼──────────────────────────────────┤
│            Backend         │                                   │
│  ┌──────────┐  ┌───────────▼───────┐  ┌──────────────────┐   │
│  │  Data    │→ │  AutoML Optimizer  │→ │  Muller Loop     │   │
│  │  Loader  │  │  (search + exit)   │  │  + CV + LC       │   │
│  └──────────┘  └───────────────────┘  └────────┬─────────┘   │
│                                                 │             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────▼──────────┐ │
│  │ Feature      │  │ Distribution │  │ Fit Diagnosis      │ │
│  │ Importance   │  │ Modifier     │  │ (overfit/underfit) │ │
│  └──────────────┘  └──────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- **Python 3.10+**
- **NVD CVE data** — either the cached `cve_dataset.csv` or raw JSON files from [nvdcve-master](https://github.com/fkie-cad/nvd-json-data-feeds)

### Quick Start

```bash
git clone https://github.com/<your-username>/assignment_week6.git
cd assignment_week6/week7_dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at **http://localhost:8502**.

### Data Setup

The dashboard ships with a pre-cached `cve_dataset.csv` (108k+ CVEs). To regenerate from raw JSON files:

```bash
# Set the NVD data directory
export NVD_DATA_DIR="/path/to/nvdcve-master/nvdcve"   # Linux/macOS
$env:NVD_DATA_DIR = "C:\path\to\nvdcve-master\nvdcve"  # Windows

# Regenerate cache
python data_loader.py
```

---

## Usage

### Workflow

1. **Launch** → `streamlit run app.py`
2. **Tab 1 (Manual Scoping)** → Use the sidebar slider to probe individual values. Observe where F1 peaks or drops. Set the search range accordingly.
3. **Tab 2 (AutoML)** → Click "Run AutoML Optimization". The loop sweeps through the range and finds the optimal slider value.
4. **Tab 3 (Overfitting)** → Review learning curves, train/val gap, CV distributions, and fit diagnoses.
5. **Tab 4 (Narrative)** → Read the auto-generated data narrative with before/after charts and conclusions.

### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Feature to Optimize** | Which feature's distribution to search |
| **Search Range** | Min/max boundaries for the AutoML sweep |
| **Manual Probe** | Single slider value for manual testing |
| **Step Size** | Increment between AutoML steps (0.02–0.20) |
| **Improvement Threshold** | Minimum Δ to continue searching |
| **Max Training Samples** | Cap for SVM tractability |
| **SMOTE** | Use synthetic oversampling for target class |
| **Algorithms** | Which models to include |

---

## Assignment Checklist

| Requirement | Status | Location |
|------------|--------|----------|
| ✅ Use slider to find range for min/max | Tab 1: Manual Scoping | Sidebar slider + search range |
| ✅ Muller loop with exit condition in AutoML fashion | Tab 2: AutoML Optimization | `automl_optimizer.py` |
| ✅ Find optimal value for dataset number | Tab 2: Optimal slider value | `run_optimization()` |
| ✅ Writeup with feature importance | Tab 4: Data Narrative §1 | SHAP/Gini/Permutation analysis |
| ✅ Up- and downsampling discussion | Tab 4: Data Narrative §2 | Before/after distribution charts |
| ✅ Data distributions and charts | Tab 4: Data Narrative §2 | Histograms + confusion matrices |
| ✅ Data narrative | Tab 4: Full narrative | Auto-generated writeup |
| ✅ Check overfitting conditions | Tab 3: Overfitting Analysis | Train/val gap, learning curves |
| ✅ Check underfitting conditions | Tab 3: Fit Diagnosis | Low train + low val detection |
| ✅ Learning curves plotted | Tab 3: Learning Curves | `plot_learning_curves()` |
| ✅ Train-test gap widening check | Tab 3: Train vs Val | Gap annotations per algorithm |
| ✅ Cross-validation for generalizability | Tab 3: CV Distributions | 5-fold stratified CV |

---

## How the Muller-AutoML Loop Works

```
Define: search_min, search_max, step_size, threshold Δ

1. Compute baseline scores at slider=0
2. For slider_value in [search_min → search_max] by step_size:
   a. Modify distribution of selected feature
   b. Prepare features from modified data
   c. Run Muller Loop (quick mode: F1 only)
   d. Record scores for all algorithms
   e. Compute delta = current_avg_F1 - previous_avg_F1
   f. EXIT if:
      - delta < -Δ AND consecutive_declines >= 2  → PEAK DETECTED
      - |delta| < Δ AND consecutive_declines >= 3  → PLATEAU
3. Return optimal_slider_value with highest average F1
4. Run full Muller Loop at optimal point (with CV + learning curves)
```

### Exit Conditions

| Condition | Trigger | Meaning |
|-----------|---------|---------|
| **Peak Detected** | F1 derivative negative for 2+ steps after a peak | We've passed the maximum |
| **Plateau** | Improvement < Δ for 3+ steps | Diminishing returns |
| **Complete** | All steps evaluated | Full sweep completed |

---

## Overfitting vs Underfitting Analysis

### Detection Criteria

| Condition | Train F1 | Val F1 | Gap | Diagnosis |
|-----------|----------|--------|-----|-----------|
| **Overfitting** | > 0.95 | < 0.90 | > 0.05 | Model memorizes noise from over-upsampling |
| **Underfitting** | < 0.70 | < 0.70 | any | Model lacks signal from aggressive downsampling |
| **Optimal Fit** | any | > 0.70 | < 0.03 | Converged train/val = good bias-variance tradeoff |
| **Slight Overfit** | any | any | 0.03–0.05 | Mild memorization, may need regularization |

### Validation Methods

1. **Train/Val Split** — 80/20 stratified split; compare F1 on each
2. **Cross-Validation** — 5-fold stratified CV within the training set
3. **Learning Curves** — Plot F1 vs training set size; gap widening = overfitting

---

## Data Narrative Structure

The auto-generated narrative (Tab 4) follows this structure:

### §1 Feature Importance & Resampling
- SHAP / Gini / Permutation importance rankings
- Why the selected feature matters for model performance
- Feature's distribution characteristics (skew, imbalance)

### §2 Distribution & Chart Analysis
- Before/after histograms of the optimized feature
- Before/after class distribution bar charts
- Before/after confusion matrices showing FP/FN changes
- Quantified sample count changes per class

### §3 Robustness Check
- Per-algorithm fit diagnosis (overfit / underfit / optimal)
- Learning curve interpretation
- Cross-validation score stability

### §4 Conclusions
- Optimal slider value and algorithm
- Average F1 improvement across models
- Exit reason and search efficiency
- Generalizability confirmation via CV

---

## Project Structure

```
week7_dashboard/
├── .gitignore
├── .streamlit/
│   └── config.toml
├── app.py                  # Main Streamlit application (7 tabs)
├── automl_optimizer.py     # AutoML optimization loop + fit diagnosis
├── data_loader.py          # NVD JSON parser + CSV caching
├── distribution.py         # Up/downsampling + SMOTE logic
├── feature_engineering.py  # Feature matrix preparation + encoding
├── feature_importance.py   # Gini, Permutation, SHAP importance
├── muller_loop.py          # Enhanced Muller loop (CV + learning curves)
├── visualizations.py       # All plotting functions (Week 6 + Week 7)
├── cve_dataset.csv         # Pre-cached CVE data (108k+ records)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Module Documentation

### `automl_optimizer.py` (NEW)
The core Week 7 module. Implements the automated distribution search loop.

**Key functions:**
- `run_optimization()` — Sweep a single feature's distribution range with exit conditions
- `run_full_optimization()` — Sweep multiple features sequentially
- `optimization_result_to_df()` — Convert optimization steps to a plottable DataFrame
- `get_fit_diagnosis()` — Classify a model result as overfit/underfit/optimal

### `muller_loop.py` (ENHANCED)
Extended from Week 6 with cross-validation, train/val scoring, and learning curve support.

**New fields in `ModelResult`:**
- `train_f1` / `val_f1` — For overfitting detection
- `cv_scores` / `cv_mean` / `cv_std` — Cross-validation results
- `learning_curve_*` — Data for learning curve plots

**New function:**
- `run_muller_loop_quick()` — Lightweight version returning only F1 scores (used by optimizer)

### `visualizations.py` (ENHANCED)
All Week 6 plots plus new Week 7 visualizations.

**New functions:**
- `plot_optimization_trajectory()` — F1 vs slider value across algorithms
- `plot_optimization_delta()` — Step-to-step improvement bar chart
- `plot_learning_curves()` — Train vs val score vs data size
- `plot_train_vs_val()` — Side-by-side train/val F1 bars with gap annotation
- `plot_cv_scores()` — Box plots of CV score distributions
- `plot_fit_diagnosis_summary()` — Visual diagnosis cards per algorithm
- `plot_multi_feature_optimization_summary()` — Cross-feature comparison

### Unchanged from Week 6
- `data_loader.py` — NVD JSON parser
- `feature_engineering.py` — Feature encoding + scaling
- `distribution.py` — Raw distribution modification + SMOTE
- `feature_importance.py` — Gini / Permutation / SHAP

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NVD_DATA_DIR` | `C:\Users\Nitish\Downloads\nvdcve-master\nvdcve` | Path to NVD CVE JSON files |

### Optimization Defaults

| Parameter | Default | Range |
|-----------|---------|-------|
| Search Range | [-0.8, 0.8] | [-1.0, 1.0] |
| Step Size | 0.05 | 0.02–0.20 |
| Improvement Threshold | 0.001 | 0.0001–0.01 |
| Max Training Samples | 10,000 | 1,000–108,822 |
| CV Folds | 5 | 2–10 |

---

## Troubleshooting

### AutoML optimization is slow
→ Increase step size (0.10 or 0.20) or narrow the search range. Reduce max training samples.

### SVM hangs during optimization
→ Deselect SVM from the algorithm list or reduce max samples to 5,000.

### Learning curves not showing
→ Learning curves are only computed at the optimal point after AutoML runs. Click "Run AutoML Optimization" first.

### High model accuracy (>99%)
→ Expected. CVSS vector components directly determine severity. The optimization still finds distribution shifts that improve minority class performance.

### "No CVE JSON files found"
→ The app uses `cve_dataset.csv` by default. Only needs raw JSON files if cache is deleted.

---

## License

MIT License — Part of the Sexy Securities project.

---

*Built with Streamlit, scikit-learn, XGBoost, and real NVD CVE data.*
