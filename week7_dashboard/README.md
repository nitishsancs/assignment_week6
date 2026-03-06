# Sexy Securities — Week 7 Dashboard

## Homework 7: Automated Optimization of Data Distributions

An interactive **Streamlit** dashboard that transitions from manual experimentation to an **AutoML-inspired optimization** of the CVE dataset. The Muller-AutoML loop defines boundaries of data shifts and automatically pinpoints the **"Goldilocks" distribution** for each model architecture — the optimal data balance that maximizes F1 while avoiding overfitting.

**Data Source:** 108,822 real NVD CVE records with CVSS v3.1 metrics
**Models:** XGBoost, MLP, Random Forest, SVM
**Target:** Severity classification (Critical / High / Medium / Low)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Assignment Checklist](#assignment-checklist)
- [1. Manual Scoping via Interactive Sliders](#1-manual-scoping-via-interactive-sliders)
- [2. The Muller-AutoML Optimization Loop](#2-the-muller-automl-optimization-loop)
- [3. Data Narrative & Technical Writeup](#3-data-narrative--technical-writeup)
- [4. Robustness Check: Overfitting vs Underfitting](#4-robustness-check-overfitting-vs-underfitting)
- [Installation & Usage](#installation--usage)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Assignment Checklist

Every requirement from the homework specification is implemented and mapped to a specific location in the code:

### Part 1: Manual Scoping via Interactive Sliders

| Requirement | Status | Implementation |
|---|---|---|
| Slide through upsampling/downsampling extremes for top features | ✅ | Tab 1 "Manual Scoping" — sidebar slider from -1.0 to +1.0 |
| Top features identified via SHAP or Gini Importance | ✅ | `feature_importance.py` — computes Gini, Permutation, and SHAP rankings |
| Note where F1 plateaus or sharply declines | ✅ | Tab 1 shows baseline-vs-probe comparison chart after each manual run |
| Define a Min and Max boundary for the search | ✅ | Sidebar dual-slider "Search Range (min, max)" sets boundaries |

### Part 2: The Muller-AutoML Optimization Loop

| Requirement | Status | Implementation |
|---|---|---|
| Iterate through range in fixed increments (e.g. 5% shifts) | ✅ | `automl_optimizer.py:run_optimization()` — configurable step_size (default 0.05) |
| Train & Evaluate at each increment (Muller Loop) | ✅ | Calls `run_muller_loop_quick()` at each step for all selected algorithms |
| Exit when derivative turns negative (peak) | ✅ | `automl_optimizer.py` line 174: `delta < -threshold and consecutive_declines >= 2` |
| Exit when improvement < threshold (e.g. Δ < 0.001) | ✅ | `automl_optimizer.py` line 179: `abs(delta) < threshold and consecutive_declines >= 3` |
| Return "Optimal Distribution Value" per feature-model pair | ✅ | `OptimizationResult.optimal_value` + `optimal_scores` per algorithm |

### Part 3: Data Narrative & Technical Writeup

| Requirement | Status | Implementation |
|---|---|---|
| Explain "Why" behind feature choice (SHAP rationale) | ✅ | Tab 4 §1 — shows top-5 features by selected importance method with narrative |
| Before/After histograms (raw vs optimized) | ✅ | Tab 4 §2 — `plot_feature_distribution_histogram()` shows raw vs optimized feature |
| Before/After class distribution bar charts | ✅ | Tab 4 §2 — `plot_distribution_comparison()` for class balance |
| Before/After Confusion Matrices (FP/FN reduction) | ✅ | Tab 4 §2 — side-by-side baseline vs optimal confusion matrix for best algorithm |
| SMOTE / upsampling / downsampling discussion | ✅ | Tab 4 §2 — embedded narrative explaining Random Oversampling vs SMOTE vs Downsampling |

### Part 4: Robustness Check — Overfitting vs Underfitting

| Requirement | Status | Implementation |
|---|---|---|
| Overfitting: High Training / Low Validation detection | ✅ | `get_fit_diagnosis()` — `train_f1 > 0.95 and gap > 0.05` → "overfit" |
| Underfitting: Low Training / Low Validation detection | ✅ | `get_fit_diagnosis()` — `train_f1 < 0.7 and val_f1 < 0.7` → "underfit" |
| Optimal Fit: Converged Training and Validation | ✅ | `get_fit_diagnosis()` — `gap < 0.03 and val_f1 > 0.7` → "optimal" |
| Learning Curves for best-performing algorithm | ✅ | Tab 3 — `plot_learning_curves()` for each algorithm with train/val ± std bands |
| Gap widening check as you upsample | ✅ | Tab 3 — "Gap Widening Analysis" button runs `run_gap_analysis()` across 5 slider values, plots train-val gap widening with `plot_gap_widening()` |
| Cross-validation within Muller Loop | ✅ | `muller_loop.py` — `StratifiedKFold` + `cross_val_score` with `f1_macro` scoring |

### Narrative Checklist (from assignment)

| Checklist Item | Status | Implementation |
|---|---|---|
| [x] Learning Curves plotted for best-performing algorithm | ✅ | Tab 3 "Learning Curves" section — all algos with LC data |
| [x] Gap between training/test metrics widening as you upsample? | ✅ | Tab 3 "Gap Widening Analysis" — `run_gap_analysis()` across slider values + per-algo interpretation |
| [x] Cross-validation within Muller Loop for generalizability | ✅ | 5-fold stratified CV in `run_muller_loop()`, results shown in Tab 3 "CV Score Distributions" |

---

## 1. Manual Scoping via Interactive Sliders

**Tab 1: Manual Scoping** in the dashboard.

Before automating, the user manually probes the feature space:

1. **Select a feature** from the sidebar (Severity Class Balance, Has Public Exploit, Attack Vector, Primary CWE, Number of Vendors, or Exploitability Score).
2. **Slide the manual probe** from -1.0 (full downsample) to +1.0 (full upsample).
3. **Click "Run Manual Probe"** — trains all 4 algorithms with cross-validation at that slider value.
4. **Observe** where F1 plateaus or sharply declines in the baseline-vs-probe comparison.
5. **Set the search range** using the dual-slider in the sidebar based on observations.

The top features for resampling are identified via three importance methods:
- **Gini (MDI) Importance** — from RandomForest feature importances
- **Permutation Importance** — model-agnostic, measures F1 drop when each feature is shuffled
- **SHAP Values** — TreeExplainer on RandomForest, mean absolute SHAP per feature

---

## 2. The Muller-AutoML Optimization Loop

**Tab 2: AutoML Optimization** in the dashboard.

Once the range is defined, the automated search loop executes:

```
ALGORITHM: Muller-AutoML Optimization Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INPUT:  raw_df, feature_name, feature_type, search_min, search_max,
        step_size, threshold Δ, selected_algos, max_samples

1. Compute BASELINE scores at slider=0 using run_muller_loop_quick()
2. Initialize: best_avg_score = -1, consecutive_declines = 0
3. FOR slider_value IN [search_min → search_max] BY step_size:
   a. Modify distribution of selected feature
   b. Prepare features from modified DataFrame
   c. Stratified subsample to max_samples (cap for SVM tractability)
   d. Run Muller Loop (quick mode): train all algorithms, get F1 scores
   e. Compute avg_score across all algorithms
   f. Compute delta = avg_score - previous_avg_score
   g. Track global best slider value if avg_score > best_avg_score
   h. EXIT CONDITIONS:
      ┌──────────────────────────────────────────────────────────┐
      │ IF delta < -Δ AND consecutive_declines >= 2:            │
      │   → EXIT "peak_detected" (derivative turned negative)   │
      │                                                          │
      │ IF |delta| < Δ AND consecutive_declines >= 3:           │
      │   → EXIT "plateau" (improvement below threshold)        │
      └──────────────────────────────────────────────────────────┘
4. Run FULL Muller Loop at optimal point (with CV + learning curves)
5. Return OptimizationResult with trajectory, optimal value, scores

OUTPUT: optimal_slider_value, per-algorithm F1 scores,
        improvement over baseline, exit reason
```

### Exit Conditions

| Condition | Trigger | Meaning |
|---|---|---|
| **Peak Detected** | F1 derivative negative for 2+ consecutive steps | We have passed the maximum; further shifts degrade performance |
| **Plateau** | Improvement < Δ for 3+ consecutive steps | Diminishing returns; the optimal region has been found |
| **Complete** | All slider values evaluated | Full sweep completed without early exit |

### Visualization Outputs
- **Optimization Trajectory** — F1 vs slider value for each algorithm, with optimal point marked
- **Performance Derivative (Δ F1)** — step-to-step improvement bar chart (green=positive, red=negative)
- **Baseline vs Optimal Comparison** — side-by-side F1 bar chart
- **Distribution Comparison** — before/after class distribution

---

## 3. Data Narrative & Technical Writeup

**Tab 4: Data Narrative** in the dashboard.

The auto-generated narrative weaves a "Data Story" structured around four pillars:

### §1 Feature Importance & Resampling Rationale

> *"We chose `severity` (target class balance) for resampling because SHAP values indicated
> that class imbalance was a primary driver of misclassification: the model under-predicted
> Critical and Low severity CVEs, which had fewer training samples."*

The dashboard:
- Displays top-5 features by the selected importance method (Gini / Permutation / SHAP)
- Plots a horizontal bar chart of feature importances
- Checks whether the selected feature is in the top-5 and explains its relevance

### §2 Distribution & Chart Analysis — Before and After

The dashboard generates paired visualizations:

| Visualization | What It Shows |
|---|---|
| **Class Distribution Bar Charts** | Original vs Optimized class counts for Low/Medium/High/Critical |
| **Feature Histograms** | Raw vs Optimized distribution of the selected feature (e.g. severity, attack_vector) |
| **Distribution Statistics Table** | Per-class counts, differences, and % change |
| **Confusion Matrix Comparison** | Side-by-side baseline vs optimal CM for the best algorithm |

The narrative also includes a discussion of resampling approaches:
- **Random Oversampling** — duplicates minority-class records (fast but amplifies noise)
- **SMOTE** — generates synthetic minority samples via interpolation (smoother boundary, risk of unrealistic samples)
- **Downsampling** — removes majority-class records (authentic data, but reduces training signal)

### §3 Robustness: Overfitting vs Underfitting

Per-algorithm diagnosis with natural-language explanations. Example:

> *"XGBoost — Slight Overfit: Training F1 (1.0000) is somewhat higher than validation F1
> (0.9950). Gap of 0.0050 indicates mild overfitting. Consider reducing model complexity."*

### §4 Conclusions

Quantified results:
- Optimal slider value and best algorithm
- Average F1 improvement across all algorithms
- Exit reason and search efficiency
- Cross-validation confirmation of generalizability

---

## 4. Robustness Check: Overfitting vs Underfitting

**Tab 3: Overfitting Analysis** in the dashboard.

### Detection Criteria (from `get_fit_diagnosis()`)

| Condition | Indicator | Data Narrative Context |
|---|---|---|
| **Overfitting** | Train F1 > 0.95 and Gap > 0.05 | Occurs if you over-upsample (SMOTE) a feature, creating synthetic points too similar to existing noise |
| **Underfitting** | Train F1 < 0.70 and Val F1 < 0.70 | Occurs if you downsample too aggressively, stripping the model of the "signal" it needs to learn |
| **Optimal Fit** | Gap < 0.03 and Val F1 > 0.70 | The point where the AutoML loop identifies the best bias-variance tradeoff |
| **Slight Overfit** | Gap > 0.03 | Mild memorization — consider reducing complexity or increasing regularization |

### Validation Methods

1. **Train/Val Split** — 80/20 stratified split; Train F1 vs Val F1 with gap annotation
2. **Cross-Validation** — 5-fold stratified CV within the training set; box plot distributions
3. **Learning Curves** — sklearn `learning_curve()` with 8 data size increments; train ± std vs val ± std bands, gap annotation at final point
4. **Gap Widening Analysis** — `run_gap_analysis()` trains all models at 5 slider values (-0.6, -0.3, 0.0, +0.3, +0.6) and plots train-val gap widening over the upsample range; per-algorithm interpretation of whether gap is stable or widening

### Tab 3 Contents

| Section | What It Shows |
|---|---|
| **Train vs Validation F1** | Bar chart with gap (Δ) annotations per algorithm, color-coded by severity |
| **Fit Diagnosis Summary** | Horizontal bar charts per algorithm showing Train F1 / Val F1 / CV Mean, color-coded by status |
| **Diagnosis Cards** | Expandable per-algorithm cards with metrics and natural-language explanations |
| **Cross-Validation Distributions** | Box plots of CV fold scores per algorithm with mean markers |
| **Learning Curves** | 2-column grid of per-algorithm learning curves (train score ± std, val score ± std) |
| **Gap Widening Analysis** | Two-panel plot: (top) train/val F1 lines across slider values, (bottom) gap magnitude with overfit threshold |
| **Overfitting Checklist** | Dynamic checklist that checks off: Learning Curves ✅, Gap Analysis ✅, Gap widening ✅/⚠️, Cross-validation ✅ |

---

## Installation & Usage

### Prerequisites

- **Python 3.10+**
- **NVD CVE data** — ships as pre-cached `cve_dataset.csv` (108k+ records)

### Quick Start

```bash
git clone https://github.com/nitishsancs/assignment_week6.git
cd assignment_week6/week7_dashboard
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8502**.

### Recommended Workflow

1. **Launch** → `streamlit run app.py`
2. **Tab 5 (Baseline Metrics)** → Review baseline model performance (auto-computed on first load)
3. **Tab 7 (Feature Importance)** → Identify top features via Gini/SHAP/Permutation
4. **Tab 1 (Manual Scoping)** → Probe 3-5 slider values to see where F1 peaks. Set search range.
5. **Tab 2 (AutoML)** → Click "Run AutoML Optimization" to find optimal distribution value
6. **Tab 3 (Overfitting)** → Click "Run Gap Widening Analysis". Review learning curves, CV, fit diagnoses.
7. **Tab 4 (Narrative)** → Read the complete data narrative with before/after charts

### Data Regeneration

To regenerate from raw NVD JSON files:

```bash
# Set path (Windows)
$env:NVD_DATA_DIR = "C:\path\to\nvdcve-master\nvdcve"

# Set path (Linux/macOS)
export NVD_DATA_DIR="/path/to/nvdcve-master/nvdcve"

# Delete cache and regenerate
python data_loader.py
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (app.py)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ Manual   │ │ AutoML   │ │ Overfit  │ │ Data     │ │Baseline│  │
│  │ Scoping  │ │ Optimizer│ │ Analysis │ │ Narrative│ │Metrics │  │
│  │ Tab 1    │ │ Tab 2    │ │ Tab 3    │ │ Tab 4    │ │Tab 5-7 │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘  │
│       └──────┬──────┴─────┬──────┴──────┬─────┘           │       │
│              │            │             │                  │       │
│  ┌───────────▼────────────▼─────────────▼──────────────────▼────┐  │
│  │               Streamlit Session State Cache                   │  │
│  └──────────────────────────┬────────────────────────────────────┘  │
├─────────────────────────────┼──────────────────────────────────────┤
│           Backend Modules   │                                       │
│                             │                                       │
│  ┌──────────────────────────▼────────────────────────────────────┐ │
│  │  automl_optimizer.py                                           │ │
│  │  ├─ run_optimization()       — sweep with exit conditions      │ │
│  │  ├─ run_gap_analysis()       — train-val gap across sliders    │ │
│  │  ├─ get_fit_diagnosis()      — overfit/underfit classification │ │
│  │  └─ optimization_result_to_df()                                │ │
│  └───────────────────────────┬───────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐ │
│  │  muller_loop.py (Enhanced)                                     │ │
│  │  ├─ run_muller_loop()        — full training + CV + LC         │ │
│  │  ├─ run_muller_loop_quick()  — fast F1-only for optimizer      │ │
│  │  └─ ModelResult (train_f1, val_f1, cv_scores, learning_curve)  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────────┐│
│  │data_loader  │  │distribution  │  │feature_importance          ││
│  │ .py         │  │ .py          │  │ .py                        ││
│  │NVD JSON →CSV│  │SMOTE/raw     │  │Gini/Permutation/SHAP      ││
│  └─────────────┘  └──────────────┘  └────────────────────────────┘│
│  ┌─────────────────┐  ┌─────────────────────────────────────────┐ │
│  │feature_engineer │  │visualizations.py                         │ │
│  │ ing.py          │  │ 20+ plot functions (Week 6 + Week 7)    │ │
│  │OHE + Scaling    │  │ Trajectories, LCs, Gap, CM, Radar, etc  │ │
│  └─────────────────┘  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
week7_dashboard/
├── .gitignore                  # Excludes __pycache__, .env, secrets
├── .streamlit/
│   └── config.toml             # Theme + server config (port 8502)
├── app.py                      # Main Streamlit app — 7 tabs, all controls
├── automl_optimizer.py         # AutoML loop + gap analysis + fit diagnosis
├── data_loader.py              # NVD JSON parser + CSV caching (from Week 6)
├── distribution.py             # Up/downsampling + SMOTE logic (from Week 6)
├── feature_engineering.py      # One-hot encoding + StandardScaler (from Week 6)
├── feature_importance.py       # Gini / Permutation / SHAP (from Week 6)
├── muller_loop.py              # Enhanced Muller loop: CV + train/val + LC
├── visualizations.py           # 20+ plot functions for all tabs
├── cve_dataset.csv             # Pre-cached 108k+ CVE records
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Module Documentation

### `automl_optimizer.py` — NEW for Week 7

The core optimization engine. Implements all Week 7 requirements.

| Function | Purpose |
|---|---|
| `run_optimization()` | Main AutoML loop — sweeps slider range with exit conditions (peak/plateau/threshold) |
| `run_full_optimization()` | Runs optimization for multiple features sequentially |
| `run_gap_analysis()` | Trains all models at multiple slider values, collects train-val F1 gaps for gap-widening analysis |
| `optimization_result_to_df()` | Converts optimization steps to a plottable DataFrame |
| `get_fit_diagnosis()` | Classifies a ModelResult as overfit / underfit / optimal / slight_overfit / acceptable with natural-language explanation |

**Data classes:**
- `OptimizationStep` — single step: slider_value, scores per algo, delta
- `OptimizationResult` — complete run: trajectory, optimal_value, exit_reason, improvements

### `muller_loop.py` — ENHANCED from Week 6

Extended with cross-validation, train/val scoring, and learning curve computation.

| Addition | Purpose |
|---|---|
| `train_f1` / `val_f1` fields | Compare training vs validation F1 for overfitting detection |
| `cv_scores` / `cv_mean` / `cv_std` fields | 5-fold stratified cross-validation scores |
| `learning_curve_*` fields | Training sizes, train scores, val scores for learning curve plots |
| `run_muller_loop_quick()` | Lightweight function returning only F1 dict (used by optimizer for speed) |
| `deepcopy(model)` per algorithm | Prevents state leakage between CV folds and repeated training |

### `visualizations.py` — ENHANCED from Week 6

All 13 Week 6 plots plus 8 new Week 7 plots:

| New Function | Purpose | Required By |
|---|---|---|
| `plot_optimization_trajectory()` | F1 vs slider value per algorithm, optimal point marked | Assignment §2 |
| `plot_optimization_delta()` | Step-to-step Δ F1 bar chart (green/red) | Assignment §2 |
| `plot_learning_curves()` | Train/val score ± std vs training size, gap annotation | Assignment §4 checklist |
| `plot_train_vs_val()` | Side-by-side train/val F1 bars with Δ annotation | Assignment §4 |
| `plot_cv_scores()` | Box plots of CV fold distributions per algorithm | Assignment §4 checklist |
| `plot_fit_diagnosis_summary()` | Per-algo horizontal bar gauges (Train/Val/CV) color-coded by status | Assignment §4 |
| `plot_gap_widening()` | Two-panel: train/val lines + gap magnitude across slider values | Assignment §4 checklist |
| `plot_multi_feature_optimization_summary()` | Cross-feature optimal value + F1 comparison | Assignment §2 |

### Unchanged from Week 6

| Module | Purpose |
|---|---|
| `data_loader.py` | Parses NVD JSON files, extracts CVSS v3.1 metrics, caches to CSV |
| `feature_engineering.py` | One-hot encodes categoricals, scales numerics, produces ML-ready X/y |
| `distribution.py` | Raw up/downsampling + SMOTE + RandomUnderSampler for target/binary/continuous/categorical |
| `feature_importance.py` | Gini, Permutation, SHAP importance + aggregated ranking |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVD_DATA_DIR` | `C:\Users\Nitish\Downloads\nvdcve-master\nvdcve` | Path to NVD CVE JSON files |

### Optimization Defaults

| Parameter | Default | Range | Description |
|---|---|---|---|
| Search Range | [-0.8, +0.8] | [-1.0, +1.0] | Min/max slider boundaries |
| Step Size | 0.05 | 0.02–0.20 | Increment per AutoML step |
| Improvement Threshold (Δ) | 0.001 | 0.0001–0.01 | Exit when improvement < Δ |
| Max Training Samples | 10,000 | 1,000–108,822 | SVM tractability cap |
| CV Folds | 5 | 2–10 | Stratified k-fold cross-validation |
| Gap Analysis Sliders | [-0.6, -0.3, 0.0, +0.3, +0.6] | — | Points sampled for gap widening |

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **AutoML optimization is slow** | Increase step size to 0.10 or 0.20. Narrow the search range. Reduce max samples. Deselect SVM. |
| **SVM hangs during optimization** | Deselect SVM from algorithm list, or reduce max samples to 5,000 |
| **Learning curves not showing** | Learning curves are computed only at the optimal point. Run AutoML Optimization (Tab 2) first. |
| **Gap widening analysis not showing** | Click the "Run Gap Widening Analysis" button in Tab 3 |
| **High model accuracy (>99%)** | Expected — CVSS vector components directly determine severity. Optimization still finds distribution shifts that improve minority class recall. |
| **"No CVE JSON files found"** | App uses `cve_dataset.csv` by default. Only needs raw JSON files if cache is deleted. |
| **Streamlit deprecation warnings** | Cosmetic only — `use_container_width` warnings do not affect functionality |

---

## License

MIT License — Part of the Sexy Securities project.

---

*Built with Streamlit, scikit-learn, XGBoost, SHAP, and 108k+ real NVD CVE records.*
