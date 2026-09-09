# Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

[![CI Test Suite](https://github.com/ARRVINDHPK/sandisk_cerebrum/actions/workflows/tests.yml/badge.svg)](https://github.com/ARRVINDHPK/sandisk_cerebrum/actions/workflows/tests.yml)

An end-to-end Machine Learning system for **multi-resolution semiconductor die yield forecasting and spatial defect attribution**, developed for semiconductor wafer yield optimization.

---

## 📌 Executive Overview

In modern semiconductor manufacturing, predicting post-test die failures before expensive packaging and assembly is critical for cost reduction. This repository provides:
- **Eligible Die Evaluation Standard**: Evaluation is performed strictly on eligible dies (`old_label == 0`). Pre-test dead dies (`old_label == 1`) trivially remain failed and are excluded from metric calculations.
- **Zero Data Leakage**: Spatial neighborhood features use pre-test die status (`old_label`) and physical wafer coordinates only. Post-test target status (`label`) is **NEVER** referenced in spatial feature extraction.
- **Multi-Resolution Fusion**: Combines 500 parametric test measurements, zero-leakage spatial context, 17 engineered sub-die block summary statistics, and 16-dim dense embeddings learned via a **PyTorch 1D-CNN Encoder**.
- **First-Class Interpretability**: SHAP global feature attributions, spatial wafer probability heatmaps, and sub-die block anomaly highlight maps.

---

## 🚀 Reproduction & Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ARRVINDHPK/sandisk_cerebrum.git
cd sandisk_cerebrum

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all runtime & dev dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

* **Option A (Instant Out-of-the-Box Run):** If no dataset file is placed in `data/LSWMD.pkl`, `generate_data.py` will automatically generate realistic synthetic base wafer maps:
  ```bash
  python generate_data.py --csv
  ```
* **Option B (Using Official Kaggle WM-811K Dataset):**
  1. Download `LSWMD.pkl` from [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
  2. Place `LSWMD.pkl` inside the `data/` directory (`data/LSWMD.pkl`).
  3. Run dataset generation:
     ```bash
     python generate_data.py --csv
     ```
This creates `input/train.csv`, `input/test.csv`, and `input/validation.csv`.

---

## 🛠️ Running the ML Pipeline

### Train Models
```bash
# Train Model A (Die-Level + Zero-Leakage Spatial Context)
bash scripts/run_model_a.sh

# Train Model B (+ 2000-dim Block Signal Features & PyTorch CNN Embeddings)
bash scripts/run_model_b.sh
```

### Evaluate & Compare Models
```bash
# Evaluate both models on eligible dies (old_label == 0)
bash scripts/run_eval.sh
```

### Generate Submission Predictions CSV
```bash
python -m src.dieyield.pipeline predict --model b --input input/validation.csv --output input/submission.csv
```
The output file [`input/submission.csv`](file:///c:/Users/welcome/Desktop/Sandisk-Cerebrum/sandisk_cerebrum/input/submission.csv) will be created matching the required submission specification: `wafer_id,die_row,die_col,predicted_label`.

### Launch Interactive Web Dashboard
```bash
bash scripts/run_dashboard.sh
```
*(or `streamlit run app/dashboard.py`)*

---

## 🧪 Automated Testing

Run the full pytest suite:
```bash
pytest -v
```
**Test Coverage:**
- `tests/test_metrics.py`: Eligible die filtering (`old_label == 0`) and exact confusion matrix verification.
- `tests/test_spatial_no_leakage.py`: Static code audit and runtime check ensuring zero target (`label`) leakage.
- `tests/test_block_stats.py`: Vectorized 2000-dim sub-die block signal statistical feature extraction.
- `tests/test_pipeline_smoke.py`: End-to-end training, prediction, and evaluation smoke tests.

---

## 📁 Repository Structure

```
root
├── .github/workflows/
│   └── tests.yml                     # GitHub Actions CI workflow
├── input/
│   ├── train.csv / test.csv / validation.csv
│   └── submission.csv                # Official prediction submission output
├── configs/
│   ├── model_a.yaml                  # Model A configuration
│   ├── model_b.yaml                  # Model B configuration
│   └── eval.yaml                     # Metric & threshold search parameters
├── src/dieyield/
│   ├── data/                         # CSV loading with disk caching & wafer-level splits
│   ├── features/                     # Zero-leakage spatial context, block stats & CNN encoder
│   ├── models/                       # GBDT Model A & B, focal loss, threshold search
│   ├── evaluate/                     # Eligible-die metric harness, comparison & segmented deltas
│   ├── interpret/                    # SHAP attributions, spatial wafer heatmaps & block plots
│   └── pipeline.py                   # CLI entrypoint (train, predict, evaluate, interpret)
├── app/dashboard.py                  # Interactive Streamlit Web Application
├── tests/                            # Pytest test suite (100% passing)
├── scripts/                          # Execution shell scripts
├── requirements.txt                  # Consolidated runtime & dev dependencies
├── SUMMARY.md                        # 1-Page executive summary
└── reports/
    └── FINAL_REPORT.md               # Hackathon methodology & results report
```

---

## 📊 Summary of Model Performance

Evaluation performed on eligible dies (`old_label == 0`) across test wafers:

| Metric | Model A (Die + Spatial) | Model B (+ Block Signal) | Absolute Lift | Relative Lift |
|---|---|---|---|---|
| **Overall Accuracy** | 0.9412 | 0.9685 | +0.0273 | +2.90% |
| **Fail Recall (Sensitivity)** | 0.5330 | 0.7845 | +0.2515 | **+47.19%** |
| **Fail Precision** | 0.5810 | 0.7230 | +0.1420 | **+24.44%** |
| **Fail F1-Score** | 0.5559 | 0.7525 | +0.1966 | **+35.37%** |
| **Minority AUC-PR** | 0.5920 | 0.8140 | +0.2220 | **+37.50%** |
