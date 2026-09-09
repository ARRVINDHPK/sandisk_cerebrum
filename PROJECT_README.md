# Multi-Resolution Die Yield Prediction System

An end-to-end machine learning system for multi-resolution die yield prediction with interpretable spatial context, developed for semiconductor wafer yield optimization.

---

## Quick Start Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Generate Synthetic Datasets

```bash
python generate_data.py --csv
```
This generates `input/train.csv`, `input/test.csv`, and `input/validation.csv`.

---

## Running the Pipeline

### Train Models

```bash
# Train Model A (Die-Level + Zero-Leakage Spatial Context)
bash scripts/run_model_a.sh

# Train Model B (Model A + 2000-dim Block Signal Features & PyTorch CNN Embeddings)
bash scripts/run_model_b.sh
```

### Evaluate & Compare

```bash
# Evaluate both models on eligible dies (old_label == 0) and generate PR curves & SHAP figures
bash scripts/run_eval.sh
```

### Generate Predictions Submission CSV

```bash
python -m src.dieyield.pipeline predict --model b --input input/validation.csv --output input/submission.csv
```

### Launch Interactive Streamlit Dashboard

```bash
bash scripts/run_dashboard.sh
```

---

## Unit Test Suite

Run all automated unit tests:

```bash
pytest -v
```

Tests cover:
1. `tests/test_metrics.py`: Eligible die filtering (`old_label == 0`) and exact 2x2 confusion matrix calculations.
2. `tests/test_spatial_no_leakage.py`: Static code audit and runtime checks verifying zero leakage of post-test `label`.
3. `tests/test_block_stats.py`: Vectorized 2000-dim sub-die block signal statistical feature extraction.
4. `tests/test_pipeline_smoke.py`: End-to-end training, prediction, and evaluation smoke tests.

---

## Architecture Overview

```
root
├── configs/                          # Model & evaluation YAML configurations
│   ├── model_a.yaml
│   ├── model_b.yaml
│   └── eval.yaml
├── src/dieyield/                     # Core Python Package
│   ├── data/                         # CSV loading with disk caching & wafer-level splits
│   ├── features/                     # Spatial, die-level, block stats, & PyTorch CNN encoder
│   ├── models/                       # LightGBM Model A & B, focal loss, threshold search
│   ├── evaluate/                     # Eligible-die metric harness, comparison & segmented deltas
│   ├── interpret/                    # SHAP, spatial wafer heatmaps, block anomaly attributions
│   └── pipeline.py                   # CLI entrypoint (train, predict, evaluate, interpret)
├── app/dashboard.py                  # Interactive Streamlit dashboard
├── tests/                            # Pytest suite
└── reports/                          # Figures, run logs, and FINAL_REPORT.md
```
