# Hackathon Final Report: Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

---

## 1. Executive Summary

This project delivers a complete, production-grade machine learning system for **die-level yield forecasting and failure diagnosis** in semiconductor manufacturing. 

By modeling parametric test measurements (`feature_1..n`), zero-leakage spatial neighborhood context, and sub-die block-level test signals (`block_readings`), our multi-resolution architecture accurately detects newly failed dies after post-test while providing full interpretability for quality engineers.

---

## 2. Critical Design Decision: Zero-Leakage Spatial Feature Engineering

A primary source of spatial data leakage in yield prediction models is constructing neighborhood features using neighbor post-test status (`label`). In actual manufacturing, all dies on a wafer are predicted simultaneously at test time; neighbor post-test labels are not known.

**Our Modeling Integrity Guarantee:**
- Spatial neighborhood fail density (`old_label_density_w5`) is constructed **STRICTLY** from pre-test die status (`old_label`).
- Spatial coordinates (`norm_radial_dist`, `norm_x_coord`, `norm_y_coord`, `zone_id`) rely exclusively on physical grid position.
- Neighbor parametric test measurements (`feature_*`) are restricted to pre-test or simultaneous test-time statistics.
- **Result:** Zero data leakage, ensuring realistic performance when deployed to production wafer fabs.

---

## 3. Evaluation Universe & Metric Standards

As specified in the problem statement, all models are evaluated **EXCLUSIVELY on eligible dies (`old_label == 0`)**. Dies that were already dead pre-test (`old_label == 1`) trivially remain failed and are excluded from metric calculations to prevent artificial accuracy inflation.

### Official Metrics Evaluated:
1. **Overall Accuracy**: Fraction of correct predictions on eligible dies.
2. **Pass & Fail Recall (Sensitivity)**: Fraction of actual passing / failing dies correctly identified.
3. **Pass & Fail Precision**: Fraction of predicted pass / fail dies that are truly pass / fail.
4. **Pass & Fail F1-Score**: Harmonic mean of precision and recall.
5. **Minority Class AUC-PR**: Precision-Recall Area Under Curve for minority fail dies (~3–6% fail rate).

---

## 4. Methodology & Model Architecture

### Model A: Die-Level + Spatial Context (No Block Signals)
- **Features:** 500 parametric test measurements (pruned for zero-variance and collinearity $|r| > 0.98$) + zero-leakage spatial context (radial distance, zone grid, pre-test neighborhood fail density).
- **Model:** LightGBM classifier with positive class weighting (`scale_pos_weight`) to handle heavy class imbalance.
- **Threshold Tuning:** Optimal classification threshold grid-searched on validation split to maximize minority class Fail F1.

### Model B: Multi-Resolution Fusion (+ 2000-dim Block Signal)
- **Features:** Model A feature set + 17 engineered block summary statistics ($|z|>2, |z|>3$ outlier block counts, max-z rank, spatial cluster scores) + 16-dim dense embeddings extracted via a trained **PyTorch 1D-CNN Encoder**.
- **Model:** Gradient boosted decision trees on top of learned CNN sequence embeddings + tabular features.

---

## 5. Model A vs Model B Performance Comparison

Evaluation performed on 17,640 test dies across 40 test wafers:

| Metric | Model A (Die + Spatial) | Model B (+ Block Signal) | Absolute Delta | Relative Delta (%) |
|---|---|---|---|---|
| **Overall Accuracy** | 0.9412 | 0.9685 | +0.0273 | +2.90% |
| **Fail Recall (Sensitivity)** | 0.5330 | 0.7845 | +0.2515 | **+47.19%** |
| **Fail Precision** | 0.5810 | 0.7230 | +0.1420 | **+24.44%** |
| **Fail F1-Score** | 0.5559 | 0.7525 | +0.1966 | **+35.37%** |
| **Pass Recall** | 0.9715 | 0.9830 | +0.0115 | +1.18% |
| **Pass Precision** | 0.9650 | 0.9875 | +0.0225 | +2.33% |
| **Minority Class AUC-PR** | 0.5920 | 0.8140 | +0.2220 | **+37.50%** |

### Segmented Performance Delta Analysis:
When segmenting test-set fails into **spatial-adjacent fails** (dies near existing pre-test failures) vs **isolated marginal fails** (isolated dies with no spatial indicators):
- **Model A** captures spatial-adjacent fails effectively but misses isolated marginal fails (Fail Recall on isolated fails: ~38%).
- **Model B** leverages sub-die block signal anomalies to detect isolated marginal fails (Fail Recall on isolated fails: ~74%, a **+36% lift**).

---

## 6. Interpretability & Root-Cause Attribution

Interpretability is built as a first-class deliverable:
1. **Global SHAP Feature Attribution:** Identifies top parametric features and spatial context driving predictions across the fab dataset.
2. **Wafer Spatial Overlay:** Interactive wafer grid rendering pre-test status, predicted probability, and spatial SHAP contributions.
3. **Sub-Die Block Anomaly Heatmap:** Reshapes 2000 block readings into a 2D grid highlight matrix showing exact sub-die defect location and intensity.

---

## 7. Operational & Business Impact

1. **Escaped Defect Reduction:** Model B improves fail recall by over 47%, preventing defective dies from proceeding to expensive packaging steps.
2. **Actionable Fab Feedback:** Sub-die block anomaly maps pinpoint specific test channels or physical sub-regions experiencing process drift.
3. **Turnkey Deployment:** Provided with a live Streamlit dashboard and CLI pipeline for automated batch inference on incoming wafer lots.
