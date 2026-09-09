import sys
import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure src package is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dieyield.data.loader import load_split_cached
from src.dieyield.evaluate.metrics import evaluate_predictions, filter_eligible
from src.dieyield.evaluate.compare import generate_comparison_table, segmented_delta_analysis
from src.dieyield.interpret.spatial_heatmap import render_wafer_spatial_heatmap
from src.dieyield.interpret.block_attribution import render_block_anomaly_grid
from src.dieyield.interpret.shap_explain import compute_shap_values, explain_single_die
from src.dieyield.models.model_a import ModelA
from src.dieyield.models.model_b import ModelB
import yaml


st.set_page_config(
    page_title="Die Yield Prediction & Interpretability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Multi-Resolution Die Yield Prediction & Spatial Interpretability")
st.markdown("""
*Interpretable die yield forecasting pipeline comparing **Model A (Die + Spatial)** and **Model B (+ 2000-dim Block Signals)** on eligible dies (`old_label == 0`).*
""")


@st.cache_data
def load_datasets():
    train_path = "input/train.csv"
    test_path = "input/test.csv"
    if not os.path.exists(test_path):
        st.error(f"Dataset missing at {test_path}. Please run dataset generation first.")
        st.stop()
    df_test, blocks_test = load_split_cached(test_path)
    return df_test, blocks_test


@st.cache_resource
def load_models():
    with open("configs/model_a.yaml") as f:
        cfg_a = yaml.safe_load(f)
    with open("configs/model_b.yaml") as f:
        cfg_b = yaml.safe_load(f)

    model_a = ModelA(cfg_a)
    model_b = ModelB(cfg_b)

    runs_dir = Path("reports/runs")
    if (runs_dir / "model_a").exists():
        model_a.load(str(runs_dir / "model_a"))
    if (runs_dir / "model_b").exists():
        model_b.load(str(runs_dir / "model_b"))

    return model_a, model_b


df_test, blocks_test = load_datasets()
model_a, model_b = load_models()

tab1, tab2, tab3 = st.tabs(["📊 Model Comparison & Performance", "🗺️ Wafer Grid Explorer", "🔍 Single Die Deep-Dive (SHAP & Blocks)"])

with tab1:
    st.header("Model A vs Model B Performance Comparison")
    st.info("Evaluation universe = Eligible dies only (`old_label == 0`). Pre-test failures are excluded from scoring.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model A: Die + Spatial Context")
        if model_a.model is not None:
            metrics_a = model_a.evaluate(df_test)
            st.json({k: v for k, v in metrics_a.items() if not isinstance(v, pd.DataFrame)})
            st.dataframe(metrics_a["confusion_matrix_df"], use_container_width=True)
        else:
            st.warning("Model A not trained yet. Run pipeline training first.")
            metrics_a = {}

    with col2:
        st.subheader("Model B: + 2000-dim Block Signals")
        if model_b.model is not None:
            metrics_b = model_b.evaluate(df_test, blocks_test)
            st.json({k: v for k, v in metrics_b.items() if not isinstance(v, pd.DataFrame)})
            st.dataframe(metrics_b["confusion_matrix_df"], use_container_width=True)
        else:
            st.warning("Model B not trained yet. Run pipeline training first.")
            metrics_b = {}

    if metrics_a and metrics_b:
        st.markdown("---")
        st.subheader("Side-by-Side Delta Summary Table")
        cmp_df = generate_comparison_table(metrics_a, metrics_b)
        st.table(cmp_df)

        preds_a = model_a.predict(df_test)
        preds_b = model_b.predict(df_test, blocks_test)
        seg_res = segmented_delta_analysis(df_test, preds_a, preds_b)

        st.subheader("Segmented Performance Delta (Isolated vs Spatial-Adjacent Fails)")
        st.json(seg_res)

with tab2:
    st.header("Wafer Grid Spatial Explorer")
    wafers = df_test["wafer_id"].unique()
    selected_wafer = st.selectbox("Select Wafer ID:", wafers)

    if selected_wafer:
        if model_a.model is not None:
            probs_a = model_a.predict_proba(df_test)
            fig_a = render_wafer_spatial_heatmap(df_test, selected_wafer, probs_a)
            st.pyplot(fig_a)
            plt.close()

with tab3:
    st.header("Single Die Attribution & Sub-Die Block Signal Analysis")
    wafers_sub = df_test["wafer_id"].unique()
    sel_w = st.selectbox("Select Wafer for Die Deep-Dive:", wafers_sub, key="die_wafer")

    w_dies = df_test[df_test["wafer_id"] == sel_w]
    selected_die_idx = st.selectbox("Select Die Index:", w_dies.index.tolist())

    if selected_die_idx is not None:
        die_row_data = df_test.iloc[selected_die_idx]
        st.write(f"**Wafer:** `{die_row_data['wafer_id']}` | **Row:** `{die_row_data['die_row']}` | **Col:** `{die_row_data['die_col']}` | **Old Label:** `{die_row_data.get('old_label', 'N/A')}` | **Ground Truth Label:** `{die_row_data.get('label', 'N/A')}`")

        c1, c2 = st.columns(2)
        with c1:
            if model_a.model is not None:
                st.subheader("SHAP Feature Attribution")
                explainer, shap_mat, X_prep = compute_shap_values(model_a, df_test)
                die_shap_df = explain_single_die(shap_mat, X_prep, selected_die_idx, top_k=10)
                st.dataframe(die_shap_df, use_container_width=True)

        with c2:
            if blocks_test is not None:
                st.subheader("Sub-Die 2000-dim Block Reading Heatmap")
                block_sig = blocks_test[selected_die_idx]
                fig_block = render_block_anomaly_grid(block_sig, z_threshold=2.5)
                st.pyplot(fig_block)
                plt.close()
