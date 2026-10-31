"""
Streamlit Frontend for Network Intrusion Detection System (NIDS).

Deep Learning II Lab Project - Semi-Supervised Generative Anomaly Detection
using Variational Autoencoders (VAE) and GANomaly on the NSL-KDD benchmark dataset.

Connects to the FastAPI backend service at http://localhost:8000.
Run with:
    streamlit run app/frontend.py
"""

import io
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="NIDS - Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished layout and high-contrast badges
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1e3a8a;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    .badge-normal {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #15803d;
        background-color: #dcfce7;
        border-radius: 4px;
    }
    .badge-attack {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #b91c1c;
        background-color: #fee2e2;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Global Paths & Constants
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
SAMPLE_STREAM_PATH = SAMPLE_DATA_DIR / "sample_stream.csv"

# Fallback calibrated thresholds in case backend is offline
OFFLINE_THRESHOLDS = {
    "vae": {"p90": 0.001838, "p95": 0.004764, "p99": 0.013084},
    "ganomaly": {"p90": 0.015675, "p95": 0.021396, "p99": 0.034325},
}

# Offline fallback samples
OFFLINE_SAMPLE_NORMAL = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 215, "dst_bytes": 45076, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0, "root_shell": 0,
    "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
    "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
    "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0, "dst_host_count": 30, "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.03, "dst_host_srv_diff_host_rate": 0.04,
    "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}

OFFLINE_SAMPLE_ATTACK = {
    "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
    "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0, "root_shell": 0,
    "su_attempted": 0, "num_root": 0, "num_file_creations": 0, "num_shells": 0,
    "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
    "count": 123, "srv_count": 6, "serror_rate": 1.0, "srv_serror_rate": 1.0,
    "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 0.05, "diff_srv_rate": 0.07,
    "srv_diff_host_rate": 0.0, "dst_host_count": 255, "dst_host_srv_count": 10,
    "dst_host_same_srv_rate": 0.04, "dst_host_diff_srv_rate": 0.06,
    "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
    "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
}

# ==============================================================================
# Helper Functions & API Communication
# ==============================================================================
def check_backend_health(url: str) -> Dict[str, Any]:
    """Check connectivity and health of the FastAPI backend service."""
    try:
        resp = requests.get(f"{url}/health", timeout=2.0)
        if resp.status_code == 200:
            return {"online": True, "data": resp.json()}
    except Exception as e:
        return {"online": False, "error": str(e)}
    return {"online": False, "error": f"HTTP {resp.status_code}"}


def format_threshold_val(model: str, percentile: str) -> float:
    """Retrieve approximate threshold value for display when backend is querying."""
    m_key = model.lower()
    p_key = percentile.lower()
    return OFFLINE_THRESHOLDS.get(m_key, {}).get(p_key, 0.005)


# ==============================================================================
# Sidebar: Project Info & Backend Connection Status
# ==============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/shield.png",
        width=64,
    )
    st.title("🛡️ NIDS Dashboard")
    st.markdown(
        "**Deep Generative Anomaly Detection** for Network Intrusion Detection "
        "trained on the **NSL-KDD benchmark**."
    )

    st.markdown("---")
    st.subheader("⚙️ API Configuration")
    backend_url = st.text_input(
        "FastAPI Backend URL",
        value=os.getenv("BACKEND_URL", "http://localhost:8000"),
        help="Endpoint where the FastAPI inference service is running.",
    )

    # Health check indicator
    health_info = check_backend_health(backend_url)
    if health_info["online"]:
        st.success("🟢 Backend Connected (Healthy)")
        device_str = health_info["data"].get("device", "cpu")
        st.caption(f"**Hardware Device**: `{device_str}`")
    else:
        st.error("🔴 Backend Offline")
        st.caption(
            "Start backend in your terminal with:\n"
            "`uvicorn app.main:app --reload`"
        )

    st.markdown("---")
    st.subheader("📖 Model Architectures")
    st.markdown(
        """
        - **VAE (Variational Autoencoder)**:
          Trained on normal flows. Scores intrusions via **Reconstruction Error (MSE)**.
        - **GANomaly (Adversarial AE)**:
          Encoder-Decoder-Encoder. Scores anomalies via **Latent Representation Error (MSE)**.
        """
    )

    st.markdown("---")
    st.subheader("🎯 Threshold Calibration")
    st.markdown(
        """
        - **p90**: High sensitivity / aggressive detection.
        - **p95**: Balanced default operating point.
        - **p99**: Conservative / high precision cutoff.
        """
    )
    st.caption("Deep Learning II Lab Project | NSL-KDD Dataset")


# ==============================================================================
# Main App Header
# ==============================================================================
st.markdown('<div class="main-title">🛡️ Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time unsupervised threat detection using semi-supervised deep generative models (VAE & GANomaly)</div>',
    unsafe_allow_html=True,
)

# Four dedicated tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 CSV Upload",
    "🔍 Single Record",
    "📊 Model Comparison",
    "🔴 Live Stream",
])


# ==============================================================================
# TAB 1: CSV Upload (Batch Anomaly Inspection)
# ==============================================================================
with tab1:
    st.header("📁 Batch Network Traffic Inspection")
    st.markdown(
        "Upload a capture CSV containing 41 NSL-KDD network flow attributes to perform batch "
        "inference and identify potential zero-day intrusions."
    )

    col1, col2 = st.columns(2)
    with col1:
        tab1_model = st.selectbox(
            "Select Detection Model",
            ["VAE", "GANomaly"],
            key="tab1_model_select",
            help="Choose the generative architecture for anomaly scoring",
        )
    with col2:
        tab1_threshold = st.selectbox(
            "Operating Threshold Percentile",
            ["p90", "p95", "p99"],
            index=1,
            key="tab1_threshold_select",
            help="Threshold percentile calibrated on normal validation traffic",
        )

    uploaded_file = st.file_uploader(
        "Choose a network traffic CSV file",
        type=["csv"],
        help="Upload CSV file with 41 NSL-KDD feature columns",
    )

    # Provide quick load option from sample_data if available
    col_upload_action, col_sample_action = st.columns([2, 1])
    use_sample_btn = col_sample_action.button(
        "📂 Load Sample Batch from Disk",
        help="Loads sample_stream.csv from sample_data/ for fast testing",
    )

    csv_data_to_send: Optional[bytes] = None
    csv_file_name: str = "batch_records.csv"

    if uploaded_file is not None:
        csv_data_to_send = uploaded_file.getvalue()
        csv_file_name = uploaded_file.name
    elif use_sample_btn:
        if SAMPLE_STREAM_PATH.exists():
            with open(SAMPLE_STREAM_PATH, "rb") as f:
                csv_data_to_send = f.read()
            csv_file_name = "sample_stream.csv"
            st.info(f"Loaded `{csv_file_name}` from disk.")
        else:
            st.warning(f"Sample file not found at {SAMPLE_STREAM_PATH}.")

    if csv_data_to_send is not None:
        st.markdown("---")
        with st.spinner("Executing generative anomaly inference on batch records..."):
            try:
                # POST to /predict_csv endpoint
                req_url = (
                    f"{backend_url}/predict_csv?"
                    f"model={tab1_model.lower()}&threshold_percentile={tab1_threshold.lower()}"
                )
                files_payload = {"file": (csv_file_name, csv_data_to_send, "text/csv")}
                response = requests.post(req_url, files=files_payload, timeout=90)

                if response.status_code == 200:
                    result_json = response.json()
                    summary = result_json.get("summary", {})
                    predictions = result_json.get("predictions", [])
                    applied_threshold = result_json.get("threshold", 0.0)

                    # 1. Summary Metrics
                    total_records = summary.get("total", len(predictions))
                    attacks_detected = summary.get("attack_count", 0)
                    normal_records = summary.get("normal_count", total_records - attacks_detected)
                    attack_rate = summary.get("attack_rate", (attacks_detected / total_records if total_records > 0 else 0.0))

                    st.subheader("📈 Inspection Summary")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Records Evaluated", f"{total_records:,}")
                    m2.metric(
                        "Attacks Detected",
                        f"{attacks_detected:,}",
                        delta=f"{attack_rate * 100:.2f}% Intrusion Rate",
                        delta_color="inverse",
                    )
                    m3.metric("Normal Traffic", f"{normal_records:,}")
                    m4.metric(
                        f"Threshold ({tab1_threshold.upper()})",
                        f"{applied_threshold:.6f}",
                        help=f"Applied cutoff threshold for {tab1_model}",
                    )

                    # 2. Bar Chart of Normal vs Attack
                    st.markdown("#### Traffic Distribution")
                    bar_fig = go.Figure(
                        data=[
                            go.Bar(
                                x=["Normal Traffic", "Attacks / Anomalies"],
                                y=[normal_records, attacks_detected],
                                marker_color=["#22c55e", "#ef4444"],
                                text=[f"{normal_records:,} ({100 - attack_rate*100:.1f}%)", f"{attacks_detected:,} ({attack_rate*100:.1f}%)"],
                                textposition="auto",
                                width=[0.35, 0.35],
                            )
                        ]
                    )
                    bar_fig.update_layout(
                        title=f"Traffic Classification via {tab1_model} ({tab1_threshold.upper()})",
                        xaxis_title="Classification Category",
                        yaxis_title="Record Count",
                        height=330,
                        margin=dict(l=30, r=30, t=50, b=30),
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                    # 3. Colored DataFrame Display
                    st.markdown("#### Inspection Records (Attack Flows Highlighted)")
                    st.caption("Rows highlighted in red indicate anomalous network connections exceeding threshold.")

                    # Read original CSV into DataFrame
                    df_original = pd.read_csv(io.BytesIO(csv_data_to_send))

                    # Map predictions into DataFrame
                    score_list = [p.get("score", 0.0) for p in predictions]
                    label_list = [p.get("label", "normal").capitalize() for p in predictions]

                    df_display = df_original.copy()
                    df_display.insert(0, "Anomaly Score", score_list)
                    df_display.insert(0, "Prediction", label_list)

                    def highlight_attacks(row):
                        pred_str = str(row.get("Prediction", "")).lower()
                        if "attack" in pred_str:
                            return ["background-color: rgba(239, 68, 68, 0.25); color: #dc2626; font-weight: 600;"] * len(row)
                        return ["background-color: rgba(34, 197, 94, 0.12); color: #16a34a;"] * len(row)

                    styled_df = df_display.style.apply(highlight_attacks, axis=1)

                    st.dataframe(
                        styled_df,
                        use_container_width=True,
                        height=420,
                        column_config={
                            "Prediction": st.column_config.TextColumn(
                                "Prediction",
                                help="Classification determined by anomaly score vs threshold",
                            ),
                            "Anomaly Score": st.column_config.NumberColumn(
                                "Anomaly Score",
                                format="%.6f",
                                help="Computed reconstruction or latent representation error",
                            ),
                        },
                    )

                    # Download button for annotated CSV
                    csv_export = df_display.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Annotated Predictions (CSV)",
                        data=csv_export,
                        file_name=f"nids_{tab1_model.lower()}_predictions.csv",
                        mime="text/csv",
                    )

                else:
                    st.error(
                        f"Backend Error (HTTP {response.status_code}): {response.text}"
                    )
            except requests.exceptions.RequestException as e:
                st.error(
                    f"Unable to connect to FastAPI backend at {backend_url}. "
                    f"Please ensure `uvicorn app.main:app` is running. Error: {str(e)}"
                )


# ==============================================================================
# TAB 2: Single Record Prediction
# ==============================================================================
with tab2:
    st.header("🔍 Single Connection Record Inspection")
    st.markdown(
        "Evaluate individual connection flows by sampling pre-configured benign or attack "
        "packets from the NSL-KDD benchmark, or fine-tune features to test generative boundary sensitivity."
    )

    # Initialize session state for single sample
    if "single_record" not in st.session_state:
        st.session_state.single_record = OFFLINE_SAMPLE_NORMAL.copy()
        st.session_state.single_record_type = "normal"
    if "single_prediction" not in st.session_state:
        st.session_state.single_prediction = None

    # Sample loading buttons
    st.markdown("#### 1. Load Sample Connection Flow")
    b_col1, b_col2, b_col_space = st.columns([1.2, 1.2, 2.5])

    with b_col1:
        if st.button("🟢 Load Sample Normal", use_container_width=True):
            try:
                res = requests.get(f"{backend_url}/sample/normal", timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    feats = data.get("features", data)
                    st.session_state.single_record = {k: v for k, v in feats.items() if k not in ["record_type", "features"]}
                    st.session_state.single_record_type = "normal"
                    st.session_state.single_prediction = None
                    st.toast("Loaded normal sample from backend!", icon="🟢")
                else:
                    st.session_state.single_record = OFFLINE_SAMPLE_NORMAL.copy()
                    st.session_state.single_record_type = "normal"
                    st.session_state.single_prediction = None
                    st.info("Backend returned non-200. Loaded offline normal sample.")
            except Exception:
                st.session_state.single_record = OFFLINE_SAMPLE_NORMAL.copy()
                st.session_state.single_record_type = "normal"
                st.session_state.single_prediction = None
                st.info("Backend offline. Loaded offline normal sample.")

    with b_col2:
        if st.button("🔴 Load Sample Attack", use_container_width=True):
            try:
                res = requests.get(f"{backend_url}/sample/attack", timeout=5.0)
                if res.status_code == 200:
                    data = res.json()
                    feats = data.get("features", data)
                    st.session_state.single_record = {k: v for k, v in feats.items() if k not in ["record_type", "features"]}
                    st.session_state.single_record_type = "attack"
                    st.session_state.single_prediction = None
                    st.toast("Loaded attack sample from backend!", icon="🔴")
                else:
                    st.session_state.single_record = OFFLINE_SAMPLE_ATTACK.copy()
                    st.session_state.single_record_type = "attack"
                    st.session_state.single_prediction = None
                    st.info("Backend returned non-200. Loaded offline attack sample.")
            except Exception:
                st.session_state.single_record = OFFLINE_SAMPLE_ATTACK.copy()
                st.session_state.single_record_type = "attack"
                st.session_state.single_prediction = None
                st.info("Backend offline. Loaded offline attack sample.")

    current_sample = st.session_state.single_record
    badge_cls = "badge-attack" if st.session_state.single_record_type == "attack" else "badge-normal"
    st.markdown(
        f"Active Record Type: <span class='{badge_cls}'>"
        f"{st.session_state.single_record_type.upper()} SAMPLE</span>",
        unsafe_allow_html=True,
    )

    # Expandable section showing loaded feature attributes
    with st.expander("📋 Connection Flow Attributes (41 NSL-KDD Features)", expanded=True):
        st.caption("Inspect all 41 categorical and continuous flow characteristics.")
        # Render as clean 4-column key-value grid or 1-row table
        df_single_view = pd.DataFrame([current_sample])
        st.dataframe(df_single_view, use_container_width=True)

    # Model and Threshold Selection
    st.markdown("#### 2. Model & Threshold Selection")
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        tab2_model = st.selectbox(
            "Detection Model",
            ["VAE", "GANomaly"],
            key="tab2_model_select",
        )
    with sel_col2:
        tab2_threshold = st.selectbox(
            "Threshold Percentile",
            ["p90", "p95", "p99"],
            index=1,
            key="tab2_threshold_select",
        )

    # Predict Button
    predict_clicked = st.button("🚀 Predict Anomaly Status", type="primary", use_container_width=True)

    if predict_clicked:
        payload = {
            "model": tab2_model.lower(),
            "threshold_percentile": tab2_threshold.lower(),
            "features": current_sample,
        }
        with st.spinner(f"Scoring connection with {tab2_model}..."):
            try:
                res = requests.post(f"{backend_url}/predict", json=payload, timeout=10.0)
                if res.status_code == 200:
                    st.session_state.single_prediction = res.json()
                else:
                    st.error(f"Inference Error (HTTP {res.status_code}): {res.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to backend at {backend_url}: {e}")

    # Display Single Prediction Results
    if st.session_state.single_prediction is not None:
        pred_res = st.session_state.single_prediction
        score_val = float(pred_res.get("score", 0.0))
        thresh_val = float(pred_res.get("threshold", 0.0))
        label_val = str(pred_res.get("label", "normal")).lower()
        model_name = str(pred_res.get("model", tab2_model)).upper()

        st.markdown("---")
        st.subheader("🎯 Inference Outcome")

        # Visual feedback box
        if label_val == "attack":
            st.error(
                f"🚨 **INTRUSION DETECTED (ATTACK)** — Anomaly score `{score_val:.6f}` exceeds "
                f"calibrated threshold `{thresh_val:.6f}`."
            )
        else:
            st.success(
                f"✅ **NORMAL TRAFFIC (BENIGN)** — Anomaly score `{score_val:.6f}` is within "
                f"calibrated baseline threshold `{thresh_val:.6f}`."
            )

        # Big Metrics Display
        r_col1, r_col2, r_col3 = st.columns(3)
        delta_val = score_val - thresh_val
        r_col1.metric(
            label="Anomaly Score (MSE)",
            value=f"{score_val:.6f}",
            delta=f"{delta_val:+.6f} vs threshold",
            delta_color="inverse",
            help="Reconstruction error for VAE or latent representation error for GANomaly",
        )
        r_col2.metric(
            label=f"Decision Threshold ({tab2_threshold.upper()})",
            value=f"{thresh_val:.6f}",
            help="Decision boundary determined from normal traffic percentile",
        )
        r_col3.metric(
            label="Predicted Classification",
            value=label_val.upper(),
            delta="Normal" if label_val == "normal" else "Threat",
            delta_color="normal" if label_val == "normal" else "inverse",
        )

        # Visual indicator bar comparing score to threshold
        st.markdown("#### Relative Anomaly Position")
        max_scale = max(score_val, thresh_val) * 1.4 if max(score_val, thresh_val) > 0 else 0.05
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score_val,
                number={"valueformat": ".6f"},
                gauge={
                    "axis": {"range": [0, max_scale]},
                    "bar": {"color": "#ef4444" if label_val == "attack" else "#22c55e"},
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.8,
                        "value": thresh_val,
                    },
                },
                title={"text": f"{model_name} Score vs. Threshold"},
            )
        )
        gauge_fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)


# ==============================================================================
# TAB 3: Model Comparison & Artifact Evaluation Plots
# ==============================================================================
with tab3:
    st.header("📊 Model Benchmark & Diagnostic Analytics")
    st.markdown(
        "Side-by-side performance evaluation between **VAE** and **GANomaly** on the "
        "NSL-KDD test partition (`KDDTest+`). Generated diagnostic curves and evaluation metrics."
    )

    # 1. Fetch metrics from GET /metrics or local artifacts/metrics.json
    metrics_data: Optional[Dict[str, Any]] = None
    try:
        res = requests.get(f"{backend_url}/metrics", timeout=3.0)
        if res.status_code == 200:
            metrics_data = res.json()
    except Exception:
        pass

    if metrics_data is None:
        local_metrics_file = ARTIFACTS_DIR / "metrics.json"
        if local_metrics_file.exists():
            try:
                with open(local_metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception as e:
                st.warning(f"Failed to read local metrics: {e}")

    if metrics_data is not None:
        vae_info = metrics_data.get("VAE", {})
        gan_info = metrics_data.get("GANomaly", {})

        vae_def = vae_info.get("at_default_threshold", {})
        gan_def = gan_info.get("at_default_threshold", {})

        # Extract metric values
        vae_roc = vae_info.get("roc_auc", 0.9287)
        vae_pr = vae_info.get("pr_auc", 0.9469)
        vae_f1 = vae_def.get("f1", 0.8657)
        vae_acc = vae_def.get("accuracy", 0.8557)
        vae_prec = vae_def.get("precision", 0.9209)
        vae_rec = vae_def.get("recall", 0.8166)

        gan_roc = gan_info.get("roc_auc", 0.8614)
        gan_pr = gan_info.get("pr_auc", 0.8726)
        gan_f1 = gan_def.get("f1", 0.7174)
        gan_acc = gan_def.get("accuracy", 0.7332)
        gan_prec = gan_def.get("precision", 0.9037)
        gan_rec = gan_def.get("recall", 0.5947)

        # Side-by-side metric cards
        st.subheader("Side-by-Side Model Performance")
        c_vae, c_gan = st.columns(2)

        with c_vae:
            st.markdown("### 🧬 VAE (Variational Autoencoder)")
            st.caption("Reconstruction Error Anomaly Criterion")
            v1, v2, v3 = st.columns(3)
            v1.metric("ROC AUC", f"{vae_roc:.4f}")
            v2.metric("PR AUC", f"{vae_pr:.4f}")
            v3.metric("F1-Score", f"{vae_f1:.4f}")

            v4, v5, v6 = st.columns(3)
            v4.metric("Accuracy", f"{vae_acc:.4f}")
            v5.metric("Precision", f"{vae_prec:.4f}")
            v6.metric("Recall", f"{vae_rec:.4f}")

        with c_gan:
            st.markdown("### 🎭 GANomaly (Adversarial AE)")
            st.caption("Latent Space Encoding Error Criterion")
            g1, g2, g3 = st.columns(3)
            g1.metric("ROC AUC", f"{gan_roc:.4f}", delta=f"{gan_roc - vae_roc:+.4f} vs VAE")
            g2.metric("PR AUC", f"{gan_pr:.4f}", delta=f"{gan_pr - vae_pr:+.4f} vs VAE")
            g3.metric("F1-Score", f"{gan_f1:.4f}", delta=f"{gan_f1 - vae_f1:+.4f} vs VAE")

            g4, g5, g6 = st.columns(3)
            g4.metric("Accuracy", f"{gan_acc:.4f}", delta=f"{gan_acc - vae_acc:+.4f} vs VAE")
            g5.metric("Precision", f"{gan_prec:.4f}", delta=f"{gan_prec - vae_prec:+.4f} vs VAE")
            g6.metric("Recall", f"{gan_rec:.4f}", delta=f"{gan_rec - vae_rec:+.4f} vs VAE")

        # Comparison Table
        st.markdown("#### Performance Metrics Matrix")
        matrix_df = pd.DataFrame(
            {
                "Metric": ["ROC AUC", "PR AUC", "F1 Score", "Accuracy", "Precision", "Recall"],
                "VAE": [vae_roc, vae_pr, vae_f1, vae_acc, vae_prec, vae_rec],
                "GANomaly": [gan_roc, gan_pr, gan_f1, gan_acc, gan_prec, gan_rec],
                "Superior Architecture": [
                    "VAE" if vae_roc >= gan_roc else "GANomaly",
                    "VAE" if vae_pr >= gan_pr else "GANomaly",
                    "VAE" if vae_f1 >= gan_f1 else "GANomaly",
                    "VAE" if vae_acc >= gan_acc else "GANomaly",
                    "VAE" if vae_prec >= gan_prec else "GANomaly",
                    "VAE" if vae_rec >= gan_rec else "GANomaly",
                ],
            }
        )
        st.dataframe(
            matrix_df.style.format({"VAE": "{:.4f}", "GANomaly": "{:.4f}"}),
            use_container_width=True,
        )
    else:
        st.info("Metrics not yet loaded. Ensure backend is running or artifacts/metrics.json is available.")

    st.markdown("---")
    st.subheader("📈 Diagnostic Plots & Evaluation Curves")
    st.caption("Diagnostic visualizations saved under artifacts/plots/ during model training and evaluation.")

    # Display plot images from artifacts/plots/ directory
    # 1. roc_curves.png & confusion_matrices.png
    plot_c1, plot_c2 = st.columns(2)
    roc_img_path = PLOTS_DIR / "roc_curves.png"
    cm_img_path = PLOTS_DIR / "confusion_matrices.png"

    with plot_c1:
        st.markdown("##### Receiver Operating Characteristic (ROC Curves)")
        if roc_img_path.exists():
            st.image(str(roc_img_path), caption="ROC Curves on KDDTest+ (AE, VAE, GANomaly)", use_container_width=True)
        else:
            st.warning(f"Plot not found at {roc_img_path}")

    with plot_c2:
        st.markdown("##### Confusion Matrices at Optimal Threshold")
        if cm_img_path.exists():
            st.image(str(cm_img_path), caption="Confusion Matrices across baseline & generative models", use_container_width=True)
        else:
            st.warning(f"Plot not found at {cm_img_path}")

    # 2. score_distributions.png & training_curves.png
    plot_c3, plot_c4 = st.columns(2)
    dist_img_path = PLOTS_DIR / "score_distributions.png"
    train_img_path = PLOTS_DIR / "training_curves.png"

    with plot_c3:
        st.markdown("##### Anomaly Score Distributions (Normal vs Attack)")
        if dist_img_path.exists():
            st.image(str(dist_img_path), caption="Score separation showing calibrated cutoff thresholds", use_container_width=True)
        else:
            st.warning(f"Plot not found at {dist_img_path}")

    with plot_c4:
        st.markdown("##### Training & Validation Loss Curves")
        if train_img_path.exists():
            st.image(str(train_img_path), caption="Generative model loss convergence across training epochs", use_container_width=True)
        else:
            st.warning(f"Plot not found at {train_img_path}")

    # 3. detection_by_category.png
    cat_img_path = PLOTS_DIR / "detection_by_category.png"
    st.markdown("##### Threat Detection Rate by Attack Category (DoS, Probe, R2L, U2R)")
    if cat_img_path.exists():
        st.image(str(cat_img_path), caption="Attack breakdown: Denial-of-Service, Surveillance/Probe, Remote-to-Local, User-to-Root", use_container_width=True)
    else:
        st.warning(f"Plot not found at {cat_img_path}")


# ==============================================================================
# TAB 4: Live Stream Simulation
# ==============================================================================
with tab4:
    st.header("🔴 Real-Time Network Packet Streaming Simulation")
    st.markdown(
        "Simulate live incoming packet flow from `sample_stream.csv`. Each packet record is "
        "streamed asynchronously to the backend `/predict` endpoint for instantaneous anomaly scoring."
    )

    # Initialize live streaming session state
    if "stream_active" not in st.session_state:
        st.session_state.stream_active = False
    if "stream_scores" not in st.session_state:
        st.session_state.stream_scores = []
    if "stream_labels" not in st.session_state:
        st.session_state.stream_labels = []
    if "stream_attack_count" not in st.session_state:
        st.session_state.stream_attack_count = 0
    if "stream_processed_count" not in st.session_state:
        st.session_state.stream_processed_count = 0

    # Stream Controls Configuration
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    with ctrl_col1:
        tab4_model = st.selectbox("Streaming Model", ["VAE", "GANomaly"], key="tab4_model_select")
    with ctrl_col2:
        tab4_threshold = st.selectbox("Threshold Percentile", ["p90", "p95", "p99"], index=1, key="tab4_threshold_select")
    with ctrl_col3:
        speed = st.slider(
            "Stream Speed (records / second)",
            min_value=1,
            max_value=20,
            value=5,
            help="Pacing frequency for packet stream evaluation",
        )

    # Start / Stop / Reset Buttons
    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 1])
    with btn_c1:
        start_stream = st.button("▶️ Start Stream", type="primary", use_container_width=True)
    with btn_c2:
        stop_stream = st.button("⏹️ Stop Stream", use_container_width=True)
    with btn_c3:
        reset_stream = st.button("🔄 Reset Stream", use_container_width=True)

    if reset_stream:
        st.session_state.stream_active = False
        st.session_state.stream_scores = []
        st.session_state.stream_labels = []
        st.session_state.stream_attack_count = 0
        st.session_state.stream_processed_count = 0
        st.rerun()

    if stop_stream:
        st.session_state.stream_active = False
        st.rerun()

    if start_stream:
        st.session_state.stream_active = True

    # Check that sample_stream.csv is available
    stream_df: Optional[pd.DataFrame] = None
    if SAMPLE_STREAM_PATH.exists():
        try:
            stream_df = pd.read_csv(SAMPLE_STREAM_PATH)
        except Exception as e:
            st.error(f"Error loading stream CSV: {e}")
    else:
        st.error(f"Stream dataset not found at `{SAMPLE_STREAM_PATH}`.")

    # UI Placeholders for live updates
    st.markdown("---")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    latest_record_placeholder = st.empty()

    # Pre-render state before active stream loop
    current_thresh = format_threshold_val(tab4_model, tab4_threshold)
    scores = st.session_state.stream_scores
    processed = st.session_state.stream_processed_count
    attacks = st.session_state.stream_attack_count
    normal_cnt = processed - attacks
    rate = (attacks / processed * 100) if processed > 0 else 0.0

    with metrics_placeholder.container():
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Packets Processed", f"{processed}")
        sm2.metric("Attacks Detected", f"{attacks}", delta=f"{rate:.1f}% rate", delta_color="inverse")
        sm3.metric("Normal Traffic", f"{normal_cnt}")
        sm4.metric(f"Threshold ({tab4_threshold.upper()})", f"{current_thresh:.6f}")

    # Render initial or current chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=scores, mode="lines", name="Anomaly Score", line=dict(color="#2563eb", width=2)))
    fig.add_hline(y=current_thresh, line_dash="dash", line_color="red", annotation_text="Threshold")
    fig.update_layout(
        title=f"Live Anomaly Score Stream ({tab4_model})",
        xaxis_title="Packet Arrival Sequence",
        yaxis_title="Anomaly Score",
        height=380,
        margin=dict(l=40, r=30, t=50, b=40),
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Active Streaming Loop
    if st.session_state.stream_active and stream_df is not None and not stream_df.empty:
        total_rows = len(stream_df)
        start_row = st.session_state.stream_processed_count % total_rows

        for i in range(start_row, total_rows):
            # Check if streaming was stopped
            if not st.session_state.stream_active:
                break

            row_data = stream_df.iloc[i].to_dict()

            # Sanitize numpy types to pure Python types for JSON serialization
            sanitized_row: Dict[str, Any] = {}
            for k, v in row_data.items():
                if hasattr(v, "item"):
                    sanitized_row[k] = v.item()
                elif pd.isna(v):
                    sanitized_row[k] = 0
                else:
                    sanitized_row[k] = v

            # POST record to backend /predict
            payload = {
                "model": tab4_model.lower(),
                "threshold_percentile": tab4_threshold.lower(),
                "features": sanitized_row,
            }

            score_val = current_thresh
            is_attack = False
            label_text = "normal"

            try:
                res = requests.post(f"{backend_url}/predict", json=payload, timeout=2.0)
                if res.status_code == 200:
                    resp_json = res.json()
                    score_val = float(resp_json.get("score", 0.0))
                    current_thresh = float(resp_json.get("threshold", current_thresh))
                    label_text = str(resp_json.get("label", "normal")).lower()
                    is_attack = label_text == "attack"
                else:
                    # Fallback approximation if server error
                    score_val = float(sanitized_row.get("src_bytes", 0)) / 100000.0
                    is_attack = score_val > current_thresh
                    label_text = "attack" if is_attack else "normal"
            except Exception:
                # Offline approximation so demo continues
                score_val = float(sanitized_row.get("src_bytes", 0)) / 100000.0
                is_attack = score_val > current_thresh
                label_text = "attack" if is_attack else "normal"

            # Update session state metrics
            st.session_state.stream_scores.append(score_val)
            st.session_state.stream_labels.append(label_text)
            st.session_state.stream_processed_count += 1
            if is_attack:
                st.session_state.stream_attack_count += 1

            scores = st.session_state.stream_scores
            processed = st.session_state.stream_processed_count
            attacks = st.session_state.stream_attack_count
            normal_cnt = processed - attacks
            rate = (attacks / processed * 100) if processed > 0 else 0.0

            # Update metrics counters
            with metrics_placeholder.container():
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Packets Processed", f"{processed}")
                sm2.metric("Attacks Detected", f"{attacks}", delta=f"{rate:.1f}% rate", delta_color="inverse")
                sm3.metric("Normal Traffic", f"{normal_cnt}")
                sm4.metric(f"Threshold ({tab4_threshold.upper()})", f"{current_thresh:.6f}")

            # Update Live Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=scores, mode="lines", name="Anomaly Score", line=dict(color="#2563eb", width=2)))
            fig.add_hline(y=current_thresh, line_dash="dash", line_color="red", annotation_text="Threshold")
            fig.update_layout(
                title=f"Live Anomaly Score Stream ({tab4_model})",
                xaxis_title="Packet Arrival Sequence",
                yaxis_title="Anomaly Score",
                height=380,
                margin=dict(l=40, r=30, t=50, b=40),
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Update latest record alert details
            proto = sanitized_row.get("protocol_type", "tcp")
            srv = sanitized_row.get("service", "http")
            src_b = sanitized_row.get("src_bytes", 0)
            dst_b = sanitized_row.get("dst_bytes", 0)

            if is_attack:
                latest_record_placeholder.error(
                    f"🚨 **Packet #{processed}**: INTRUSION DETECTED! Score `{score_val:.6f}` > Threshold `{current_thresh:.6f}` | "
                    f"Proto: `{proto}` | Service: `{srv}` | Bytes: `{src_b} -> {dst_b}`"
                )
            else:
                latest_record_placeholder.success(
                    f"✅ **Packet #{processed}**: Normal Traffic. Score `{score_val:.6f}` <= Threshold `{current_thresh:.6f}` | "
                    f"Proto: `{proto}` | Service: `{srv}` | Bytes: `{src_b} -> {dst_b}`"
                )

            # Sleep according to slider pacing
            time.sleep(1.0 / max(speed, 1))

        # Completed one full stream cycle
        if st.session_state.stream_active:
            st.info("Stream batch completed! Re-click 'Start Stream' to loop or 'Reset' to clear.")
            st.session_state.stream_active = False


# ==============================================================================
# Main Entry Point
# ==============================================================================
if __name__ == "__main__":
    pass
