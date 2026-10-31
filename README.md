---
title: NIDS Dashboard
sdk: docker
app_port: 7860
---
# 🛡️ Network Intrusion Detection System (NIDS)
### Deep Learning II Lab Project - VAE & GANomaly on NSL-KDD

---

## 📌 Project Overview
This project implements an end-to-end **Network Intrusion Detection System (NIDS)** powered by semi-supervised / one-class deep generative models: **Variational Autoencoder (VAE)** and **GANomaly**. The models are trained exclusively on normal network traffic from the benchmark **NSL-KDD dataset** to learn baseline behavior. During inference, unseen attacks and deviations manifest high reconstruction or latent representation errors, enabling robust zero-day and anomaly detection without relying on attack class labels.

The application features a production-ready **FastAPI** backend exposing REST endpoints for single-flow and batch predictions, paired with an interactive **Streamlit** dashboard for monitoring, batch inspection, live stream simulation, and model performance comparisons.

> **Note**: Deep learning models were trained on GPU using Google Colab and deployed here using pre-trained PyTorch weight checkpoints and scikit-learn preprocessor pipelines.

---

## 🏛️ System Architecture

The system decouples model inference from UI presentation:
- **FastAPI Backend (`app/main.py`)**: Asynchronous REST API service handling model loading, request validation, one-hot encoding & standard scaling, inference execution across PyTorch models, and threshold-based decision classification.
- **Streamlit Frontend (`app/frontend.py`)**: Interactive visualization dashboard communicating with the FastAPI backend via HTTP requests. Provides real-time streaming charts, CSV upload inspection, and metric evaluation.
- **PyTorch Models (`models/models.py`)**: Architectural definitions for AE, VAE, and GANomaly along with dedicated anomaly scoring functions.
- **Artifacts Store (`artifacts/`)**: Pre-trained weights, fitted preprocessing pipeline, optimal thresholds, and evaluation curves.

---

## 📁 Project Structure

```
DL2-project/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI backend
│   └── frontend.py      # Streamlit frontend
├── models/
│   ├── __init__.py
│   └── models.py         # PyTorch model definitions
├── artifacts/
│   ├── vae.pt
│   ├── ganomaly_generator.pt
│   ├── preprocessor.joblib
│   ├── thresholds.json
│   ├── config.json
│   ├── feature_columns.json
│   ├── metrics.json
│   └── plots/
├── sample_data/
│   ├── sample_normal.csv
│   ├── sample_attack.csv
│   ├── sample_stream.csv
│   └── sample_stream_with_labels.csv
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Execution Instructions

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed. Create a virtual environment and install project dependencies:

```bash
# Optional: Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Start the Backend API
Run the FastAPI service from the project root directory:

```bash
uvicorn app.main:app --reload
```
The API server will start at `http://127.0.0.1:8000`. You can explore interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 3. Start the Frontend Dashboard
In a **new terminal** (from the project root directory), run:

```bash
streamlit run app/frontend.py
```

### 4. Access the Application
Open your web browser and navigate to:
```
http://localhost:8501
```

---

## 🌟 Application Features

The Streamlit frontend is structured into four dedicated functional tabs:

1. **Single Flow Prediction**:
   - Inspect individual network connection records.
   - Pre-populate inputs with sample benign or attack flows, or manually edit 41 network flow attributes.
   - Real-time gauge visualization of anomaly score vs. calibrated decision threshold for selected model (VAE or GANomaly).
2. **Batch CSV Inspection**:
   - Upload network connection capture CSV files.
   - Run batch inference with summary statistics (Total Flows, Detected Attacks, Normal Traffic %).
   - Filter and explore detected anomalous connections with CSV export capability.
3. **Real-Time Stream Simulation**:
   - Simulate live network traffic arrival using continuous packet batches from sample streams.
   - Configurable stream speed, pause/resume controls, dynamic live anomaly alerts, and rolling intrusion rate charts.
4. **Model Analytics & Comparison**:
   - Side-by-side benchmark comparison between VAE and GANomaly.
   - Evaluation metrics including Precision, Recall, F1-Score, ROC-AUC, and PR-AUC.
   - Pre-computed ROC curves, Precision-Recall curves, and reconstruction error distributions.

---

## 🧠 Deep Learning Models

### 1. Variational Autoencoder (VAE)
- **Concept**: Learns a low-dimensional probabilistic latent representation $q(z|x)$ approximating normal network traffic.
- **Anomaly Criterion**: Reconstruction Error (MSE):
  $$\text{Score}_{\text{VAE}}(x) = \frac{1}{d} \sum_{i=1}^d (x_i - \hat{x}_i)^2$$
- Unseen attacks deviate from the learned latent manifold and yield significantly higher reconstruction errors than normal flows.

### 2. GANomaly
- **Concept**: An adversarial autoencoder architecture consisting of an Encoder1-Decoder-Encoder2 generator ($G$) and a discriminator ($D$).
- **Anomaly Criterion**: Latent Representation Error (MSE between latent bottleneck $z$ and re-encoded latent $z'$):
  $$\text{Score}_{\text{GANomaly}}(x) = \frac{1}{l} \sum_{j=1}^l (z_j - \hat{z}_j)^2$$
- Because $G$ is trained to reconstruct only normal samples, when an attack is processed, the second encoder fails to match the original latent space, resulting in elevated latent distance.

---

## 📊 Dataset: NSL-KDD
- **Description**: Standard benchmark dataset resolving redundancy and synthetic biases present in the original KDD Cup 99 dataset.
- **Features**: 41 total connection features including:
  - Basic features: `duration`, `protocol_type`, `service`, `flag`, `src_bytes`, `dst_bytes`.
  - Content features: `hot`, `num_failed_logins`, `logged_in`, `num_compromised`, `root_shell`.
  - Time-based traffic features: `count`, `srv_count`, `serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`.
  - Host-based traffic features: `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`, etc.
- **Training Strategy**: Models were trained solely on **Normal** network records to establish unsupervised boundary detection.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check API status, loaded models, and preprocessor state |
| `GET` | `/models/info` | Retrieve model architectures, optimal thresholds, and benchmark metrics |
| `POST` | `/predict` | Predict anomaly status and score for a single network flow record |
| `POST` | `/predict/batch` | High-throughput batch inference for tabular network records |
| `GET` | `/samples` | Retrieve pre-packaged normal and attack test records for quick testing |
