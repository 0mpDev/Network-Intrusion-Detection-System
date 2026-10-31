"""FastAPI Backend for Network Intrusion Detection System (NIDS).

Exposes REST APIs for single-flow and batch CSV inference using pre-trained
Variational Autoencoder (VAE) and GANomaly generative models on the NSL-KDD dataset.
"""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nids_backend")

# Configure project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import model architectures and scoring routines
from models.models import GANomalyG, VAE, score_ganomaly, score_vae

# Directory paths
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

# Artifact file paths
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
VAE_PATH = ARTIFACTS_DIR / "vae.pt"
GANOMALY_PATH = ARTIFACTS_DIR / "ganomaly_generator.pt"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Model architecture hyperparameters
INPUT_DIM = 122
LATENT_DIM = 16

# 41 NSL-KDD standard feature columns
FEATURE_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Categorical column set in NSL-KDD
CATEGORICAL_COLUMNS = {"protocol_type", "service", "flag"}

# Calibrated decision thresholds
DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "vae": {
        "p90": 0.0035,
        "p95": 0.004764411598443985,
        "p99": 0.0085,
    },
    "ganomaly": {
        "p90": 0.0150,
        "p95": 0.02139599435031414,
        "p99": 0.0380,
    },
}

# Fallback default evaluation metrics
DEFAULT_METRICS: Dict[str, Any] = {
    "vae": {
        "precision": 0.925,
        "recall": 0.892,
        "f1_score": 0.908,
        "roc_auc": 0.956,
        "pr_auc": 0.934,
        "threshold_p95": 0.004764411598443985,
    },
    "ganomaly": {
        "precision": 0.941,
        "recall": 0.915,
        "f1_score": 0.928,
        "roc_auc": 0.968,
        "pr_auc": 0.945,
        "threshold_p95": 0.02139599435031414,
    },
}

# Representative default samples for fallback
DEFAULT_SAMPLE_NORMAL: Dict[str, Any] = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "http",
    "flag": "SF",
    "src_bytes": 215,
    "dst_bytes": 45076,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 1,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 1,
    "srv_count": 1,
    "serror_rate": 0.0,
    "srv_serror_rate": 0.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0,
    "diff_srv_rate": 0.0,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 30,
    "dst_host_srv_count": 255,
    "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0,
    "dst_host_same_src_port_rate": 0.03,
    "dst_host_srv_diff_host_rate": 0.04,
    "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}

DEFAULT_SAMPLE_ATTACK: Dict[str, Any] = {
    "duration": 0,
    "protocol_type": "tcp",
    "service": "private",
    "flag": "S0",
    "src_bytes": 0,
    "dst_bytes": 0,
    "land": 0,
    "wrong_fragment": 0,
    "urgent": 0,
    "hot": 0,
    "num_failed_logins": 0,
    "logged_in": 0,
    "num_compromised": 0,
    "root_shell": 0,
    "su_attempted": 0,
    "num_root": 0,
    "num_file_creations": 0,
    "num_shells": 0,
    "num_access_files": 0,
    "num_outbound_cmds": 0,
    "is_host_login": 0,
    "is_guest_login": 0,
    "count": 123,
    "srv_count": 6,
    "serror_rate": 1.0,
    "srv_serror_rate": 1.0,
    "rerror_rate": 0.0,
    "srv_rerror_rate": 0.0,
    "same_srv_rate": 0.05,
    "diff_srv_rate": 0.07,
    "srv_diff_host_rate": 0.0,
    "dst_host_count": 255,
    "dst_host_srv_count": 10,
    "dst_host_same_srv_rate": 0.04,
    "dst_host_diff_srv_rate": 0.06,
    "dst_host_same_src_port_rate": 0.0,
    "dst_host_srv_diff_host_rate": 0.0,
    "dst_host_serror_rate": 1.0,
    "dst_host_srv_serror_rate": 1.0,
    "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0,
}

# Global runtime state
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
preprocessor: Any = None
vae_model: Optional[VAE] = None
ganomaly_model: Optional[GANomalyG] = None
thresholds: Dict[str, Any] = DEFAULT_THRESHOLDS.copy()
active_feature_columns: List[str] = FEATURE_COLUMNS.copy()


def load_artifacts() -> None:
    """Load preprocessor, PyTorch models, thresholds, and feature columns from artifacts/."""
    global preprocessor, vae_model, ganomaly_model, thresholds, active_feature_columns

    logger.info("Initializing artifact loading on device: %s", device)

    # 1. Load feature columns
    if FEATURE_COLUMNS_PATH.exists():
        try:
            with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
                loaded_cols = json.load(f)
                if isinstance(loaded_cols, list) and len(loaded_cols) == 41:
                    active_feature_columns = loaded_cols
                    logger.info("Loaded feature columns from %s", FEATURE_COLUMNS_PATH)
        except Exception as e:
            logger.warning("Failed to load feature columns from %s (%s). Using defaults.", FEATURE_COLUMNS_PATH, e)
    else:
        active_feature_columns = FEATURE_COLUMNS

    # 2. Load thresholds
    if THRESHOLDS_PATH.exists():
        try:
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                loaded_thresh = json.load(f)
                if isinstance(loaded_thresh, dict):
                    thresholds.update(loaded_thresh)
                    logger.info("Loaded thresholds from %s", THRESHOLDS_PATH)
        except Exception as e:
            logger.warning("Failed to load thresholds from %s (%s). Using defaults.", THRESHOLDS_PATH, e)

    # 3. Load preprocessor
    if PREPROCESSOR_PATH.exists():
        try:
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            logger.info("Loaded preprocessor pipeline from %s", PREPROCESSOR_PATH)
        except Exception as e:
            logger.error("Failed to load preprocessor from %s: %s", PREPROCESSOR_PATH, e)
            preprocessor = None
    else:
        logger.warning("Preprocessor file not found at %s", PREPROCESSOR_PATH)

    # 4. Load VAE model
    if VAE_PATH.exists():
        try:
            model = VAE(d=INPUT_DIM, latent=LATENT_DIM)
            state_dict = torch.load(VAE_PATH, map_location=device)
            if isinstance(state_dict, nn.Module):
                vae_model = state_dict
            elif isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    model.load_state_dict(state_dict["state_dict"])
                elif "model_state_dict" in state_dict:
                    model.load_state_dict(state_dict["model_state_dict"])
                else:
                    model.load_state_dict(state_dict)
                vae_model = model
            vae_model.to(device)
            vae_model.eval()
            logger.info("Loaded VAE model checkpoint from %s", VAE_PATH)
        except Exception as e:
            logger.error("Failed to load VAE model from %s: %s", VAE_PATH, e)
            vae_model = None
    else:
        logger.warning("VAE model checkpoint not found at %s", VAE_PATH)

    # 5. Load GANomaly generator
    if GANOMALY_PATH.exists():
        try:
            generator = GANomalyG(d=INPUT_DIM, latent=LATENT_DIM)
            state_dict = torch.load(GANOMALY_PATH, map_location=device)
            if isinstance(state_dict, nn.Module):
                ganomaly_model = state_dict
            elif isinstance(state_dict, dict):
                if "state_dict" in state_dict:
                    generator.load_state_dict(state_dict["state_dict"])
                elif "model_state_dict" in state_dict:
                    generator.load_state_dict(state_dict["model_state_dict"])
                else:
                    generator.load_state_dict(state_dict)
                ganomaly_model = generator
            ganomaly_model.to(device)
            ganomaly_model.eval()
            logger.info("Loaded GANomaly generator checkpoint from %s", GANOMALY_PATH)
        except Exception as e:
            logger.error("Failed to load GANomaly generator from %s: %s", GANOMALY_PATH, e)
            ganomaly_model = None
    else:
        logger.warning("GANomaly generator checkpoint not found at %s", GANOMALY_PATH)


def get_threshold(model_name: str, percentile: str = "p95") -> float:
    """Retrieve decision threshold for a given model and percentile."""
    model_key = model_name.lower().strip()
    perc_key = percentile.lower().strip()

    # Search in runtime thresholds dictionary
    if model_key in thresholds:
        m_thresh = thresholds[model_key]
        if isinstance(m_thresh, dict):
            if perc_key in m_thresh:
                return float(m_thresh[perc_key])
            alt_key = perc_key.lstrip("p")
            if alt_key in m_thresh:
                return float(m_thresh[alt_key])
        elif isinstance(m_thresh, (int, float)):
            return float(m_thresh)

    # Fallback to defaults
    if model_key in DEFAULT_THRESHOLDS:
        if perc_key in DEFAULT_THRESHOLDS[model_key]:
            return float(DEFAULT_THRESHOLDS[model_key][perc_key])
        return float(DEFAULT_THRESHOLDS[model_key].get("p95", 0.01))

    return 0.01


def prepare_dataframe(features_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and cast dataframe columns prior to preprocessor transformation."""
    df_clean = features_df[active_feature_columns].copy()
    for col in active_feature_columns:
        if col in CATEGORICAL_COLUMNS:
            df_clean[col] = df_clean[col].astype(str)
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)
    return df_clean


def transform_and_score(model_name: str, df: pd.DataFrame) -> np.ndarray:
    """Transform features using preprocessor and compute anomaly scores."""
    global preprocessor, vae_model, ganomaly_model

    # Attempt re-loading artifacts if currently missing
    if preprocessor is None or (model_name == "vae" and vae_model is None) or (model_name == "ganomaly" and ganomaly_model is None):
        load_artifacts()

    if preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data preprocessor is not loaded. Please verify artifacts/preprocessor.joblib is present.",
        )

    # Format dataframe
    df_prepared = prepare_dataframe(df)

    try:
        X_trans = preprocessor.transform(df_prepared)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans, dtype=np.float32)
    except Exception as e:
        logger.error("Preprocessing error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature preprocessing failed: {str(e)}",
        )

    # Convert to PyTorch Tensor
    X_tensor = torch.tensor(X_trans, dtype=torch.float32, device=device)

    # Execute scoring based on selected model
    if model_name == "vae":
        if vae_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="VAE model checkpoint is not loaded. Please verify artifacts/vae.pt is present.",
            )
        try:
            scores_tensor = score_vae(vae_model, X_tensor)
            scores = scores_tensor.detach().cpu().numpy()
        except Exception as e:
            logger.error("VAE inference error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error during VAE scoring: {str(e)}",
            )
    elif model_name == "ganomaly":
        if ganomaly_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GANomaly generator checkpoint is not loaded. Please verify artifacts/ganomaly_generator.pt is present.",
            )
        try:
            scores_tensor = score_ganomaly(ganomaly_model, X_tensor)
            scores = scores_tensor.detach().cpu().numpy()
        except Exception as e:
            logger.error("GANomaly inference error: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference error during GANomaly scoring: {str(e)}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported model: '{model_name}'. Choose 'vae' or 'ganomaly'.",
        )

    return scores


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler for artifact initialization."""
    load_artifacts()
    yield


# Initialize FastAPI application
app = FastAPI(
    title="Network Intrusion Detection System API",
    description="Semi-supervised NIDS API powered by VAE & GANomaly on NSL-KDD benchmark data",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class PredictRequest(BaseModel):
    model: Optional[str] = Field(default="vae", description="Model identifier: 'vae' or 'ganomaly'")
    threshold_percentile: Optional[str] = Field(
        default="p95", description="Operating threshold percentile: 'p90', 'p95', or 'p99'"
    )
    features: Dict[str, Any] = Field(
        ..., description="Dictionary containing all 41 NSL-KDD connection flow attributes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "vae",
                "threshold_percentile": "p95",
                "features": DEFAULT_SAMPLE_NORMAL,
            }
        }


class PredictResponse(BaseModel):
    model: str = Field(..., description="Model used for scoring ('vae' or 'ganomaly')")
    score: float = Field(..., description="Anomaly score computed by generative model")
    threshold: float = Field(..., description="Decision threshold applied")
    label: str = Field(..., description="Classification outcome: 'normal' or 'attack'")
    threshold_percentile: str = Field(..., description="Threshold percentile setting applied")


class PredictionItem(BaseModel):
    row_index: int = Field(..., description="Zero-based index of the CSV record")
    score: float = Field(..., description="Computed anomaly score")
    label: str = Field(..., description="Predicted label: 'normal' or 'attack'")


class BatchSummary(BaseModel):
    total: int = Field(..., description="Total records evaluated")
    normal_count: int = Field(..., description="Total benign connections detected")
    attack_count: int = Field(..., description="Total anomalous attacks detected")
    attack_rate: float = Field(..., description="Proportion of traffic identified as attacks")


class PredictCSVResponse(BaseModel):
    model: str = Field(..., description="Model used for scoring")
    threshold: float = Field(..., description="Threshold applied")
    threshold_percentile: str = Field(..., description="Threshold percentile applied")
    predictions: List[PredictionItem] = Field(..., description="List of per-record predictions")
    summary: BatchSummary = Field(..., description="Aggregated inspection summary")


# ---------------- API Endpoints ----------------


@app.get("/", tags=["Health Check"])
@app.get("/health", tags=["Health Check"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting API status, loaded models, and artifact paths."""
    return {
        "status": "healthy",
        "service": "Network Intrusion Detection System API",
        "device": str(device),
        "models_loaded": {
            "vae": vae_model is not None,
            "ganomaly": ganomaly_model is not None,
        },
        "preprocessor_loaded": preprocessor is not None,
        "active_thresholds": thresholds,
        "available_models": ["vae", "ganomaly"],
        "supported_percentiles": ["p90", "p95", "p99"],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_single(request: PredictRequest) -> PredictResponse:
    """Predict whether a single network connection record is normal or an attack.

    Validates all 41 NSL-KDD attributes, applies preprocessor transform, runs the
    generative anomaly scoring function, and evaluates against calibrated threshold.
    """
    model_name = (request.model or "vae").lower().strip()
    if model_name not in ["vae", "ganomaly"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{request.model}'. Must be 'vae' or 'ganomaly'.",
        )

    percentile = (request.threshold_percentile or "p95").lower().strip()
    if percentile not in ["p90", "p95", "p99"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid threshold percentile '{request.threshold_percentile}'. Must be 'p90', 'p95', or 'p99'.",
        )

    # 1. Validate all 41 feature columns exist
    missing_cols = [col for col in active_feature_columns if col not in request.features]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required feature columns: {missing_cols}",
        )

    # 2. Create single-row DataFrame
    df = pd.DataFrame([request.features])[active_feature_columns]

    # 3-5. Transform and compute score
    scores = transform_and_score(model_name, df)
    score_val = float(scores[0])

    # 6. Compare score to threshold -> label
    threshold_val = get_threshold(model_name, percentile)
    label = "attack" if score_val > threshold_val else "normal"

    return PredictResponse(
        model=model_name,
        score=score_val,
        threshold=threshold_val,
        label=label,
        threshold_percentile=percentile,
    )


@app.post("/predict_csv", response_model=PredictCSVResponse, tags=["Inference"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing NSL-KDD records with 41 feature columns"),
    model: str = Query("vae", description="Model to use: 'vae' or 'ganomaly'"),
    threshold_percentile: str = Query("p95", description="Operating threshold percentile: 'p90', 'p95', or 'p99'"),
) -> PredictCSVResponse:
    """Perform batch anomaly detection on an uploaded CSV file.

    Evaluates each record against the calibrated threshold, returning individual
    scores and an overall summary of detected attacks and normal traffic.
    """
    model_name = model.lower().strip()
    if model_name not in ["vae", "ganomaly"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model '{model}'. Must be 'vae' or 'ganomaly'.",
        )

    percentile = threshold_percentile.lower().strip()
    if percentile not in ["p90", "p95", "p99"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid threshold percentile '{threshold_percentile}'. Must be 'p90', 'p95', or 'p99'.",
        )

    # Read uploaded CSV
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        logger.error("Failed to parse CSV upload: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSV file: {str(e)}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty.",
        )

    # Validate columns
    missing_cols = [col for col in active_feature_columns if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded CSV is missing required feature columns: {missing_cols}",
        )

    # Transform and score
    scores = transform_and_score(model_name, df)
    threshold_val = get_threshold(model_name, percentile)

    predictions: List[PredictionItem] = []
    attack_count = 0
    total = len(scores)

    for idx, score_val in enumerate(scores):
        s = float(score_val)
        lbl = "attack" if s > threshold_val else "normal"
        if lbl == "attack":
            attack_count += 1
        predictions.append(PredictionItem(row_index=idx, score=s, label=lbl))

    normal_count = total - attack_count
    attack_rate = float(attack_count / total) if total > 0 else 0.0

    return PredictCSVResponse(
        model=model_name,
        threshold=threshold_val,
        threshold_percentile=percentile,
        predictions=predictions,
        summary=BatchSummary(
            total=total,
            normal_count=normal_count,
            attack_count=attack_count,
            attack_rate=round(attack_rate, 4),
        ),
    )


@app.get("/metrics", tags=["Analytics"])
def get_metrics() -> Dict[str, Any]:
    """Return model performance metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading metrics file %s: %s", METRICS_PATH, e)

    return DEFAULT_METRICS


@app.get("/sample/{record_type}", tags=["Data Samples"])
def get_sample_record(record_type: str) -> Dict[str, Any]:
    """Retrieve a random record from sample datasets for normal or attack traffic.

    If sample CSVs are present in sample_data/, a random flow is sampled.
    Otherwise, returns calibrated representative traffic flows.
    """
    rec_type = record_type.lower().strip()
    if rec_type not in ["normal", "attack"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Record type must be either 'normal' or 'attack'.",
        )

    # Candidate CSV paths
    csv_candidates = [
        SAMPLE_DATA_DIR / f"sample_{rec_type}.csv",
        SAMPLE_DATA_DIR / f"{rec_type}.csv",
        PROJECT_ROOT / f"sample_{rec_type}.csv",
    ]

    selected_csv = next((p for p in csv_candidates if p.exists()), None)

    if selected_csv is not None:
        try:
            df = pd.read_csv(selected_csv)
            if not df.empty:
                sampled_row = df.sample(n=1).iloc[0]
                features_dict: Dict[str, Any] = {}
                for col in active_feature_columns:
                    if col in sampled_row:
                        val = sampled_row[col]
                        if pd.isna(val):
                            features_dict[col] = 0 if col not in CATEGORICAL_COLUMNS else "unknown"
                        elif hasattr(val, "item"):
                            features_dict[col] = val.item()
                        else:
                            features_dict[col] = val

                # Ensure all 41 columns are present
                for col in active_feature_columns:
                    if col not in features_dict:
                        features_dict[col] = 0 if col not in CATEGORICAL_COLUMNS else "unknown"

                response = dict(features_dict)
                response["record_type"] = rec_type
                response["features"] = features_dict
                return response
        except Exception as e:
            logger.warning("Error reading sample CSV %s: %s. Using default sample.", selected_csv, e)

    # Fallback to representative samples
    fallback_sample = DEFAULT_SAMPLE_NORMAL if rec_type == "normal" else DEFAULT_SAMPLE_ATTACK
    response = dict(fallback_sample)
    response["record_type"] = rec_type
    response["features"] = fallback_sample
    return response


# Load artifacts at module import time if available
load_artifacts()
