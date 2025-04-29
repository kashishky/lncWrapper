# Retry and Timeout
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

try:
    response = session.get(api_url, timeout=10)
    response.raise_for_status()
    data = response.json()
except requests.RequestException as e:
    logger.error(f"API call failed: {e}")


# Cache Data Loads
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=None)
def load_summary():
    return pd.read_csv("preprocessing/summary.csv")

summary_df = load_summary()

# Load Model/Data Once
import torch
from model import DQNModel

model = DQNModel(...)
model.load_state_dict(torch.load("models/dqn_model.pth"))
model.eval()

@app.post("/predict")
async def predict(request: Request):
    ...


# deployment/lncrna_api_server.py

import torch
from model import DQNModel
from fastapi import FastAPI, Request
import asyncio
import logging

# ----------------------------------------------------------------
# 1. Structured logging setup
# Configure logging to include timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("lncrna_api")
logger.info("FastAPI application has started successfully")

# ----------------------------------------------------------------
# 2. Load model once at module import (global scope)
# This avoids reloading the model on every request
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel(state_dim=5, action_dim=9)
model.load_state_dict(torch.load("models/dqn_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------------------------------------------
# 3. FastAPI app initialization
app = FastAPI()

# ----------------------------------------------------------------
# 4. Predict endpoint using the preloaded model
@app.post("/predict")
async def predict(request: Request):
    """
    Accepts JSON with fields:
      - loss: float
      - interpretability: float
      - attn_mean: float
      - attn_std: float
      - gwas_score: float

    Returns a JSON with recommended adjustments.
    """
    payload = await request.json()
    state_tensor = torch.tensor(
        [[
            payload["loss"],
            payload["interpretability"],
            payload["attn_mean"],
            payload["attn_std"],
            payload["gwas_score"]
        ]],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        q_values = model(state_tensor)

    best_action = int(q_values.argmax().item())
    att_scaling = best_action // 3
    bias_adjustment = best_action % 3

    return {
        "attention_scaling_adjustment": att_scaling,
        "bias_adjustment": bias_adjustment
    }

# ----------------------------------------------------------------
# 5. Example heavy compute endpoint (async with thread pool)
def heavy_compute(x):
    """
    Simulate a CPU-intensive or blocking operation,
    e.g., complex model inference or data processing.
    """
    # ... perform computation ...
    return {"processed": x * 2}

@app.post("/compute")
async def compute_endpoint(data: dict):
    """
    Runs heavy_compute in a thread pool to avoid blocking.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, heavy_compute, data["input"])
    return result

# ----------------------------------------------------------------
# 6. Example error handling logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error while processing request: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
