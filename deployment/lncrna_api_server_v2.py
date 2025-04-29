# deployment/lncrna_api_server_v2.py

import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from preprocessing.load_preprocessed import load_summary, load_base_array

app = FastAPI(title="lncRNA DRL API v2")

class PredictRequest(BaseModel):
    transcript_id: str

class PredictResponse(BaseModel):
    transcript_id: str
    predicted_value: float
    gwas_score: float

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    # load summary table
    try:
        df = load_summary()
    except Exception as e:
        raise HTTPException(500, f"Failed to load summary: {e}")

    # find record
    recs = df[df["id"] == req.transcript_id]
    if recs.empty:
        raise HTTPException(404, f"Transcript ID '{req.transcript_id}' not found")
    row = recs.iloc[0]
    gwas_score = float(row.get("gwas_sum_logp", 0.0))

    # load base array
    try:
        base = load_base_array(req.transcript_id)
    except FileNotFoundError:
        raise HTTPException(404, f"No base array for '{req.transcript_id}'")
    except Exception as e:
        raise HTTPException(500, f"Error loading base array: {e}")

    # compute a simple prediction (sum of a ‘gmask’ array if present)
    try:
        if isinstance(base, dict) or hasattr(base, "files"):
            arr = base["gmask"] if "gmask" in base.files else sum(base[f] for f in base.files)
        else:
            arr = base
        predicted_value = float(arr.sum())
    except Exception:
        predicted_value = 0.0

    return PredictResponse(
        transcript_id=req.transcript_id,
        predicted_value=predicted_value,
        gwas_score=gwas_score
    )
