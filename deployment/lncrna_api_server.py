#!/usr/bin/env python3
"""
lncrna_api_server.py

FastAPI service for lncWrapper that handles:
  • Precomputed transcript IDs (loads your summary.csv and base_arrays)
  • New FASTA uploads (header >chr:start-end) for on-the-fly preprocessing

Per-base preprocessing (for FASTA uploads) includes:
  • RNA secondary structure (RNAfold via Python bindings or CLI fallback)
  • GWAS SNP mask & –log10(p) via GWAS Catalog TSV
  • Stubbed conservation, TFBS, regulatory-feature, and ATAC arrays

Then performs DRL inference and gathers:
  • Quantitative metrics: loss, interpretability, reward, parameter adjustments
  • Position-specific signals: attention weights, hotspot ranges, sequence snippets
  • SNP overlap table (positions, rsID, –log10(pp), trait)
  • GO annotations via QuickGO REST
  • eQTLs via GTEx API
  • Narrative via Ollama LLM
  • Literature snippets via DuckDuckGo

Returns a JSON object with all outputs for your frontend.

Usage:
  cd <project_root>
  pip install fastapi uvicorn biopython numpy pandas requests torch gym \
              scikit-learn scipy matplotlib langchain_community duckduckgo-search viennarna
  uvicorn deployment.lncrna_api_server:app --reload --host 0.0.0.0 --port 8000
"""

import pathlib
# Monkey-patch Path.read_text for older environments
if not hasattr(pathlib.Path, "read_text"):
    def _read_text(self, encoding="utf-8", errors=None):
        with open(self, encoding=encoding, errors=errors) as f:
            return f.read()
    pathlib.Path.read_text = _read_text

import os
import re
import io
import json
import shutil
import tempfile
import subprocess
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from Bio import SeqIO
import numpy as np
import pandas as pd
import requests
import torch

# Secondary-structure folding helper
try:
    import RNA
    def fold_seq(seq: str):
        # Returns (dot-bracket string, MFE)
        return RNA.fold(seq)
except ImportError:
    def fold_seq(seq: str):
        p = subprocess.Popen(
            ["RNAfold", "--noPS"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = p.communicate(seq.encode())
        if p.returncode != 0:
            raise RuntimeError(f"RNAfold error: {err.decode()}")
        lines = out.decode().splitlines()
        dot, mfe = lines[1].split()
        return dot, float(mfe.strip("()"))

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from environment.lncrna_explain_env import LncRNAExplainEnv
from agent.drl_agent_explain import DQNAgent

# LLM & Literature search
from langchain_community.llms import Ollama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
llm = Ollama(model="deepseek-r1:1.5b")
ddg = DuckDuckGoSearchAPIWrapper()

# Prepare GWAS Catalog DataFrame
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
gwas_tsv = os.path.join(proj_root, 'data', 'gwas',
    'gwas_catalog_v1.0-associations_e113_r2025-04-14.tsv')
gwas_df = pd.read_csv(gwas_tsv, sep='\t', low_memory=False)
pos = gwas_df['CHR_POS'].astype(str).str.extract(r'(\d+).*?(\d+)', expand=True)
gwas_df['pos_min'] = pd.to_numeric(pos[0], errors='coerce').astype('Int64')
gwas_df['pos_max'] = pd.to_numeric(pos[1], errors='coerce').astype('Int64')
gwas_df['log10p'] = -np.log10(
    pd.to_numeric(gwas_df['P-VALUE'], errors='coerce').clip(lower=1e-300)
)

# GTEx eQTL API base
GTEX_API = 'https://gtexportal.org/rest/v1/association/singleTissueEqtl'

app = FastAPI()

def parse_fasta(upload: UploadFile):
    """
    Read a FASTA file upload, parse header >chr:start-end, and return
    (chr, start, end, sequence).
    """
    data = upload.file.read().decode()
    fasta_io = io.StringIO(data)
    rec = next(SeqIO.parse(fasta_io, 'fasta'), None)
    if rec is None:
        raise HTTPException(status_code=400, detail="Invalid FASTA")
    header = rec.description
    m = re.search(r"(\w+):(\d+)-(\d+)", header)
    if not m:
        raise HTTPException(status_code=400, detail="FASTA header must be >chr:start-end")
    chr_, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    return chr_, start, end, str(rec.seq)

@app.post("/infer")
async def infer(
    transcript_id: str = Form(None),
    fasta: UploadFile = File(None)
):
    """
    Infer endpoint:
      - If transcript_id provided, use precomputed data.
      - If FASTA file provided, preprocess on-the-fly.
    """
    # Determine data source
    if transcript_id:
        summary_csv = os.path.join(proj_root, 'data', 'preprocessed', 'summary.csv')
        base_dir    = os.path.join(proj_root, 'data', 'preprocessed', 'base_arrays')
        tmp_root    = None
        chr_, start, end, seq = None, None, None, None
    elif fasta:
        # On-the-fly preprocessing
        chr_, start, end, seq = parse_fasta(fasta)
        tmp_root = tempfile.mkdtemp(prefix='lnc_')
        tmp_pre  = os.path.join(tmp_root, 'preprocessed')
        os.makedirs(os.path.join(tmp_pre,'base_arrays'), exist_ok=True)
        L = len(seq)
        # 1) RNAfold
        struct, _ = fold_seq(seq)
        # 2) GWAS masks
        gmask = np.zeros(L, dtype=int)
        gvals = np.zeros(L, dtype=float)
        sub = gwas_df[
            (gwas_df['CHR_ID']==chr_) &
            (gwas_df['pos_min'] <= end) &
            (gwas_df['pos_max'] >= start)
        ]
        for _, r in sub.iterrows():
            i0 = max(int(r.pos_min) - start, 0)
            i1 = min(int(r.pos_max) - start, L-1)
            gmask[i0:i1+1] = 1
            gvals[i0:i1+1] = r.log10p
        # 3) Stub arrays
        cons = np.zeros(L, dtype=float)
        tfbs = np.zeros(L, dtype=int)
        reg  = np.zeros(L, dtype=int)
        atac = np.zeros(L, dtype=int)
        # Write summary.csv and .npz
        df = pd.DataFrame([{
            'id': 'CUSTOM',
            'length': L,
            'structure': struct,
            'gwas_count': int(gmask.sum()),
            'mean_cons': float(cons.mean()),
            'tfbs_count': int(tfbs.sum()),
            'reg_count': int(reg.sum()),
            'atac_count': int(atac.sum())
        }]).set_index('id')
        df.to_csv(os.path.join(tmp_pre,'summary.csv'))
        np.savez_compressed(
            os.path.join(tmp_pre,'base_arrays','CUSTOM.npz'),
            gmask=gmask, gvals=gvals, cons=cons, tfbs=tfbs, reg=reg, atac=atac
        )
        summary_csv = os.path.join(tmp_pre,'summary.csv')
        base_dir    = os.path.join(tmp_pre,'base_arrays')
    else:
        raise HTTPException(status_code=400, detail="Provide transcript_id or FASTA file")

    # Initialize environment and agent
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_dir)
    agent = DQNAgent(
        env.observation_space.shape[0],
        int(np.prod(env.action_space.nvec))
    )
    ckpt = os.path.join(proj_root, 'checkpoints', 'final_model.pth')
    agent.policy_net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    agent.policy_net.eval()
    agent.eps_start = agent.eps_end = 0.0  # freeze exploration

    # Run DRL policy
    state = env.reset()
    adjustments = []
    for _ in range(500):
        flat = agent.select_action(state)
        a0, a1 = divmod(flat, 3)
        adjustments.append((a0-1, a1-1))
        state, r, done, _ = env.step([a0, a1])
        if done:
            break

    # Collect core metrics
    loss   = float(1.0 - env.interp_score)
    interp = env.interp_score
    reward = float(r)
    att    = env.att_weights.tolist()

    # Identify hotspots
    hotspots = []
    start_idx = None
    for i, w in enumerate(att):
        if w > 0.8 and start_idx is None:
            start_idx = i
        if (w <= 0.8 or i == len(att)-1) and start_idx is not None:
            end_idx = i if w <= 0.8 else i
            seq_snip = seq[start_idx:end_idx+1] if seq else None
            hotspots.append({'start': start_idx, 'end': end_idx, 'sequence': seq_snip})
            start_idx = None

    # SNP overlaps
    overlaps = []
    arr = np.load(os.path.join(base_dir, f"{env.current_id if transcript_id else 'CUSTOM'}.npz"))
    true_snp = arr['gmask']
    for h in hotspots:
        for pos_idx in range(h['start'], h['end']+1):
            if true_snp[pos_idx]:
                # Lookup in GWAS Catalog by absolute pos if FASTA, else skip trait
                if chr_ is not None:
                    abs_pos = start + pos_idx
                    hits = gwas_df[
                        (gwas_df['CHR_ID']==chr_) &
                        (gwas_df['pos_min'] <= abs_pos) &
                        (gwas_df['pos_max'] >= abs_pos)
                    ]
                else:
                    hits = pd.DataFrame([{
                        'SNPS': None,
                        'log10p': float(arr['gvals'][pos_idx]),
                        'DISEASE/TRAIT': None
                    }])
                for _, r2 in hits.iterrows():
                    overlaps.append({
                        'position': pos_idx,
                        'rsID': r2.get('SNPS'),
                        'log10p': float(r2.get('log10p', 0.0)),
                        'trait': r2.get('DISEASE/TRAIT'),
                        'attention': w
                    })

    # eQTL associations
    eqtls = []
    if overlaps:
        var_ids = ','.join({o['rsID'] for o in overlaps if o['rsID']})
        resp = requests.get(GTEX_API, params={'variantId': var_ids})
        eqtls = resp.json().get('eqtls', [])

    # GO annotations via QuickGO
    go_list = []
    url = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    params = {'geneProductId': transcript_id or 'CUSTOM', 'limit': 50}
    resp = requests.get(url, params=params, headers={'Accept':'application/json'})
    go_list = [a['goId'] for a in resp.json()
               .get('_embedded', {}).get('annotations', [])]

    # Generate narrative with LLM
    prompt = (
        f"Transcript {env.current_id} (len={len(att)}), "
        f"Loss={loss:.3f}, Interp={interp:.3f}, Reward={reward:.3f}\n"
        f"Hotspots: {[(h['start'],h['end']) for h in hotspots]}\n"
        f"SNPs: {[o['rsID'] for o in overlaps]}\n"
        f"GO: {go_list[:5]}... {len(go_list)} total\n"
        f"eQTLs: {[e.get('geneId') for e in eqtls][:3]}...\n"
        "Summarize regulatory insights."
    )
    narrative = llm.invoke(prompt)

    # Literature snippets via DuckDuckGo
    snippets = []
    for o in overlaps[:3]:
        snippets += ddg.run(f"lncRNA hotspot {o['rsID']} {o['trait']} function")[:2]

    result = {
        'final_loss': loss,
        'interpretability': interp,
        'composite_reward': reward,
        'parameter_adjustments': adjustments,
        'attention_weights': att,
        'hotspots': hotspots,
        'snp_overlap': overlaps,
        'go_terms': go_list,
        'eqtls': eqtls,
        'narrative': narrative,
        'literature_snippets': snippets
    }

    # Cleanup temporary files if any
    if fasta:
        shutil.rmtree(tmp_root)

    return result
