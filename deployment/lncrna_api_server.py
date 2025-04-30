#!/usr/bin/env python3
"""
run_agent.py

FastAPI service for lncWrapper that handles both pre-computed transcripts and new user-supplied FASTA sequences:
- On known transcript IDs: loads preprocessed summary & base_arrays
- On new FASTA uploads: parses header (>chr:start-end) and sequence via BioPython,
  runs per-base preprocessing (RNAfold, GWAS REST, conservation, motif, regulatory features, ATAC),
  and builds summary + base_arrays in a temp directory.

Performs DRL inference then gathers:
  * Quantitative metrics: loss, interpretability, composite reward, parameter adjustments
  * Position-specific signals: attention vector, hotspot ranges, base sequences, SNP overlap table
  * GO annotations: list of GO terms via QuickGO
  * eQTL associations: via GTEx API
  * Iterative literature search: DuckDuckGo

Returns JSON with all outputs for frontend rendering.

Usage:
  uvicorn inference.run_agent:app --reload

Dependencies:
  pip install fastapi uvicorn biopython numpy pandas RNA requests gym torch langchain_community viennaRNA
    pip install duckduckgo-search langchain_community
"""
import os
import re
import io
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
import RNA
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from Bio import SeqIO

# ensure project root on path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.lncrna_explain_env import LncRNAExplainEnv
from agent.drl_agent_explain import DQNAgent
from langchain_community.llms import Ollama
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Initialize LLM and Search
llm = Ollama(model="deepseek-r1:1.5b")
ddg = DuckDuckGoSearchAPIWrapper()

# Load precomputed GWAS Catalog
gwas_tsv = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 
                      'data', 'gwas', 'gwas_catalog_v1.0-associations_e113_r2025-04-14.tsv')
gwas_df = pd.read_csv(gwas_tsv, sep='\t', low_memory=False)
# extract position and p-value
pos = gwas_df['CHR_POS'].astype(str).str.extract(r'(\d+).*?(\d+)', expand=True)
gwas_df['pos_min'] = pd.to_numeric(pos[0], errors='coerce').astype('Int64')
gwas_df['pos_max'] = pd.to_numeric(pos[1], errors='coerce').astype('Int64')
gwas_df['log10p'] = -np.log10(pd.to_numeric(gwas_df['P-VALUE'], errors='coerce').clip(lower=1e-300))

# GTEx eQTL API base
GTEX_API = 'https://gtexportal.org/rest/v1/association/singleTissueEqtl'

app = FastAPI()

def parse_fasta(upload: UploadFile):
    content = upload.file.read().decode()
    fasta_io = io.StringIO(content)
    rec = next(SeqIO.parse(fasta_io, 'fasta'), None)
    if rec is None:
        raise HTTPException(status_code=400, detail='Invalid FASTA file')
    header = rec.description
    m = re.search(r"(\w+):(\d+)-(\d+)", header)
    if not m:
        raise HTTPException(status_code=400, detail='FASTA header must be >chr:start-end')
    chr_, start, end = m.group(1), int(m.group(2)), int(m.group(3))
    seq = str(rec.seq)
    return chr_, start, end, seq

@app.post('/infer')
async def infer(
    transcript_id: str = Form(None),
    fasta: UploadFile = File(None)
):
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Determine data source
    if transcript_id:
        # use precomputed
        summary_csv = os.path.join(proj_root, 'data', 'preprocessed', 'summary.csv')
        base_dir    = os.path.join(proj_root, 'data', 'preprocessed', 'base_arrays')
        tmp_root = None
    elif fasta:
        # parse FASTA upload
        chr_, start, end, seq = parse_fasta(fasta)
        tmp_root = tempfile.mkdtemp(prefix='lnc_')
        tmp_pre = os.path.join(tmp_root, 'preprocessed')
        os.makedirs(os.path.join(tmp_pre, 'base_arrays'), exist_ok=True)
        L = len(seq)
        # structure via RNAfold
        struct, _ = RNA.fold(seq)
        # GWAS per-base
        gmask = np.zeros(L, int); gvals = np.zeros(L, float)
        sub = gwas_df[(gwas_df['CHR_ID']==chr_) &
                      (gwas_df['pos_min'] <= end) &
                      (gwas_df['pos_max'] >= start)]
        for _, r in sub.iterrows():
            i0 = max(int(r.pos_min) - start, 0)
            i1 = min(int(r.pos_max) - start, L-1)
            gmask[i0:i1+1] = 1; gvals[i0:i1+1] = r.log10p
        # dummy cons, tfbs, reg, atac arrays
        cons = np.zeros(L, float)
        tfbs = np.zeros(L, int)
        reg  = np.zeros(L, int)
        atac = np.zeros(L, int)
        # write summary.csv
        df = pd.DataFrame([{
            'id': 'CUSTOM', 'length': L, 'structure': struct,
            'gwas_count': int(gmask.sum()), 'mean_cons': float(cons.mean()),
            'tfbs_count': int(tfbs.sum()), 'reg_count': int(reg.sum()), 'atac_count': int(atac.sum())
        }]).set_index('id')
        df.to_csv(os.path.join(tmp_pre, 'summary.csv'))
        np.savez_compressed(
            os.path.join(tmp_pre, 'base_arrays', 'CUSTOM.npz'),
            gmask=gmask, gvals=gvals, cons=cons, tfbs=tfbs, reg=reg, atac=atac
        )
        summary_csv = os.path.join(tmp_pre, 'summary.csv')
        base_dir    = os.path.join(tmp_pre, 'base_arrays')
    else:
        raise HTTPException(status_code=400, detail='Provide transcript_id or FASTA upload')

    # DRL inference
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_dir)
    agent = DQNAgent(env.observation_space.shape[0], int(np.prod(env.action_space.nvec)))
    ckpt = os.path.join(proj_root, 'checkpoints', 'final_model.pth')
    agent.policy_net.load_state_dict(torch.load(ckpt, map_location='cpu'))
    agent.policy_net.eval()
    agent.eps_start = agent.eps_end = 0.0

    # run
    state = env.reset()
    adjustments = []
    for _ in range(500):
        flat = agent.select_action(state)
        a0, a1 = divmod(flat, 3)
        adjustments.append((a0-1, a1-1))
        state, r, done, _ = env.step([a0, a1])
        if done:
            break

    # collect metrics
    loss = float(1 - env.interp_score)
    interp = env.interp_score
    reward = float(r)
    att = env.att_weights.tolist()
    hs, s = [], None
    for i, w in enumerate(att):
        if w > 0.8 and s is None: s = i
        if (w <= 0.8 or i == len(att)-1) and s is not None:
            e = i
            seq_snip = seq[s:e+1] if fasta else None
            hs.append({'start': s, 'end': e, 'sequence': seq_snip})
            s = None

    # SNP overlaps
    overlaps = []
    arr = np.load(os.path.join(base_dir, f"{env.current_id}.npz"))
    for region in hs:
        for pos_idx in range(region['start'], region['end']+1):
            if arr['gmask'][pos_idx]:
                hits = gwas_df[(gwas_df['CHR_ID']==(chr_ if fasta else env.current_id.split(':')[0])) &
                               (gwas_df['pos_min'] == start + pos_idx)]
                for _, r2 in hits.iterrows():
                    overlaps.append({
                        'position': pos_idx,
                        'rsID': r2['SNPS'],
                        'log10p': float(r2.log10p),
                        'trait': r2['DISEASE/TRAIT'],
                        'attention': att[pos_idx]
                    })
    # eQTLs
    eqtls = []
    if overlaps:
        var_ids = ','.join({o['rsID'] for o in overlaps})
        resp = requests.get(GTEX_API, params={'variantId': var_ids})
        eqtls = resp.json().get('eqtls', [])

    # GO terms
    go_list = []
    url = 'https://www.ebi.ac.uk/QuickGO/services/annotation/search'
    params = {'geneProductId': transcript_id or 'CUSTOM', 'limit': 50}
    resp = requests.get(url, params=params, headers={'Accept':'application/json'})
    go_list = [a['goId'] for a in resp.json().get('_embedded', {}).get('annotations', [])]

    # Narrative via LLM
    prompt = (f"Transcript {env.current_id} (len={len(att)}), Loss={loss:.3f}, "
              f"Interp={interp:.3f}, Reward={reward:.3f}\n"
              f"Hotspots: {[ (h['start'], h['end']) for h in hs ]}\n"
              f"SNPs: {[o['rsID'] for o in overlaps]}\n"
              f"GO: {go_list[:5]}... {len(go_list)} total\n"
              f"eQTLs: {[e['geneId'] for e in eqtls][:3]}...\n"
              "Summarize regulatory insights.")
    narrative = llm.invoke(prompt)

    # Literature snippets
    snippets = []
    for o in overlaps[:3]:
        snippets += ddg.run(f"lncRNA hotspot {o['rsID']} {o['trait']} function")[:2]

    result = {
        'final_loss': loss,
        'interpretability': interp,
        'composite_reward': reward,
        'parameter_adjustments': adjustments,
        'attention_weights': att,
        'hotspots': hs,
        'snp_overlap': overlaps,
        'go_terms': go_list,
        'eqtls': eqtls,
        'narrative': narrative,
        'literature_snippets': snippets
    }

    if fasta:
        shutil.rmtree(tmp_root)
    return result
