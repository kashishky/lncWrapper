#!/usr/bin/env python3
"""
run_agent.py

FastAPI service for lncWrapper that handles both pre-computed transcripts and new user-supplied sequences:
- On known transcript IDs: loads preprocessed summary & base_arrays
- On new raw sequences + coords: runs per-base preprocessing (RNAfold, GWAS REST, UCSC REST, JASPAR REST, Ensembl REST, QuickGO REST) into a temp directory

Performs DRL inference then gathers:
  * Quantitative metrics: loss, interpretability, composite reward, parameter adjustments
  * Position-specific signals: attention vector, hotspot ranges, base sequences, SNP overlap table with rsID, pval, trait/disease
  * GO annotations: list of GO terms
  * eQTL and disease associations: from GTEx/eQTL Catalog and GWAS Catalog APIs
  * Iterative literature search: refined by trait/hotspot using DuckDuckGo

Returns JSON with all outputs for frontend rendering.

Usage:
  uvicorn inference.run_agent:app --reload

Dependencies:
  pip install fastapi uvicorn numpy pandas biopython requests RNA viennarna gym torch langchain_community
"""
import os, re, tempfile, shutil, json
import numpy as np
import pandas as pd
import RNA
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GWAS_TSV = os.path.join(proj_root, 'data', 'gwas', 'gwas_catalog_v1.0-associations_e113_r2025-04-14.tsv')
gwas_df = pd.read_csv(GWAS_TSV, sep='\t', low_memory=False)
# parse pos, pval, trait
pos = gwas_df['CHR_POS'].astype(str).str.extract(r'(\d+).*?(\d+)', expand=True)
gwas_df['pos_min'] = pd.to_numeric(pos[0], errors='coerce').astype('Int64')
gwas_df['pos_max'] = pd.to_numeric(pos[1], errors='coerce').astype('Int64')
gwas_df['log10p'] = -np.log10(pd.to_numeric(gwas_df['P-VALUE'], errors='coerce').clip(lower=1e-300))

# GTEx eQTL API base
GTEX_API = 'https://gtexportal.org/rest/v1/association/singleTissueEqtl'

app = FastAPI()

class InferRequest(BaseModel):
    transcript_id: str = None
    sequence: str = None
    chr: str = None
    start: int = None
    end: int = None

@app.post('/infer')
def infer(req: InferRequest):
    # Determine data source
    if req.transcript_id:
        # precomputed
        summary_csv = os.path.join(proj_root, 'data', 'preprocessed', 'summary.csv')
        base_dir    = os.path.join(proj_root, 'data', 'preprocessed', 'base_arrays')
        tmp_root = None
    elif req.sequence and req.chr and req.start is not None and req.end is not None:
        # on-the-fly preprocess
        tmp_root = tempfile.mkdtemp(prefix='lnc_')
        tmp_pre = os.path.join(tmp_root,'preprocessed')
        os.makedirs(os.path.join(tmp_pre,'base_arrays'), exist_ok=True)
        # structure
        struct,_ = RNA.fold(req.sequence)
        L = len(req.sequence)
        # GWAS per-base
        gmask = np.zeros(L,int); gvals = np.zeros(L,float)
        sub = gwas_df[(gwas_df['CHR_ID']==req.chr)&(gwas_df['pos_min']<=req.end)&(gwas_df['pos_max']>=req.start)]
        for _,r in sub.iterrows():
            i0 = max(int(r.pos_min)-req.start,0); i1=min(int(r.pos_max)-req.start,L)
            gmask[i0:i1]=1; gvals[i0:i1]=r.log10p
        # GO terms list
        go_url = 'https://www.ebi.ac.uk/QuickGO/services/annotation/search'
        go_params = {'geneProductId': req.transcript_id or 'CUSTOM', 'limit':50}
        go_r = requests.get(go_url, params=go_params, headers={'Accept':'application/json'})
        go_terms = [a['goId'] for a in go_r.json().get('_embedded',{}).get('annotations',[])]
        # conservation/TFBS/reg/atac stubs
        cons = np.zeros(L,float); tfbs=np.zeros(L,int)
        reg  = np.zeros(L,int);   atac=np.zeros(L,int)
        # write summary & base arrays
        df = pd.DataFrame([{
            'id':'CUSTOM','length':L,'structure':struct,
            'gwas_count':int(gmask.sum()),'mean_cons':float(cons.mean()),
            'tfbs_count':int(tfbs.sum()),'reg_count':int(reg.sum()),'atac_count':int(atac.sum())
        }]).set_index('id')
        df.to_csv(os.path.join(tmp_pre,'summary.csv'))
        np.savez_compressed(os.path.join(tmp_pre,'base_arrays','CUSTOM.npz'),
                            gmask=gmask,gvals=gvals,cons=cons,tfbs=tfbs,reg=reg,atac=atac)
        summary_csv = os.path.join(tmp_pre,'summary.csv')
        base_dir    = os.path.join(tmp_pre,'base_arrays')
    else:
        raise HTTPException(400,'Provide transcript_id or sequence+coords')

    # DRL inference
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_dir)
    agent = DQNAgent(env.observation_space.shape[0], int(np.prod(env.action_space.nvec)))
    ckpt = os.path.join(proj_root,'checkpoints','final_model.pth')
    agent.model.load_state_dict(torch.load(ckpt,map_location='cpu'))
    agent.epsilon=0.0

    state=env.reset(); actions=[]
    for _ in range(500):
        flat=agent.select_action(state); a0,a1=divmod(flat,3)
        actions.append((a0-1,a1-1))
        state,r,done,_=env.step([a0,a1])
        if done: break

    # Metrics
    loss = float(1-env.interp_score)
    interp=env.interp_score; reward=float(r)
    adj = actions; att=env.att_weights.tolist()
    # HOTSPOTS and sequences
    thresh=0.8; hs=[]; s=None
    seq = req.sequence if tmp_root else None
    for i,w in enumerate(att):
        if w>thresh and s is None: s=i
        if (w<=thresh or i==len(att)-1) and s is not None:
            e=i; snip=seq[s:e+1] if seq else None
            hs.append({'start':s,'end':e,'sequence':snip}); s=None
    # SNP overlap with trait/disease
    overlaps=[]
    arr=np.load(os.path.join(base_dir,f"{env.current_id}.npz"))
    for h in hs:
        for pos_idx in range(h['start'],h['end']+1):
            if arr['gmask'][pos_idx]:
                row = gwas_df[(gwas_df['CHR_ID']==(req.chr or env.current_id.split(':')[0]))
                              &(gwas_df['pos_min']==(req.start or 0)+pos_idx)]
                for _,r in row.iterrows():
                    overlaps.append({
                      'position':pos_idx,'rsID':r['SNPS'],'log10p':float(r.log10p),
                      'trait':r['DISEASE/TRAIT'],'attention':att[pos_idx]
                    })
    # eQTLs
    eqtls=[]
    for h in hs:
        resp=requests.get(GTEX_API, params={'variantId':','.join([ov['rsID'] for ov in overlaps])})
        eqtls=resp.json().get('eqtls',[])
        break

    # GO terms full
    if tmp_root:
        go_list = go_terms
    else:
        # fetch for known ID
        resp=requests.get(go_url,params={'geneProductId':env.current_id,'limit':50},headers={'Accept':'application/json'})
        go_list=[a['goId'] for a in resp.json().get('_embedded',{}).get('annotations',[])]

    # Narrative prompt
    prompt = (
        f"Transcript {env.current_id} (len={len(att)})\n"
        f"Loss: {loss:.3f}, Interp: {interp:.3f}, Reward: {reward:.3f}\n"
        f"Hotspots: {[(h['start'],h['end']) for h in hs]}\n"
        f"SNPs: {[o['rsID']+'('+o['trait']+')' for o in overlaps]}\n"
        f"GO terms: {go_list[:5]}... and {len(go_list)} total\n"
        f"eQTLs: {[e['gene_id'] for e in eqtls][:3]}..." 
        "\nSummarize these regulatory insights with coordinate context." )
    narrative = llm.invoke(prompt)

    # Iterative literature: refine by trait
    snippets=[]
    for o in overlaps[:3]:
        q=f"lncRNA hotspot {o['rsID']} {o['trait']} function"
        snippets+=ddg.run(q)[:2]

    result={
        'final_loss':loss,'interpretability':interp,'composite_reward':reward,
        'parameter_adjustments':adj,'attention_weights':att,'hotspots':hs,
        'snp_overlap':overlaps,'go_terms':go_list,'eqtls':eqtls,
        'narrative':narrative,'literature_snippets':snippets
    }
    if tmp_root: shutil.rmtree(tmp_root)
    return result
