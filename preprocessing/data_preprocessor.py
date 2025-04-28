#!/usr/bin/env python3
"""
data_preprocessor.py

Builds per-transcript and per-base feature tables for DRL training, including:
  - Sequence length & dot-bracket structure
  - GWAS metrics (per-base -log10(p), SNP counts)
  - Conservation (phastCons via UCSC REST)
  - TFBS motif hits (JASPAR PWMs)
  - Regulatory feature mask (Ensembl REST)
  - Accessibility mask (Ensembl REST stub)

Saves:
  - summary.csv (per-transcript summary)
  - base_arrays/{transcript_id}.npz (arrays for each base)

Usage:
  python preprocessing/data_preprocessor.py [--limit N]

Dependencies:
  conda install viennarna biopython pandas numpy requests
"""

import os
import gzip
import re
import argparse
import glob
import time
from Bio import SeqIO, motifs
import numpy as np
import pandas as pd
import RNA
import requests

# Config paths
SCRIPT_DIR      = os.path.dirname(__file__)
FASTA_GZ        = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/sequences/lncRNA_LncBookv2.1_GRCh38.fa.gz'))
GWAS_TSV        = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/gwas/gwas_catalog_v1.0-associations_e113_r2025-04-14.tsv'))
MOTIFS_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/motifs'))
OUTPUT_SUM      = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/preprocessed/summary.csv'))
OUTPUT_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/preprocessed/base_arrays'))
ENSEMBL_REST    = "https://rest.ensembl.org/overlap/region/human"
UCSC_TRACK      = "http://api.genome.ucsc.edu/getData/track"

COORD_RX = re.compile(r'(chr[\w\d]+):(\d+)-(\d+)')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None, help='Max transcripts to process')
    return p.parse_args()

def parse_fasta(path, limit=None):
    recs = []
    with gzip.open(path, 'rt') as h:
        for i, rec in enumerate(SeqIO.parse(h, 'fasta')):
            if limit and i >= limit:
                break
            m = COORD_RX.search(rec.description)
            chr_, s, e = (m.group(1), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
            recs.append({'id': rec.id, 'seq': str(rec.seq), 'chr': chr_, 'start': s, 'end': e})
    return recs

def load_gwas(tsv):
    df = pd.read_csv(tsv, sep='\t', low_memory=False)
    if 'P-VALUE (TEXT)' in df.columns and 'P-VALUE' not in df.columns:
        df.rename(columns={'P-VALUE (TEXT)': 'P-VALUE'}, inplace=True)
    df.rename(columns=str.strip, inplace=True)
    df['chr'] = df['CHR_ID'].astype(str)
    # Extract numeric positions
    pos = df['CHR_POS'].astype(str).str.extract(r'(\d+).*?(\d+)', expand=True)
    df['pos_min'] = pd.to_numeric(pos[0], errors='coerce')
    df['pos_max'] = pd.to_numeric(pos[1], errors='coerce')
    df = df.dropna(subset=['pos_min', 'pos_max'])
    df['pos_min'] = df['pos_min'].astype(int)
    df['pos_max'] = df['pos_max'].astype(int)
    # Parse p-values
    df['p'] = pd.to_numeric(df['P-VALUE'], errors='coerce')
    df = df[df['p'] > 0]
    df['pval_log'] = -np.log10(df['p'])
    return df[['chr', 'pos_min', 'pos_max', 'pval_log']]

def fetch_conservation(chr_, start, end):
    params = {
        'genome': 'hg38', 'track': 'phastCons100way',
        'chrom': chr_, 'start': start, 'end': end, 'type': 'bedGraph'
    }
    resp = requests.get(UCSC_TRACK, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get('data', [])
    L = end - start
    arr = np.zeros(L, dtype=float)
    for rec in data:
        i0 = max(rec['start'] - start, 0)
        i1 = min(rec['end'] - start, L)
        arr[i0:i1] = rec.get('value', 0.0)
    return arr

def load_pwms(mdir):
    pwm_list = []
    for f in glob.glob(os.path.join(mdir, '*.pfm')):
        m = motifs.read(open(f), 'pfm')
        pwm_list.append(m.counts.normalize(pseudocounts=0.5).log_odds())
    return pwm_list

def scan_tfbs(seq, pwms, threshold=5.0):
    L = len(seq)
    mask = np.zeros(L, dtype=int)
    s = seq.upper().replace('T', 'U')
    for pwm in pwms:
        k = pwm.length
        for i in range(L - k + 1):
            score = sum(pwm[j].get(nt, 0) for j, nt in enumerate(s[i:i+k]))
            if score >= threshold:
                mask[i:i+k] = 1
    return mask

def fetch_mask_rest(chr_, start, end, feature):
    try:
        url = f"{ENSEMBL_REST}/{chr_}:{start}-{end}"
        resp = requests.get(url, params={'feature': feature},
                            headers={'Accept':'application/json'}, timeout=10)
        resp.raise_for_status()
        feats = resp.json()
        L = end - start
        mask = np.zeros(L, dtype=int)
        for f in feats:
            i0 = max(f['start'] - start, 0)
            i1 = min(f['end'] - start, L)
            mask[i0:i1] = 1
        return mask
    except:
        return np.zeros(end - start, dtype=int)

def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    txs = parse_fasta(FASTA_GZ, args.limit)
    print(f"[INFO] Processing {len(txs)} transcripts")

    gwas = load_gwas(GWAS_TSV)
    print(f"[INFO] Loaded {len(gwas)} GWAS records")

    pwms = load_pwms(MOTIFS_DIR)
    print(f"[INFO] Loaded {len(pwms)} PWMs")

    summary = []
    for idx, tx in enumerate(txs, 1):
        lid, seq, chr_, s, e = tx['id'], tx['seq'], tx['chr'], tx['start'], tx['end']
        L = len(seq)
        struct = RNA.fold(seq)[0]

        # GWAS masks
        gmask = np.zeros(L, dtype=int)
        gvals = np.zeros(L, dtype=float)
        if chr_ and s is not None:
            sel = gwas[(gwas['chr']==chr_) & (gwas['pos_min']<e) & (gwas['pos_max']>s)]
            for _, r in sel.iterrows():
                i0 = max(r.pos_min - s, 0)
                i1 = min(r.pos_max - s, L)
                gmask[i0:i1] = 1
                gvals[i0:i1] = r.pval_log

        cons = fetch_conservation(chr_, s, e) if chr_ else np.zeros(L)
        tfb  = scan_tfbs(seq, pwms)
        reg  = fetch_mask_rest(chr_, s, e, 'regulatory') if chr_ else np.zeros(L)
        atac = fetch_mask_rest(chr_, s, e, 'atac') if chr_ else np.zeros(L)

        # Save per-base arrays
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{lid}.npz"),
                            gmask=gmask, gvals=gvals, cons=cons,
                            tfbs=tfb, reg=reg, atac=atac)

        # Summary entry
        summary.append({
            'id': lid,
            'length': L,
            'snp_count': int(gmask.sum()),
            'snp_max_logp': float(gvals.max()),
            'mean_cons': float(cons.mean()),
            'tfbs_count': int(tfb.sum()),
            'reg_count': int(reg.sum()),
            'atac_count': int(atac.sum()),
            'structure': struct
        })

        if idx % 100 == 0 or idx == len(txs):
            print(f"[{idx}/{len(txs)}] {lid}: SNPs={int(gmask.sum())}, TFBS={int(tfb.sum())}")

        time.sleep(0.05)  # throttle REST calls

    # Write summary CSV
    pd.DataFrame(summary).to_csv(OUTPUT_SUM, index=False)
    print(f"[INFO] Summary written to {OUTPUT_SUM}")

if __name__ == '__main__':
    main()

