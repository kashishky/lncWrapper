#!/usr/bin/env python3
"""
evaluate_agent_lncrna_explain.py

Evaluate a trained DRL agent on a held-out test set of lncRNA transcripts,
comparing its fixed-policy performance against a uniform-attention baseline.
- Computes metrics for each transcript:

Metrics computed per transcript:
  - baseline_loss       = 1 − interp_score(initial uniform attention)
  - final_loss          = 1 − interp_score(after DRL tuning)
  - loss_improvement    = baseline_loss − final_loss
  - baseline_interp     = interp_score(initial)
  - final_interp        = interp_score(after)
  - interp_improvement  = final_interp − baseline_interp
  - baseline_auc        = AUROC(true SNP mask, uniform attention)
  - final_auc           = AUROC(true SNP mask, tuned attention)
  - auc_improvement     = final_auc − baseline_auc
  - spearman_r, spearman_p between attention weights and −log10(p)
  - go_count            = total GO annotations via QuickGO REST
  - baseline_overlap    = count of SNPs in hotspots (attention > 0.8) before
  - final_overlap       = count of SNPs in hotspots after
  - overlap_improvement = final_overlap − baseline_overlap

Outputs:
  - evaluation_metrics_test.json       (detailed per-transcript records)
  - evaluation_improvements_test.png   (histograms of improvements)

Usage:
  cd <project_root>
  pip install torch numpy pandas scikit-learn scipy matplotlib requests gym
  python -m evaluation.evaluate_agent_lncrna_explain
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import requests

# allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.lncrna_explain_env import LncRNAExplainEnv
from agent.drl_agent_explain import DQNAgent


def evaluate(
    checkpoint_path: str,
    summary_csv: str = "data/preprocessed_test/summary.csv",
    base_array_dir: str = "data/preprocessed_test/base_arrays",
    max_steps: int = 500,
    output_json: str = "evaluation_metrics_test.json"
):
    # Resolve paths
    proj_root      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    summary_csv    = os.path.join(proj_root, summary_csv)
    base_array_dir = os.path.join(proj_root, base_array_dir)

    # Sanity checks
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"summary.csv not found at {summary_csv}")
    if not os.path.isdir(base_array_dir):
        raise FileNotFoundError(f"base_arrays dir not found at {base_array_dir}")

    # Load env and agent
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_array_dir)
    state_dim  = env.observation_space.shape[0]
    action_dim = int(np.prod(env.action_space.nvec))

    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    agent.policy_net.eval()
    # freeze policy
    agent.eps_start = agent.eps_end = 0.0

    records = []
    for tid in env.ids:
        # setup baseline
        length = int(env.summary.at[tid, 'length'])
        env.current_id   = tid
        env.interp_score = 0.2
        env.att_scaling  = 1.0
        env.bias         = 0.0
        uniform_att = np.ones(length, dtype=np.float32) / length
        env.att_weights = uniform_att.copy()

        # load arrays
        arr = np.load(os.path.join(base_array_dir, f"{tid}.npz"))
        true_snp = arr['gmask']
        true_vals= arr['gvals']

        # skip if no SNPs
        if true_snp.sum() == 0:
            continue

        # baseline metrics
        baseline_loss   = float(np.clip(1.0 - env.interp_score, 0.0, 1.0))
        baseline_interp = float(env.interp_score)
        baseline_auc    = roc_auc_score(true_snp, uniform_att)
        baseline_overlap = int(np.sum((uniform_att > 0.8) & (true_snp > 0)))

        # run fixed-policy
        state = env._get_state()
        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            action = agent.select_action(state)
            a0, a1 = divmod(action, 3)
            state, reward, done, _ = env.step([a0, a1])
            total_reward += reward
            steps += 1

        # final metrics
        final_interp = float(env.interp_score)
        final_loss   = float(np.clip(1.0 - final_interp, 0.0, 1.0))
        final_auc    = roc_auc_score(true_snp, env.att_weights)
        final_overlap= int(np.sum((env.att_weights > 0.8) & (true_snp > 0)))

        # spearman
        try:
            if np.all(env.att_weights == env.att_weights[0]) or np.all(true_vals == true_vals[0]):
                raise ValueError("constant input")
            corr_r, corr_p = spearmanr(env.att_weights, true_vals)
        except Exception:
            corr_r, corr_p = None, None

        # GO count
        go_count = 0
        try:
            url = (f"https://www.ebi.ac.uk/QuickGO/services/annotation/search"
                   f"?geneProductId={tid}&limit=0")
            resp = requests.get(url, headers={"Accept":"application/json"})
            resp.raise_for_status()
            go_count = int(resp.json().get('pageInfo', {}).get('total', 0))
        except:
            pass

        # improvements
        records.append({
            'transcript_id': tid,
            'baseline_loss': baseline_loss,
            'final_loss': final_loss,
            'loss_improvement': baseline_loss - final_loss,
            'baseline_interp': baseline_interp,
            'final_interp': final_interp,
            'interp_improvement': final_interp - baseline_interp,
            'baseline_auc': baseline_auc,
            'final_auc': final_auc,
            'auc_improvement': final_auc - baseline_auc,
            'spearman_r': corr_r,
            'spearman_p': corr_p,
            'go_count': go_count,
            'baseline_overlap': baseline_overlap,
            'final_overlap': final_overlap,
            'overlap_improvement': final_overlap - baseline_overlap,
            'total_reward': total_reward,
            'steps': steps
        })

    # write JSON
    out_path = os.path.join(proj_root, output_json)
    with open(out_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved evaluation metrics to {out_path}")

    # summarize stats
    df = pd.DataFrame(records)
    stats = {
        'avg_baseline_loss': df['baseline_loss'].mean(),
        'avg_final_loss': df['final_loss'].mean(),
        'avg_baseline_interp': df['baseline_interp'].mean(),
        'avg_final_interp': df['final_interp'].mean(),
        'avg_baseline_auc': df['baseline_auc'].mean(),
        'avg_final_auc': df['final_auc'].mean(),
        'avg_spearman_r': df['spearman_r'].dropna().mean(),
        'avg_go_count': df['go_count'].mean()
    }
    print("\nSummary statistics:")
    for k,v in stats.items(): print(f"  {k}: {v:.3f}")

    # plot histograms
    plt.figure(figsize=(10,8))
    for i, metric in enumerate(['loss_improvement','interp_improvement','auc_improvement','overlap_improvement'],1):
        plt.subplot(2,2,i)
        df[metric].hist(bins=20)
        plt.title(metric.replace('_',' ').title())
    plt.tight_layout()
    plt.savefig(os.path.join(proj_root,'evaluation_improvements_test.png'))
    print("Saved evaluation_improvements_test.png")

if __name__ == '__main__':
    checkpoint = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'policy_net_checkpoint.pth')
    evaluate(checkpoint)
