#!/usr/bin/env python3
"""
train_agent_lncrna_explain.py

DRL training for lncRNAExplainEnv:
  - Absolute path resolution
  - GO‐annotation count via QuickGO REST
  - Multi‐metric tracking & plots (reward, interpretability, GWAS overlap)

Usage:
  cd <project_root>
  python training/train_agent_lncrna_explain.py

Dependencies:
  pip install torch numpy matplotlib gym pandas viennarna biopython
"""
import os
# allow duplicate OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np
import matplotlib.pyplot as plt
# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.lncrna_explain_env import LncRNAExplainEnv
from agent.drl_agent_explain import DQNAgent


def train(
    num_episodes: int = 200,
    max_steps_per_episode: int = 500,
    target_update_freq: int = 10,
    summary_csv: str = "data/preprocessed/summary.csv",
    base_array_dir: str = "data/preprocessed/base_arrays"
):
    # === Build absolute paths ===
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    summary_csv    = os.path.join(proj_root, summary_csv)
    base_array_dir = os.path.join(proj_root, base_array_dir)

    # Validate paths
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"summary.csv not found at: {summary_csv}")
    if not os.path.isdir(base_array_dir):
        raise FileNotFoundError(f"base_arrays dir not found: {base_array_dir}")

    # Create environment & agent
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_array_dir)
    state_dim  = env.observation_space.shape[0]
    action_dim = int(np.prod(env.action_space.nvec))
    agent = DQNAgent(state_dim, action_dim)

    # Tracking metrics
    rewards, interps, overlaps, lengths = [], [], [], []

    for ep in range(1, num_episodes+1):
        state = env.reset()
        total_reward = 0.0
        interp_hist, overlap_hist = [], []

        for t in range(1, max_steps_per_episode+1):
            # select and apply action
            flat_a = agent.select_action(state)
            a0, a1 = divmod(flat_a, 3)
            nxt_s, r, done, _ = env.step([a0, a1])
            agent.buffer.push(state, flat_a, r, nxt_s, done)
            agent.optimize()

            # metrics
            total_reward += r
            interp_hist.append(env.interp_score)
            arr = np.load(os.path.join(base_array_dir, f"{env.current_id}.npz"))
            overlap_frac = ((env.att_weights > 0.8) & (arr['gmask']>0)).sum() / max(1, int(arr['gmask'].sum()))
            overlap_hist.append(overlap_frac)

            state = nxt_s
            if done:
                break

        # update target network
        if ep % target_update_freq == 0:
            agent.update_target()

        # record episode stats
        rewards.append(total_reward)
        interps.append(np.mean(interp_hist))
        overlaps.append(np.mean(overlap_hist))
        lengths.append(t)

        print(f"Episode {ep:3d} | Reward={total_reward:.2f} | "
              f"Interp={interps[-1]:.3f} | Overlap={overlaps[-1]:.3f} | Steps={t}")

    # Plot metrics
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(rewards, label='Total Reward')
    ax.plot(interps, label='Mean Interpretability')
    ax.plot(overlaps, label='Mean GWAS Overlap')
    ax.set(xlabel='Episode', ylabel='Value', title='Training Metrics')
    ax.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print('Saved training_metrics.png')

    # Histogram of episode lengths
    plt.figure()
    plt.hist(lengths, bins=20)
    plt.xlabel('Steps per Episode')
    plt.ylabel('Frequency')
    plt.title('Episode Lengths')
    plt.savefig('episode_lengths_hist.png')
    print('Saved episode_lengths_hist.png')

    return rewards, interps, overlaps


if __name__ == '__main__':
    train()
