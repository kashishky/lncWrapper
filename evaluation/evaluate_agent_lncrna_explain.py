import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.lncrna_explain_env import LncRNAExplainEnv
from agent.drl_agent_explain import DQN, DQNAgent


def evaluate(
    checkpoint_path: str,
    summary_csv: str = "lncWrapper/data/preprocessed/summary.csv",
    base_array_dir: str = "lncWrapper/data/preprocessed/base_arrays",
    max_steps: int = 500,
    output_json: str = "evaluation_metrics.json"
):
    # Initialize environment and agent
    env = LncRNAExplainEnv(summary_csv=summary_csv, base_array_dir=base_array_dir)
    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.nvec.prod())

    # Load policy network
    agent = DQNAgent(state_dim, action_dim)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=agent.device))
    agent.policy_net.eval()

    # Store evaluation results
    records = []
    for tid in env.ids:
        # Manually reset to specific transcript
        env.current_id = tid
        length = int(env.summary.at[tid, 'length'])
        env.interp_score = 0.2
        env.att_scaling = 1.0
        env.bias = 0.0
        env.att_weights = np.ones(length, dtype=np.float32) / length

        state = env._get_state()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            flat_action = agent.select_action(state)
            a0, a1 = divmod(flat_action, 3)
            state, reward, done, _ = env.step([a0, a1])
            total_reward += reward
            steps += 1

        # After run, compute final metrics
        loss_norm = np.clip(1.0 - env.interp_score, 0.0, 1.0)
        arr = np.load(os.path.join(base_array_dir, f"{tid}.npz"))
        hotspots = np.where(env.att_weights > 0.8)[0]
        # extract contiguous regions
        regions = []
        if hotspots.size > 0:
            start = hotspots[0]
            for i in range(1, hotspots.size):
                if hotspots[i] != hotspots[i-1] + 1:
                    regions.append({'start': int(start), 'end': int(hotspots[i-1])})
                    start = hotspots[i]
            regions.append({'start': int(start), 'end': int(hotspots[-1])})
        overlap_snps = [
            {"position": int(pos), "rsID": "rsXXXXX",  # synthetic
             "-log10(p)": float(arr['gvals'][pos]), "attention": float(env.att_weights[pos])}
            for pos in np.where((env.att_weights > 0.8) & (arr['gmask']>0))[0]
        ]

        records.append({
            'transcript_id': tid,
            'total_reward': total_reward,
            'steps': steps,
            'final_interp_score': float(env.interp_score),
            'final_loss_norm': float(loss_norm),
            'hotspot_regions': regions,
            'gwas_attention_overlap': overlap_snps
        })

    # Save metrics
    with open(output_json, 'w') as f:
        json.dump(records, f, indent=2)

    # Create summary DataFrame
    df = pd.DataFrame(records)
    df[['total_reward', 'final_interp_score', 'final_loss_norm']].hist(bins=20)
    plt.tight_layout()
    plt.savefig('evaluation_histograms.png')
    print(f"Evaluation complete. Metrics saved to {output_json} and histograms to evaluation_histograms.png")


if __name__ == '__main__':
    # Path to the trained policy network weights (.pt or .pth)
    checkpoint = 'policy_net_checkpoint.pth'
    summary = os.path.join('lncWrapper', 'data', 'preprocessed', 'summary.csv')
    base_dir = os.path.join('lncWrapper', 'data', 'preprocessed', 'base_arrays')
    evaluate(checkpoint, summary_csv=summary, base_array_dir=base_dir)
