import os
import numpy as np
import pandas as pd
import gym
from gym import spaces
import random

class LncRNAExplainEnv(gym.Env):
    """Gym environment for tuning an lncRNA predictor using DRL."""

    def __init__(self,
                 summary_csv='data/preprocessed/summary.csv',
                 base_array_dir='data/preprocessed/base_arrays',
                 interp_init=0.2):
        super().__init__()

        # === ensure we build absolute paths from the project root ===
        proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        summary_csv     = os.path.join(proj_root, summary_csv)
        base_array_dir  = os.path.join(proj_root, base_array_dir)

        # sanity‐check
        if not os.path.exists(summary_csv):
            raise FileNotFoundError(f"Could not find summary.csv at: {summary_csv}")
        if not os.path.isdir(base_array_dir):
            raise FileNotFoundError(f"Could not find base_arrays dir at: {base_array_dir}")

        # Load summary data
        self.summary   = pd.read_csv(summary_csv, sep=None, engine='python').set_index('id')
        self.ids       = list(self.summary.index)
        self.base_dir  = base_array_dir

        # Normalize static features
        feat = self.summary[['gwas_count','mean_cons','tfbs_count','reg_count','atac_count']]
        self.feat_min, self.feat_max = feat.min(), feat.max()
        self.features = (feat - self.feat_min) / (self.feat_max - self.feat_min + 1e-9)

        # Action & observation spaces
        self.action_space = spaces.MultiDiscrete([3,3])
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Dynamic state
        self.interp_score = interp_init
        self.att_scaling  = 1.0
        self.bias         = 0.0
        self.att_weights  = None
        self.current_id   = None

    def reset(self):
        self.current_id = random.choice(self.ids)
        length = int(self.summary.at[self.current_id,'length'])

        self.interp_score = 0.2
        self.att_scaling  = 1.0
        self.bias         = 0.0
        self.att_weights  = np.ones(length, dtype=np.float32) / length
        return self._get_state()

    def step(self, action):
        delta = {0:-0.1, 1:0.0, 2:0.1}
        self.att_scaling = max(0.0, self.att_scaling + delta[action[0]])
        self.bias       += delta[action[1]]

        # update & renormalize attention
        self.att_weights = np.clip(self.att_weights * self.att_scaling + self.bias, 0.0, None)
        self.att_weights /= (self.att_weights.sum() + 1e-9)

        # compute interpretability
        entropy = -np.sum(self.att_weights * np.log(self.att_weights + 1e-9))
        max_ent = np.log(len(self.att_weights))
        self.interp_score = float(1.0 - entropy / max_ent)

        # proxy loss
        loss_norm = float(np.clip(1.0 - self.interp_score, 0.0, 1.0))

        # base reward
        reward = (1.0 - loss_norm) + self.interp_score

        # GWAS‐attention overlap bonus
        arr = np.load(os.path.join(self.base_dir, f"{self.current_id}.npz"))
        overlap = int(np.sum((self.att_weights > 0.8) & (arr['gmask'] > 0)))
        reward += 0.1 * overlap

        done = bool(self.interp_score > 0.8 and loss_norm < 0.2)
        return self._get_state(), float(reward), done, {}

    def _get_state(self):
        feats = self.features.loc[self.current_id].values.astype(np.float32)
        return np.concatenate([feats, [self.interp_score]]).astype(np.float32)

    def render(self, mode='human'):
        print(f"ID: {self.current_id}  interp={self.interp_score:.3f}  scale={self.att_scaling:.3f}  bias={self.bias:.3f}")
        print("Features:", self.features.loc[self.current_id].to_dict())
