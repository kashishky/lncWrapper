
# Target Network
import copy

# Create the target network at initialization
target_network = copy.deepcopy(q_network)

# In the training loop, every N steps update the target network
if step % target_update_interval == 0:
    target_network.load_state_dict(q_network.state_dict())

# Fixed Random Seeds
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# TensorBoard Logging
# Create a TensorBoard writer
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/lncrna_explain")
...
# Inside your training loop, after computing loss and total_reward:
writer.add_scalar("Loss/train", loss.item(), episode)
writer.add_scalar("Reward/train", total_reward, episode)

# Interpretability & Error Weights
interp_weight = 0.7
error_weight = 0.3
interpretability_score = compute_interpretability(state, action)
prediction_error = compute_prediction_error(state, action)
reward = interp_weight * interpretability_score - error_weight * prediction_error


# Hyperparameter Tuning (Optional)
import itertools

best_reward = -float('inf')
for lr in [0.001, 0.01, 0.1]:
    for gamma in [0.9, 0.99]:
        agent = DQN(learning_rate=lr, gamma=gamma, ...)
        reward = agent.train(max_steps=5000)
        if reward > best_reward:
            best_reward = reward
            best_params = {'lr': lr, 'gamma': gamma}
print("Best hyperparams:", best_params, "Reward:", best_reward)
