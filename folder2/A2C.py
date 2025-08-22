
import numpy as np

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from itertools import product
from A_arc import train 
from dsl import PRIMITIVE
import functools ,collections,time
PRIMITIVE_NAMES = list(PRIMITIVE.keys())




class PolicyNetwork(nn.Module):
    def __init__(self, feature_extractor, num_primitives):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # The input size for the heads, after concatenating features
        combined_feature_size = 32 * 2 

        # 1. Actor Head: Decides which action to take
        self.policy_head = nn.Linear(combined_feature_size, num_primitives)

        # 2. Critic Head: Estimates the value of the current state
        self.value_head = nn.Linear(combined_feature_size, 1)

    def forward(self, inputs):
        current_grid_tensor, target_grid_tensor = inputs
        current_feat = self.feature_extractor(current_grid_tensor)
        target_feat = self.feature_extractor(target_grid_tensor)
        
        combined = torch.cat([current_feat, target_feat], dim=-1)

        # Get action probabilities (for the Actor)
        action_probs = F.softmax(self.policy_head(combined), dim=-1)
        
        # Get state value (for the Critic)
        state_value = self.value_head(combined)
        
        return action_probs, state_value

    

class NeuralSymbolicSolverRL_A2C:
    def __init__(self, PRIMITIVE_NAMES, feature_extractor, gamma=0.99, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#"mps" if torch.mps.is_available() else "cpu")#'cpu')#()
        self.gamma = gamma
        # Use the new PolicyNetwork with two heads
        self.policy = PolicyNetwork(feature_extractor, len(PRIMITIVE_NAMES)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory to store trajectories
        # self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = [] # Need to store critic's values

    def select_action(self, state):
        current, target = state
        
        # self.policy() now returns both probs and value
        action_probs, state_value = self.policy([current, target])
        
        m = Categorical(action_probs)
        action = m.sample()
        
        # Store everything needed for the update
        # self.states.append(state)
        self.log_probs.append(m.log_prob(action))
        self.state_values.append(state_value)
        
        return action.item()

    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        if not self.rewards:
            return

        # --- Calculate Returns (Targets for the Critic) ---
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        # Optional: Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # --- Calculate Advantages (How much better was the action than expected) ---
        log_probs = torch.stack(self.log_probs)
        state_values = torch.cat(self.state_values).squeeze()
        # Advantage A(s,a) = R - V(s)
        advantages = returns - state_values
        
        # --- Calculate Losses ---
        # 1. Actor Loss (Policy Gradient)
        # We use .detach() on advantages so that gradients from this loss
        # only affect the actor, not the critic.
        actor_loss = -(log_probs * advantages.detach()).mean()

        # 2. Critic Loss (Mean Squared Error)
        # How wrong was the critic's prediction of the state value?
        critic_loss = F.mse_loss(state_values, returns)

        # 3. Total Loss (Combine actor and critic loss)
        # The critic loss is often weighted (e.g., by 0.5)
        total_loss = actor_loss + 0.5 * critic_loss

        # --- Optimize ---
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # --- Reset storage ---
        # self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        del returns, advantages, log_probs, state_values, actor_loss, critic_loss, total_loss

    def _preprocess_to_tensor(self, grid, size=30):
        """Preprocess grid and convert to a tensor on the correct device."""
        h, w = grid.shape
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:h, :w] = grid
        # Add batch and channel dimensions (N, C, H, W)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

