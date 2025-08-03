import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
torch.mps.empty_cache()

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


class QNetwork(nn.Module):
    def __init__(self, feature_extractor, num_actions):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        combined_feature_size = 32 * 2  # Based on your previous feature extractor
        
        # A single head that outputs one Q-value per action
        self.q_head = nn.Linear(combined_feature_size, num_actions)

    def forward(self, inputs):
        current_grid_tensor, target_grid_tensor = inputs
        current_feat = self.feature_extractor(current_grid_tensor)
        target_feat = self.feature_extractor(target_grid_tensor)
        
        combined = torch.cat([current_feat, target_feat], dim=-1)
        
        # Output raw Q-values for each action
        q_values = self.q_head(combined)
        
        return q_values

import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        """Save an experience tuple (state, action, reward, next_state)"""
        self.memory.append(experience)

    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
import torch
import torch.optim as optim
import random

class DQN_Solver:
    def __init__(self, ACTION_NAMES, feature_extractor, gamma=0.99, lr=1e-4, batch_size=128, memory_size=10000, target_update=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_names = ACTION_NAMES
        self.num_actions = len(ACTION_NAMES)
        
        # Main network (gets updated frequently)
        self.policy_net = QNetwork(feature_extractor, self.num_actions).to(self.device)
        # Target network (provides stable targets, updated less often)
        self.target_net = QNetwork(feature_extractor, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        self.update_counter = 0
        self.target_update_frequency = target_update

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randrange(self.num_actions)
        else:
            # Exploitation: choose the best action from the policy network
            with torch.no_grad():
                current, target = state
                # current_tensor = self._preprocess_to_tensor(current)
                # target_tensor = self._preprocess_to_tensor(target)
                current_tensor =current
                target_tensor=target
                
                # Get Q-values and find the action with the highest value
                q_values = self.policy_net([current_tensor, target_tensor])
                return torch.argmax(q_values).item()

    def update_policy(self):
        """Train the model using a batch from the replay memory"""
        if len(self.memory) < self.batch_size:
            return # Don't train until we have enough experiences

        experiences = self.memory.sample(self.batch_size)
        # Transpose the batch for easier access
        batch = list(zip(*experiences))

        # Unpack the batch
        states, actions, rewards, next_states = batch
        
        # --- Prepare Tensors ---
        # Convert all experiences to tensors
        current_grids = torch.cat([self._preprocess_to_tensor(s[0]) for s in states])
        target_grids = torch.cat([self._preprocess_to_tensor(s[1]) for s in states])
        next_current_grids = torch.cat([self._preprocess_to_tensor(ns[0]) for ns in next_states])
        next_target_grids = torch.cat([self._preprocess_to_tensor(ns[1]) for ns in next_states])
        
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # --- Calculate Q-Values ---
        # 1. Get Q-values for the actions that were actually taken
        current_q_values = self.policy_net([current_grids, target_grids]).gather(1, actions_tensor)

        # 2. Get the maximum Q-value for the next states from the target network
        with torch.no_grad():
            next_q_values = self.target_net([next_current_grids, next_target_grids]).max(1)[0]
        
        # --- Calculate Target Q-Value (Bellman Equation) ---
        target_q_values = rewards_tensor + (self.gamma * next_q_values)

        # --- Calculate Loss ---
        # Using Smooth L1 Loss is common in DQN for stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # Gradient clipping
        self.optimizer.step()

        # --- Update Target Network ---
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _preprocess_to_tensor(self, grid, size=30):
        # This is your existing preprocessing function
        
        # ✅ FIX: Ensure the grid is at least 2D
        grid = np.atleast_2d(grid) 
        
        h, w = grid.shape # This line will now work correctly
        
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:h, :w] = grid
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def store_experience(self, state, action, reward, next_state):
        """
        Saves a complete experience tuple to the replay memory.
        This is a helper function that uses the ReplayMemory module.
        """
        # Here it is using the memory module you created!
        self.memory.push((state, action, reward, next_state))