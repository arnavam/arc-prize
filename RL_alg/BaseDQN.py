import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import random
from collections import deque


# --- Replay Memory ---
class ReplayMemory:
    """A simple ring buffer for storing experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        """Save an experience tuple (state, action, reward, next_state, done)"""
        self.memory.append(experience)

    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- Parent/Base DQN Class ---
class BaseDQN:
    """
    A base template for DQN agents.
    This class contains all the core logic for training a DQN,
    but it is independent of the specific neural network architecture used.
    """
    def __init__(self, policy_net, target_net, optimizer, device, gamma, batch_size, memory_size, target_update):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        
        # The models and optimizer are now passed in during initialization
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        
        self.memory = ReplayMemory(memory_size)
        self.update_counter = 0
        self.target_update_frequency = target_update

    def select_action(self, state, epsilon=0.4):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            # Exploration: choose a random action
            return random.randrange(self.policy_net.q_head.out_features)
        else:
            # Exploitation: choose the best action from the policy network
            with torch.no_grad():
                # Preprocess state and get Q-values
                state_tensors = [self._preprocess_to_tensor(s) for s in state]
                q_values = self.policy_net(state_tensors)
                action_idx = torch.argmax(q_values).item()
                return action_idx

    def update_policy(self):
        """Train the model using a batch from the replay memory."""
        if len(self.memory) < self.batch_size:
            return  # Don't train until we have enough experiences

        experiences = self.memory.sample(self.batch_size)
        batch = list(zip(*experiences))

        states, actions, rewards, next_states = batch
        
        # Prepare Tensors for the batch
        # This part is specific to your state structure ([grid1, grid2])
        current_grids = torch.cat([self._preprocess_to_tensor(s[0]) for s in states])
        target_grids = torch.cat([self._preprocess_to_tensor(s[1]) for s in states])
        next_current_grids = torch.cat([self._preprocess_to_tensor(ns[0]) for ns in next_states])
        next_target_grids = torch.cat([self._preprocess_to_tensor(ns[1]) for ns in next_states])
        
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # --- Calculate Q-Values ---
        # 1. Get the Q-values for the actions that were actually taken
        current_q_values = self.policy_net([current_grids, target_grids]).gather(1, actions_tensor)

        # 2. Get the maximum Q-value for the next states from the target network
        with torch.no_grad():
            next_q_values = self.target_net([next_current_grids, next_target_grids]).max(1)[0]
        
        # --- Calculate Target Q-Value (Bellman Equation) ---
        target_q_values = rewards_tensor + (self.gamma * next_q_values)

        # --- Calculate Loss ---
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0) # More common clipping method
        self.optimizer.step()

        # --- Update Target Network ---
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())





    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

        if isinstance(grid, torch.Tensor):

            return grid.to(device=self.device, dtype=dtype)

        array = np.asarray(grid)

        # If it's object dtype, try to convert to numeric
        if array.dtype == np.object_:
            print("WARNING: NumPy array has dtype=object. Attempting to convert to numeric dtype...")

            try:
                # Try float conversion by default
                array = array.astype(np.float32 if dtype.is_floating_point else np.int32)
                print(f"Successfully converted object array to dtype={array.dtype}")
            except Exception as e:
                print("ERROR: Failed to convert object array to numeric type.",array)
                print("Details:", e)
                raise ValueError("Grid contains non-numeric data or inconsistent structure.")

        # Convert to tensor
        try:
            tensor = torch.from_numpy(array).to(dtype)
        except Exception as e:
            print("ERROR: torch.from_numpy failed.")
            print("Details:", e)
            raise ValueError("Failed to convert numpy array to tensor.")
        tensor = tensor.view(1, -1)

        return tensor.to(self.device)


    def store_experience(self, state, action, reward, next_state):
        """Saves an experience tuple to the replay memory."""
        self.memory.push((state, action, reward, next_state))

    def load(self, path):
        """Loads the policy network's weights."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
        print(f"Model loaded from {path}")

    def save(self, path):
        """Saves the policy network's weights."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

