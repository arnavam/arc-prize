import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_alg.BaseDQN import BaseDQN
import random  

class ObjectPolicy(nn.Module):


    def __init__(self, feature_extractor, num_actions, n=3):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # The size of the feature vector after being processed by the feature_extractor
        # Assuming the feature_extractor outputs a tensor of size 32.
        combined_feature_size = 32 * n
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 64)
        self.q_head = nn.Linear(64, num_actions)

    def forward(self, input_tensors, x=-1):
        """
        Forward pass through the network.
        It processes a list of input tensors, combines them, and outputs Q-values.
        """

        features = [self.feature_extractor(tensor) for tensor in input_tensors]

        # Concatenate the features from all inputs
        combined = torch.cat(features, dim=-1)
        
        # Pass through the fully connected layers to get Q-values
        out1 = F.relu(self.layer1(combined))
        q_values = self.q_head(out1)
        return q_values



# --- REINFORCE Agent ---
class Likelihood:
    def __init__(self, feature_extractor, output_dim,n=3, lr=1e-3, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy_net = ObjectPolicy(feature_extractor, output_dim, n).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []
        
    def select_action(self, input):

        current_grid, objects, target_grid = input
        current_grid = self._preprocess_to_tensor(current_grid)
        target_grid = self._preprocess_to_tensor(target_grid)
        scores=[]
        for obj in objects:
            score= self.policy_net([current_grid,self._preprocess_to_tensor(obj['grid']),target_grid])
            score = score.squeeze() 
            scores.append(score)

        scores = torch.stack(scores).to(self.device)  # [n, input_dim]

        probs = torch.softmax(scores , dim=0)         # [n]

        # Sample from categorical distribution
        dist = D.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_experience(self, log_prob, reward):

        self.memory.append((log_prob, reward))

    def update_policy(self, gamma=0.99):

        # compute discounted returns
        rewards = [r for _, r in self.memory]
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # compute loss
        loss = 0
        for (logp, _), R in zip(self.memory, returns):
            loss += -logp * R

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # clear memory
        self.memory = []

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
    

    def load(self, path):
        """Loads the policy_net network's weights."""
        self.policy_net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def save(self, path):
        """Saves the policy_net network's weights."""
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")
