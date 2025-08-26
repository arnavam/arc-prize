import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_models.BaseDQN import BaseDQN
import random  
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to predicted_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

class ObjectPolicy(nn.Module):


    def __init__(self, feature_extractor, num_actions, no_of_inputs=3):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # The size of the feature vector after being processed by the feature_extractor
        combined_feature_size = 32 * no_of_inputs
        
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
    def __init__(self, feature_extractor, output_dim,no_of_inputs=3, lr=1e-3, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = ObjectPolicy(feature_extractor, output_dim, no_of_inputs).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        
    def select_action(self, input):
        predicted_grid, objects, target_grid = input
        predicted_grid = self._preprocess_to_tensor(predicted_grid)
        target_grid = self._preprocess_to_tensor(target_grid)
    
        scores=[]
        for obj in objects:
            score= self.policy([predicted_grid,self._preprocess_to_tensor(obj['grid']),target_grid])
            score = score.squeeze() 
            scores.append(score)

        scores = torch.stack(scores).to(self.device)  # [n, input_dim]

        probs = torch.softmax(scores , dim=0)         # [n]

        
        dist = D.Categorical(probs) # Sample from categorical distribution
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

       
        returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normalize

        if returns.isnan().any():
            logging.debug("Warning: NaN values found in returns. Replacing with 0.")
            returns = torch.nan_to_num(returns, nan=0.0)

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
            tensor = grid.to(device=self.device, dtype=dtype)
        else:
            array = np.asarray(grid)

            # If it's object dtype, try to convert to numeric
            if array.dtype == np.object_:
                logging.debug("WARNING: NumPy array has dtype=object. Attempting to convert to numeric dtype...")

                try:
                    # Try float conversion by default
                    array = array.astype(np.float32 if dtype.is_floating_point else np.int32)
                except Exception as e:
                    raise ValueError("ERROR: Failed to convert object array to numeric type.",array,"\nDetails:", e)

            # Convert to tensor
            try:
                tensor = torch.from_numpy(array).to(dtype)
            except Exception as e:
                raise ValueError("ERROR: torch.from_numpy failed.",array,"Details:", e)
            
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)     # shape becomes (1, 6, 6)

        return tensor.to(self.device)
    

    def load(self):
        self.policy.load_state_dict(torch.load(f'weights/{self.__class__.__name__}.pth'))
    def save(self):
        torch.save(self.policy.state_dict(), f'weights/{self.__class__.__name__}.pth')

    def show_structure(self):
        for name, param in self.policy.state_dict().items():
            print(name, param.shape)