import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_alg.BaseDQN import BaseDQN
import random
class QNetwork(nn.Module):
    """
    The specific Q-Network architecture. This can be swapped out with other
    architectures without changing the core DQN logic.
    """
    def __init__(self, feature_extractor, num_actions, n=2):
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


# --- Child/Specific DQN Class ---
class Likelihood(BaseDQN):
    """
    A specific implementation of a DQN agent.
    This class defines the exact network architecture and optimizer to use,
    and then passes them to the BaseDQN parent class to handle the training logic.
    """
    def __init__(self, feature_extractor, num_actions, n=2, device='cpu', gamma=0.99, lr=1e-4, batch_size=128, memory_size=10000, target_update=10):
        
        # Set up the device
        _device = torch.device(device if getattr(torch, device).is_available() else "cpu")
        print(f'Using device: {_device}')

        # 1. Create the specific network instances for this solver
        policy_net = QNetwork(feature_extractor, num_actions, n).to(_device)
        target_net = QNetwork(feature_extractor, num_actions, n).to(_device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        # 2. Create the specific optimizer for this solver
        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        
        # 3. Initialize the parent class with the networks, optimizer, and other parameters
        super().__init__(
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            device=_device,
            gamma=gamma,
            batch_size=batch_size,
            memory_size=memory_size,
            target_update=target_update
        )


    def select_action(self, state, epsilon=0.4, use_sigmoid=True):
        if random.random() < epsilon:
            rand_val = random.random()
            return rand_val
        else:
            with torch.no_grad():
                state_tensors = [self._preprocess_to_tensor(s) for s in state]
                q_value = self.policy_net(state_tensors)  # Should be shape (1,)
                q_value_val = q_value.item()

                if use_sigmoid:
                    q_value_val = torch.sigmoid(torch.tensor(q_value_val)).item()

                return q_value_val

  # Return scalar
            
    # Overwrite the save/load methods to use a default name
    def load(self, path=f'DQN_Solver.pth'):
        super().load(path)

    def save(self, path=f'DQN_Solver.pth'):
        super().save(path)

