import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from RL_alg.BaseDQN import BaseDQN

import random
from collections import deque

# --- Neural Network Definition ---
# This remains a separate, modular component.
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
        combined_feature_size = 32 * n + 2
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 64)
        self.layer21 = nn.Linear(64, 32)
        self.layer22 = nn.Linear(64, 32)


        self.q_head = nn.Linear(32, num_actions)
        self.pos_head = nn.Linear(32, 2) 
        

    def forward(self, input_tensors, x=2):
        """
        Forward pass through the network.
        It processes a list of input tensors, combines them, and outputs Q-values.
        """
        # Apply the feature extractor to each input tensor
        # The 'x' parameter seems intended to skip feature extraction for a specific tensor,

        features = [
            tensor if i == x else self.feature_extractor(tensor)
            for i, tensor in enumerate(input_tensors)
        ]
        # for i in features:
        #     print('features shape',i.shape)
        # Concatenate the features from all inputs
        combined = torch.cat(features, dim=-1)
        
        # Pass through the fully connected layers to get Q-values
        out1 = F.relu(self.layer1(combined))
        out2= F.relu(self.layer21(out1))
        q_values = self.q_head(out2)
        
        out2= F.relu(self.layer22(out1))
        pos_values = self.pos_head(out2)
        pos_values = torch.sigmoid(pos_values) 
        return q_values, pos_values



# --- NEW Child Class for Multi-Head DQN ---
class DQN_Solver_MultiHead(BaseDQN):
    """
    A specific implementation for a multi-head DQN agent.
    It uses the QNetwork with the position head enabled and overrides the
    update logic to handle a combined loss with masking.
    """
    def __init__(self, feature_extractor, num_actions, n=2, device='cpu', gamma=0.99, lr=1e-4, batch_size=128, memory_size=10000, target_update=10):
        _device = torch.device(device if getattr(torch, device).is_available() else "cpu")
        print(f'Using device: {_device} for Multi-Head Solver')

        # 1. Create a QNetwork with use_pos_head=True
        policy_net = QNetwork(feature_extractor, num_actions, n, ).to(_device)
        target_net = QNetwork(feature_extractor, num_actions, n).to(_device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        
        # 2. Call the parent init
        super().__init__(
            policy_net=policy_net, target_net=target_net, optimizer=optimizer,
            device=_device, gamma=gamma, batch_size=batch_size,
            memory_size=memory_size, target_update=target_update
        )

    def store_experience(self, state, action, reward, next_state, true_position, is_place_action):
        """
        Saves an experience tuple including the ground truth position and a flag
        indicating if the position is relevant for this action.
        """
        self.memory.push((state, action, reward, next_state, true_position, is_place_action))
    def select_action(self, state, epsilon=0.4):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            # Exploration: choose a random action
            action_idx= random.randrange(self.policy_net.q_head.out_features)

            out_features = self.policy_net.pos_head.out_features

            values = [random.random(), random.random()]

            print('random values',action_idx,values)
            return action_idx , values
        else:
            # Exploitation: choose the best action from the policy network
            with torch.no_grad():
                # Preprocess state and get Q-values
                state_tensors = [self._preprocess_to_tensor(s) for s in state]
                q_values , pos_values = self.policy_net(state_tensors)
                action_idx = torch.argmax(q_values).item()
                pos_values = pos_values.tolist()[0]

                print('action_idx,pos_values:',action_idx,pos_values)
                return action_idx , pos_values

            
    def update_policy(self):
        """
        Custom update logic for the multi-head network. It calculates a combined
        loss from both the Q-learning objective and the position prediction,
        using a mask to only apply the position loss on relevant actions.
        """
        if len(self.memory) < self.batch_size:
            return

        experiences = self.memory.sample(self.batch_size)
        # Unpack 6 items, including the is_place_action flag
        batch = list(zip(*experiences))
        states, actions, rewards, next_states, true_positions, is_place_flags = batch
        
        # Prepare Tensors
        current_grids = torch.cat([self._preprocess_to_tensor(s[0]) for s in states])
        target_grids = torch.cat([self._preprocess_to_tensor(s[1]) for s in states])
        next_current_grids = torch.cat([self._preprocess_to_tensor(ns[0]) for ns in next_states])
        next_target_grids = torch.cat([self._preprocess_to_tensor(ns[1]) for ns in next_states])
        
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        true_pos_tensor = torch.tensor(true_positions, dtype=torch.float32, device=self.device)
        # Create the loss mask from the flags
        loss_mask = torch.tensor(is_place_flags, dtype=torch.float32, device=self.device)


        # --- Calculate Predicted Values from Policy Network ---
        pred_q_values_all, pred_pos_values = self.policy_net([current_grids, target_grids])
        pred_q_values = pred_q_values_all.gather(1, actions_tensor)

        # --- Calculate Target Q-Value ---
        with torch.no_grad():
            next_q_values, _ = self.target_net([next_current_grids, next_target_grids])
            max_next_q = next_q_values.max(1)[0]
        target_q_values = rewards_tensor + (self.gamma * max_next_q)

        # --- Calculate COMBINED Loss with Masking ---
        loss_q = F.smooth_l1_loss(pred_q_values, target_q_values.unsqueeze(1))
        
        # Calculate position loss for the whole batch, but don't reduce it to a single number yet
        loss_pos_unmasked = F.mse_loss(pred_pos_values, true_pos_tensor, reduction='none').mean(dim=1)
        
        # Apply the mask to zero out irrelevant losses
        masked_loss_pos = loss_pos_unmasked * loss_mask
        
        # Calculate the final position loss, only averaging over the relevant samples
        # Add a small epsilon to avoid division by zero if no 'place' actions are in the batch
        loss_pos = masked_loss_pos.sum() / (loss_mask.sum() + 1e-8)
        
        total_loss = loss_q + loss_pos

        # --- Optimize ---
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # --- Update Target Network ---
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load(self, path='DQN_Solver_MultiHead.pth'):
        super().load(path)

    def save(self, path='DQN_Solver_MultiHead.pth'):
        super().save(path)
