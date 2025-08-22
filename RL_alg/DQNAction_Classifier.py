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
        combined_feature_size = 32 * n 
        
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

        features = [self.feature_extractor(tensor) for i, tensor in enumerate(input_tensors) if i != x]
        for i in features:
            print('features shape',i.shape)
        
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
        """Store experience with tensor conversion."""
        # Convert to tensors before storing (target grid is not stored)
        state_tensors = [ self._preprocess_to_tensor(i) for i in  state]
        
        next_state_tensors = [self._preprocess_to_tensor(i)  for i in state]
        
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        true_pos_tensor = torch.tensor(true_position, dtype=torch.float32)
        is_place_tensor = torch.tensor(is_place_action, dtype=torch.bool)
        print(true_pos_tensor.shape)
        self.memory.push((state_tensors, action_tensor, reward_tensor, 
                        next_state_tensors, true_pos_tensor, is_place_tensor))


    def select_action(self, state, epsilon=0.4):
        self.target_grid_tensor = self._preprocess_to_tensor(state[3])
        self.target_grid_shape = state[3].shape
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
        """Optimized policy update with pre-tensorized experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample already tensorized experiences
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, true_positions, is_place_flags = zip(*experiences)
        
        # ... (sampling from memory) ...
        
        # --- FIX #1: Use torch.cat for states ---
        # This correctly creates a 4D batch: [128, 1, H, W]
        current_grids = torch.cat([s[0] for s in states], dim=0)
        obj_grids = torch.cat([s[1] for s in states], dim=0)
        obj_positions = torch.cat([s[2] for s in states], dim=0) # Also fix this one

        next_current_grids = torch.cat([s[0] for s in next_states], dim=0)
        next_obj_grids = torch.cat([s[1] for s in next_states], dim=0)
        next_obj_positions = torch.cat([s[2] for s in next_states], dim=0)

        # --- FIX #2: Correctly repeat the target_grid_tensor ---
        # Remove the extra .unsqueeze(0) to create a proper 4D batch.
        target_grids = self.target_grid_tensor.repeat(self.batch_size, 1, 1, 1)

        actions_tensor = torch.stack(actions).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        true_pos_tensor = torch.stack(true_positions).to(self.device)
        loss_mask = torch.stack(is_place_flags).float().to(self.device)
        
        # Get Q-values and position predictions
        pred_q_values_all, pred_pos_values = self.policy_net([
            current_grids, obj_grids, obj_positions, target_grids
        ])
        pred_q_values = pred_q_values_all.gather(1, actions_tensor.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_net([
                next_current_grids, next_obj_grids, next_obj_positions, target_grids
            ])
            target_q_values = rewards_tensor + (self.gamma * next_q_values.max(1)[0])

        # Calculate losses
        loss_q = F.smooth_l1_loss(pred_q_values, target_q_values.unsqueeze(1))
        
        # Position loss with masking
        loss_pos_unmasked = F.mse_loss(pred_pos_values, true_pos_tensor, reduction='none').mean(dim=1)
        masked_loss_pos = loss_pos_unmasked * loss_mask
        loss_pos = masked_loss_pos.sum() / (loss_mask.sum() + 1e-8)
        
        total_loss = loss_q + loss_pos

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())