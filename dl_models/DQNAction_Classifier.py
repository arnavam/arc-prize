import numpy as np
import torch 
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from dl_models.BaseDQN import BaseDQN
from helper_env import placement , place_object
from  helper_arc import display
import logging
from dsl import ALL_ACTIONS
action_names= list(ALL_ACTIONS.keys())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/action_classifer.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

import random
from collections import deque

# --- Neural Network Definition ---
# This remains a separate, modular component.
class QNetwork(nn.Module):
    """
    The specific Q-Network architecture. This can be swapped out with other
    architectures without changing the core DQN logic.
    """
    def __init__(self, feature_extractor, num_actions, device,no_of_inputs=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device=device
        
        combined_feature_size = 32 * no_of_inputs # Assuming the feature_extractor outputs a tensor of size 32.
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 128)
        self.layer2 = nn.Linear(128,64)
        self.layer21 = nn.Linear(64, 32)
        self.layer22 = nn.Linear(64, 32)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.bn2 = nn.BatchNorm1d(64)

        self.q_head = nn.Linear(32, num_actions)
        self.pos_head = nn.Linear(32, 2) 
        
        
    def forward(self, inputs, x=2):

        # for i in input_tensors:
        #     print('features shape',i.shape)
        input_tensors = [ self._preprocess_to_tensor(i) for i in  inputs]
        features = [self.feature_extractor(tensor) for i, tensor in enumerate(input_tensors) if i != x]
        # for i in features:
        #     print('features shape',i.shape)
        
        combined = torch.cat(features, dim=-1)
        
        out1 = F.relu(self.layer1(combined))
        out1 = F.relu(self.layer2(out1))

        out2= F.relu(self.layer21(out1))
        action_values = self.q_head(out2)
        
        out2= F.relu(self.layer22(out1))
        pos_values = self.pos_head(out2)
        pos_values = torch.sigmoid(pos_values) 
        return action_values, pos_values

    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

        if isinstance(grid, torch.Tensor):  
            tensor = grid.to(device=self.device, dtype=dtype)
        else:
            array = np.asarray(grid)

            # If it's object dtype, try to convert to numeric
            if array.dtype == np.object_:
                logger.debug("WARNING: NumPy array has dtype=object. Attempting to convert to numeric dtype...")

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


# ---  Multi-Head DQN ---
class DQN_Classifier(BaseDQN):

    def __init__(self, feature_extractor, num_actions, no_of_inputs=2, device='cpu', gamma=0.99, lr=1e-2, batch_size=128, memory_size=1000, target_update=10):
        _device = torch.device(device if getattr(torch, device).is_available() else "cpu")
        print(f'Using device: {_device} for Multi-Head Solver')

        # 1. Create a QNetwork with use_pos_head=True
        policy_net = QNetwork(feature_extractor, num_actions,_device,no_of_inputs ).to(_device)
        target_net = QNetwork(feature_extractor, num_actions,_device,no_of_inputs ).to(_device)
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
        # Convert to tensors before storing (target_net grid is not stored)
        state_tensors = [ self._preprocess_to_tensor(i) for i in  state]
        
        next_state_tensors = [self._preprocess_to_tensor(i)  for i in state]
        
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        true_pos_tensor = torch.tensor(true_position, dtype=torch.float32)
        is_place_tensor = torch.tensor(is_place_action, dtype=torch.bool)
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

            logger.debug(f"'random values',{action_idx,values}")
            return action_idx , values
        else:
            # Exploitation: choose the best action from the policy_net network
            with torch.no_grad():
                # Preprocess state and get Q-values
                action_values , pos_values = self.policy_net(state)
                action_idx = torch.argmax(action_values).item()
                pos_values = pos_values.tolist()[0]

                logger.debug(f"'action_idx,pos_values:',{action_idx,pos_values}")
                
                return action_idx , pos_values

    def update_policy(self):
        """Optimized policy_net update with pre-tensorized experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample already tensorized experiences
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, true_positions, is_place_flags = zip(*experiences)
        

        current_grids = torch.cat([s[0] for s in states], dim=0)

        try:
         obj_grids = torch.cat([s[1] for s in states], dim=0)
        except Exception as e:
            logger.warning(f"Concatenation failed: {e}")
            for i, s in enumerate(states):
                logger.warning(f"Tensor {i}: shape {s[1].shape}")

        obj_positions = torch.cat([s[2] for s in states], dim=0)

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
        if rewards_tensor.isnan().any():
            ("Warning: NaN values found in rewards_tensor. Replacing with 0.")
            rewards_tensor = torch.nan_to_num(rewards_tensor, nan=0.0)

        pred_q_values = pred_q_values_all.gather(1, actions_tensor.unsqueeze(1))

        # Compute target_net Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_net([
                next_current_grids, next_obj_grids, next_obj_positions, target_grids
            ])


            next_q_values = torch.clamp(next_q_values, min=-1e6, max=1e6)  
            target_q_values = rewards_tensor + (self.gamma * next_q_values.max(1)[0])



        # Calculate losses
        loss_q = F.smooth_l1_loss(pred_q_values, target_q_values.unsqueeze(1))
        
        # Check the loss itself
        if loss_q.isnan():
            logger.warning("loss_q is NaN! Halting or debugging.")
            loss_q = torch.nan_to_num(loss_q, nan=0.0)

            # import pdb; pdb.set_trace() # Uncomment for interactive debugging     # Position loss with masking
        loss_pos_unmasked = F.mse_loss(pred_pos_values, true_pos_tensor, reduction='none').mean(dim=1)
        masked_loss_pos = loss_pos_unmasked * loss_mask
        loss_pos = masked_loss_pos.sum() / (loss_mask.sum() + 1e-8)
        
        total_loss = loss_q + loss_pos

        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target_net network
        self.update_counter += 1
        if self.update_counter % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    def train_supervised(self, inputs, obj_labels, action_labels):

        action_criterion = nn.CrossEntropyLoss()
        pos_criterion = nn.MSELoss()  # or SmoothL1Loss()
        self.optimizer.zero_grad()
        pos_loss = torch.tensor(0.0, device=self.device) # Default to 0

        all_scores = []
        pos_preds = []
        pos_targets = []

        for (current_grid, objects, target_grid , obj_pos) ,obj_label ,action_label in zip(inputs, obj_labels ,action_labels):
            obj = objects[obj_label]
            obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])


            action_values, pos_pred = self.policy_net([current_grid, obj_grid, obj['position'], target_grid])

            all_scores.append(action_values.squeeze())  # [num_actions]
            action_name= action_names[action_label]

            if action_name == 'place':

                logger.debug('performed')
                pos_preds.append(pos_pred.squeeze())  
                pos_targets.append([obj_pos[0] / target_grid.shape[0],obj_pos[1] / target_grid.shape[1]])

            
            logger.debug(f"predictions: \n{current_grid},\n{obj_grid},\n{target_grid},{action_values.squeeze()},{action_values.argmax(dim=1)}")
            logger.debug(f"\naction-name: {action_name},predictedpos :{[int(pos_pred[0][0] * target_grid.shape[1]), int(pos_pred[0][1] * target_grid.shape[0])],obj_pos}")

        # Functional (classification) loss
        scores_batch = torch.stack(all_scores).to(self.device)  # [batch_size, num_actions]
        action_labels = torch.tensor(action_labels, dtype=torch.long, device=self.device)
        action_loss = action_criterion(scores_batch, action_labels)
        
        if pos_preds:
            pos_preds   = torch.stack(pos_preds).to(self.device)           # [batch_size, 2]
            pos_targets = torch.tensor(pos_targets,dtype=torch.float32).to(self.device)     # [batch_size, 2]
            pos_loss    = pos_criterion(pos_preds, pos_targets)
        
        alpha = 1 # This is a hyperparameter you can tune

        total_loss = action_loss + alpha * pos_loss

        total_loss.backward()

        grad_check = self.policy_net.layer1.weight.grad
        if grad_check is not None:
            print(f"Gradient mean for layer1: {grad_check.abs().mean().item()}")
        else:
            print("WARNING: Gradients for layer1 are None!")


        self.optimizer.step()

        # Accuracy
        with torch.no_grad():
            preds = scores_batch.argmax(dim=1)
            acc = (preds == action_labels).float().mean().item()

        return total_loss.item(),pos_loss.item(), acc
