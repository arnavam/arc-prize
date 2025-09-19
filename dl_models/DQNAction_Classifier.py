import numpy as np
import random
from collections import deque
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
torch.autograd.set_detect_anomaly(True)
import logging


from dl_models.BaseDL import BaseDL
from helper_env import placement , place_object
from  helper_arc import display , get_module_logger
from dsl import ALL_ACTIONS

logger = get_module_logger(__name__)

action_names= list(ALL_ACTIONS.keys())



# --- Replay Memory ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        """Clear the memory."""
        self.memory.clear()
        print("Memory cleared!")
# --- Neural Network Definition ---
# This remains a separate, modular component.
class Policy(nn.Module):

    def __init__(self, feature_extractor, no_of_outputs, device, no_of_inputs=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device=device
        
        combined_feature_size = 64 * no_of_inputs # Assuming the feature_extractor outputs a tensor of size 32.
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 128)
        self.layer2 = nn.Linear(128,64)
        self.layer21 = nn.Linear(64, 32)
        self.layer22 = nn.Linear(64, 32)

        self.q_head = nn.Linear(32, no_of_outputs)
        self.pos_head = nn.Linear(32, 2) 
        
        
    def forward(self, inputs, ignore=2):

        input_tensors = [ self._preprocess_to_tensor(i) for i in  inputs]
        features = [self.feature_extractor(tensor) for tensor in input_tensors]


        
        combined = torch.cat(features, dim=-1)
        
        x = F.relu(self.layer1(combined))
        x = F.relu(self.layer2(x))

        x1= F.relu(self.layer21(x))
        action_values = self.q_head(x1)
        
        x2= F.relu(self.layer22(x))
        pos_values = self.pos_head(x2)
        pos_values = torch.sigmoid(pos_values) 
        return action_values, pos_values

    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

        if isinstance(grid, torch.Tensor):  
            tensor = grid.to(device=self.device, dtype=dtype)
        else:
            array = np.asarray(grid)

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



# ---  Multi-Head DQN ---
class DQN_Classifier(BaseDL):

    def __init__(self, feature_extractor, no_of_outputs, no_of_inputs=2, device='cpu', gamma=0.99, lr=1e-4, batch_size=30, memory_size=1000, target_update=10):
        super().__init__( device=device)

        # 1. Create a Policy 
        policy_net = Policy(feature_extractor, no_of_outputs,self.device,no_of_inputs ).to(self.device)
        target_net = Policy(feature_extractor, no_of_outputs,self.device,no_of_inputs ).to(self.device)
        
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        
        # 2. Call the parent init

        self.gamma = gamma
        self.batch_size = batch_size
        
        # The models and optimizer are now passed in during initialization
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        
        self.memory = ReplayMemory(memory_size)
        self.update_counter = 0
        self.target_update_frequency = target_update


    def store_experience(self, state, action, reward, next_state ,true_position, is_place_action):

        state_tensors = [ self._preprocess_to_tensor(i) for i in  state]
        next_state_tensors = [self._preprocess_to_tensor(i)  for i in next_state]
        
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        position_tensor = torch.tensor(true_position,dtype=torch.float32)
        is_place_tensor = torch.tensor(is_place_action, dtype=torch.bool)
        
        self.memory.push((state_tensors, action_tensor, reward_tensor, 
                        next_state_tensors, position_tensor,is_place_tensor))


    def select_action(self, state, epsilon=0.4):
        self.target_grid_tensor = self._preprocess_to_tensor(state[2])
        self.target_grid_shape = state[2].shape
        
        if random.random() < epsilon: # Exploration: choose a random action
            
            action_idx= random.randrange(self.policy_net.q_head.out_features)
            values = [random.random(), random.random()]

            logger.debug(f"'random values',{action_idx,values}")

            return action_idx , values
        
        else: # Exploitation: choose the best action from the policy_net network
            
            with torch.no_grad():
        
                action_values , pos_values = self.policy_net(state)
                action_idx = torch.argmax(action_values).item()
                pos_values = pos_values.tolist()[0]

                logger.debug(f"'action_idx,pos_values:',{action_idx,pos_values}")

                return action_idx , pos_values

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, true_positions, is_place_flags = zip(*experiences)
        
        # upack states  & concat them to tensor
        try:
            # Attempt to concatenate tensors
            current_grids, obj_grids = (torch.cat(tensors, dim=0) for tensors in zip(*states))
        except Exception as e:
            # Catch any exception and print the error message
            print(f"An error occurred: {e}")
            for state in states:
                    for tensor in state:
                        print(tensor.shape)
                    for tensor in state: 
                        print(tensor)
        
        next_current_grids, next_obj_grids  = (torch.cat(tensors, dim=0) for tensors in zip(*states))

        target_grids = self.target_grid_tensor.repeat(self.batch_size, 1, 1, 1)
        true_pos_tensor = torch.stack(true_positions).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        loss_mask = torch.stack(is_place_flags).float().to(self.device)
        

        # Get action-values and position predictions
        pred_action_values_all, pred_pos_values = self.policy_net([ current_grids, obj_grids, target_grids])
        # Handle NaNs in rewards    
        if rewards_tensor.isnan().any(): 
            ("Warning: NaN values found in rewards_tensor. Replacing with 0.")
            rewards_tensor = torch.nan_to_num(rewards_tensor, nan=0.0)


        with torch.no_grad():  # Compute target_net Q-values
            next_action_values, _ = self.target_net([next_current_grids,next_obj_grids,target_grids])
            next_action_values = torch.clamp(next_action_values, min=-1e6, max=1e6)  
            target_action_values = rewards_tensor + (self.gamma * next_action_values.max(1)[0])

         # Gather predicted action-values for taken actions
        pred_action_values = pred_action_values_all.gather(1, actions_tensor.unsqueeze(1))


        loss_action = F.smooth_l1_loss(pred_action_values, target_action_values.unsqueeze(1))# Calculate losses

        # Position prediction loss (masked -> only for place action) 
        loss_pos_unmasked = F.mse_loss(pred_pos_values, true_pos_tensor, reduction='none').mean(dim=1)
        masked_loss_pos = loss_pos_unmasked * loss_mask
        loss_pos = masked_loss_pos.sum() / (loss_mask.sum() + 1e-8)

        # Total loss
        total_loss = loss_action + loss_pos

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
        pos_loss = torch.tensor(0.0, device=self.device) # Default to 0
        
        pos_preds=[]
        pos_targets=[]
        all_scores = []

        for (current_grid, objects, target_grid , obj_pos) ,obj_label ,action_label in zip(inputs, obj_labels ,action_labels):
            
            # place object inside a target shaped input 
            obj = objects[obj_label]
            obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])

            action_values, pos_pred = self.policy_net([current_grid, obj_grid, target_grid])

            all_scores.append(action_values.squeeze())  
            action_name= action_names[action_label]

            if action_name == 'place':   # only consider pos_pred if the action is place

                pos_preds.append(pos_pred.squeeze())  
                pos_targets.append([obj_pos[0] / target_grid.shape[0],obj_pos[1] / target_grid.shape[1]])

            # Debugging logs......
            logger.debug(f"predictions: \n{current_grid},\n{obj_grid},\n{target_grid},{action_values.squeeze()},{action_values.argmax(dim=1)}")
            logger.debug(f"\naction-name: {action_name},predictedpos :{[int(pos_pred[0][0] * target_grid.shape[1]), int(pos_pred[0][1] * target_grid.shape[0])],obj_pos}")

        # classification loss
        scores_batch = torch.stack(all_scores).to(self.device)  # [batch_size, no_of_outputs]
        action_labels = torch.tensor(action_labels, dtype=torch.long, device=self.device)
        action_loss = action_criterion(scores_batch, action_labels)
        
        if pos_preds:
            pos_preds   = torch.stack(pos_preds).to(self.device)           # [batch_size, 2]
            pos_targets = torch.tensor(pos_targets,dtype=torch.float32).to(self.device)     # [batch_size, 2]
            pos_loss    = pos_criterion(pos_preds, pos_targets)
        
        alpha = 1 # This is a hyperparameter you can tune

        total_loss = action_loss + alpha * pos_loss
        
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Accuracy
        with torch.no_grad():
            preds = scores_batch.argmax(dim=1)
            acc = (preds == action_labels).float().mean().item()

        return total_loss.item(), pos_loss.item(), acc 
