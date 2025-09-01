import numpy as np
import torch 
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_models.BaseDQN import BaseDQN
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to predicted_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)


import random
from collections import deque

# --- Neural Network Definition ---
# This remains a separate, modular component.
class Policy(nn.Module):
    """
    The specific Q-Network architecture. This can be swapped out with other
    architectures without changing the core DQN logic.
    """
    def __init__(self, feature_extractor, num_actions, no_of_inputs=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        
<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
        # The size of the feature vector after being processed by the feature_extractor
        # Assuming the feature_extractor outputs a tensor of size 32.
        combined_feature_size = 32 * no_of_inputs 
=======
        combined_feature_size = 64 * no_of_inputs # Assuming the feature_extractor outputs a tensor of size 32.
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 128)
        self.layer2 = nn.Linear(128,64)
        self.layer21 = nn.Linear(64, 32)
        self.layer22 = nn.Linear(64, 32)


        self.q_head = nn.Linear(32, num_actions)
        self.pos_head = nn.Linear(32, 2) 
        
<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py

    def forward(self, input_tensors, x=2):
        """
        Forward pass through the network.
        It processes a list of input tensors, combines them, and outputs Q-values.
        """
        # The 'x' parameter seems intended to skip feature extraction for a specific tensor,
        # for i in input_tensors:
        #     print('features shape',i.shape)

        features = [self.feature_extractor(tensor) for i, tensor in enumerate(input_tensors) if i != x]
        # for i in features:
        #     print('features shape',i.shape)
=======
        
    def forward(self, inputs, ignore=2):

        input_tensors = [ self._preprocess_to_tensor(i) for i in  inputs]
        features = [self.feature_extractor(tensor) for tensor in enumerate(input_tensors)]

>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        
        # Concatenate the features from all inputs
        combined = torch.cat(features, dim=-1)
        
<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
        # Pass through the fully connected layers to get Q-values
        out1 = F.relu(self.layer1(combined))
        out1 = F.relu(self.layer2(out1))

        out2= F.relu(self.layer21(out1))
        q_values = self.q_head(out2)
=======
        x = F.relu(self.layer1(combined))
        x = F.relu(self.layer2(x))

        x1= F.relu(self.layer21(x))
        action_values = self.q_head(x1)
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        
        x2= F.relu(self.layer22(x))
        pos_values = self.pos_head(x2)
        pos_values = torch.sigmoid(pos_values) 
        return q_values, pos_values



# ---  Multi-Head DQN ---
class DQN_Classifier(BaseDQN):

    def __init__(self, feature_extractor, num_actions, no_of_inputs=2, device='cpu', gamma=0.99, lr=1e-4, batch_size=128, memory_size=1000, target_update=10):
        _device = torch.device(device if getattr(torch, device).is_available() else "cpu")
        print(f'Using device: {_device} for Multi-Head Solver')

<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
        # 1. Create a QNetwork with use_pos_head=True
        policy_net = QNetwork(feature_extractor, num_actions,no_of_inputs ).to(_device)
        target_net = QNetwork(feature_extractor, num_actions,no_of_inputs ).to(_device)
=======
        # 1. Create a Policy 
        policy_net = Policy(feature_extractor, num_actions,_device,no_of_inputs ).to(_device)
        target_net = Policy(feature_extractor, num_actions,_device,no_of_inputs ).to(_device)
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        
        # 2. Call the parent init
        super().__init__(
            policy_net=policy_net, target_net=target_net, optimizer=optimizer,
            device=_device, gamma=gamma, batch_size=batch_size,
            memory_size=memory_size, target_update=target_update)


    def store_experience(self, state, action, reward, next_state , is_place_action):

        state_tensors = [ self._preprocess_to_tensor(i) for i in  state]
        next_state_tensors = [self._preprocess_to_tensor(i)  for i in next_state]
        
        action_tensor = torch.tensor(action, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        is_place_tensor = torch.tensor(is_place_action, dtype=torch.bool)
        
        self.memory.push((state_tensors, action_tensor, reward_tensor, 
                        next_state_tensors, is_place_tensor))


    def select_action(self, state, epsilon=0.4):
        self.target_grid_tensor = self._preprocess_to_tensor(state[3])
        self.target_grid_shape = state[3].shape
        
        if random.random() < epsilon: # Exploration: choose a random action
            
            action_idx= random.randrange(self.policy_net.q_head.out_features)
            values = [random.random(), random.random()]

<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
            logging.debug(f"'random values',{action_idx,values}")
=======
            logger.debug(f"'random values',{action_idx,values}")

>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
            return action_idx , values
        
        else: # Exploitation: choose the best action from the policy_net network
            
            with torch.no_grad():
<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
                # Preprocess state and get Q-values
                state_tensors = [self._preprocess_to_tensor(s) for s in state]
                q_values , pos_values = self.policy_net(state_tensors)
                action_idx = torch.argmax(q_values).item()
                pos_values = pos_values.tolist()[0]

                logging.debug(f"'action_idx,pos_values:',{action_idx,pos_values}")
=======
        
                action_values , pos_values = self.policy_net(state)
                action_idx = torch.argmax(action_values).item()
                pos_values = pos_values.tolist()[0]

                logger.debug(f"'action_idx,pos_values:',{action_idx,pos_values}")

>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
                return action_idx , pos_values

    def update_policy(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        experiences = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, true_positions, is_place_flags = zip(*experiences)
        
        # upack states  & concat them to tensor
        current_grids, obj_grids  = (torch.cat(tensors, dim=0) for tensors in zip(*states))
        next_current_grids, next_obj_grids  = (torch.cat(tensors, dim=0) for tensors in zip(*states))

<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
        current_grids = torch.cat([s[0] for s in states], dim=0)

        try:
         obj_grids = torch.cat([s[1] for s in states], dim=0)
        except Exception as e:
            logging.warning(f"Concatenation failed: {e}")
            for i, s in enumerate(states):
                logging.warning(f"Tensor {i}: shape {s[1].shape}")

        obj_positions = torch.cat([s[2] for s in states], dim=0)

        next_current_grids = torch.cat([s[0] for s in next_states], dim=0)
        next_obj_grids = torch.cat([s[1] for s in next_states], dim=0)
        next_obj_positions = torch.cat([s[2] for s in next_states], dim=0)

        # --- FIX #2: Correctly repeat the target_grid_tensor ---
        # Remove the extra .unsqueeze(0) to create a proper 4D batch.
=======
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        target_grids = self.target_grid_tensor.repeat(self.batch_size, 1, 1, 1)

        actions_tensor = torch.stack(actions).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        true_pos_tensor = torch.stack(true_positions).to(self.device)
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

<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py


        # Calculate losses
        loss_q = F.smooth_l1_loss(pred_q_values, target_q_values.unsqueeze(1))
        
        # Check the loss itself
        if loss_q.isnan():
            logging.warning("loss_q is NaN! Halting or debugging.")
            loss_q = torch.nan_to_num(loss_q, nan=0.0)

            # import pdb; pdb.set_trace() # Uncomment for interactive debugging     # Position loss with masking
=======
        # Position prediction loss (masked -> only for place action) 
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
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
<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py

    def train_supervised(self, inputs, obj_labels,func_labels):


        criterion = nn.CrossEntropyLoss()
        self.optimizer.zero_grad()
=======
    
    def train_supervised(self, inputs, obj_labels, action_labels):

        action_criterion = nn.CrossEntropyLoss()
        pos_criterion = nn.MSELoss()  # or SmoothL1Loss()
        pos_loss = torch.tensor(0.0, device=self.device) # Default to 0
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py

        all_scores = []

<<<<<<< Updated upstream:rl_models/DQNAction_Classifier.py
        for (current_grid, objects, target_grid ) ,obj_label in zip(inputs,obj_labels):
            current_grid_t = self._preprocess_to_tensor(current_grid)
            target_grid_t = self._preprocess_to_tensor(target_grid)

            scores_per_sample = []
            obj=objects[obj_label]
            obj_grid_t = self._preprocess_to_tensor(obj['grid'])

  
            obj_pos_t = self._preprocess_to_tensor(obj['position'])

                # Feed forward through the policy_net
            q_values, _ = self.policy_net([current_grid_t, obj_grid_t, obj_pos_t, target_grid_t])
            scores_per_sample.append(q_values.squeeze())

            all_scores.append(torch.stack(scores_per_sample))

        # Stack into batch
        scores_batch = torch.stack(all_scores).to(self.device)   
        # print('score_batch :',scores_batch.shape,scores_batch)
        func_labels = torch.tensor(func_labels, dtype=torch.long, device=self.device)
        # print('func_labels: ',func_labels.shape,func_labels)

        scores_batch = scores_batch.squeeze(1)  # shape: [4, 10]

        loss = criterion(scores_batch, func_labels)
        loss.backward()
=======
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
        scores_batch = torch.stack(all_scores).to(self.device)  # [batch_size, num_actions]
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
>>>>>>> Stashed changes:dl_models/DQNAction_Classifier.py
        self.optimizer.step()

        # Accuracy
        with torch.no_grad():
            preds = scores_batch.argmax(dim=1)
            acc = (preds == func_labels).float().mean().item()

        return loss.item(), acc
