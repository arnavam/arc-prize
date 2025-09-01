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

class Policy(nn.Module):


    def __init__(self, feature_extractor, num_actions, no_of_inputs=3):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # The size of the feature vector after being processed by the feature_extractor
        combined_feature_size = 64 * no_of_inputs
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 128)
<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
        self.layer2 = nn.Linear(128, 64)

        self.q_head = nn.Linear(64, num_actions)
=======
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 64)
        self.q_head = nn.Linear(64, no_of_outputs)
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py

    def forward(self, input_tensors, x=-1):
        """
        Forward pass through the network.
        It processes a list of input tensors, combines them, and outputs Q-values.
        """

        features = [self.feature_extractor(tensor) for tensor in input_tensors]

<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
        # Concatenate the features from all inputs
        combined = torch.cat(features, dim=-1)
        
        # Pass through the fully connected layers to get Q-values
        out1 = F.relu(self.layer1(combined))
        out2= F.relu(self.layer2(out1))
        q_values = self.q_head(out2)
        return q_values
=======
        combined = torch.cat(features, dim=-1) # Concatenate the features from all inputs
        
       
        x = F.relu(self.layer1(combined))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))

        x = self.q_head(x)
        return x

    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

        if isinstance(grid, torch.Tensor):  
            tensor = grid.to(device=self.device, dtype=dtype)
        else:
            array = np.asarray(grid)

            if array.dtype == np.object_:# If it's object dtype, try to convert to numeric
                logger.debug("WARNING: NumPy array has dtype=object. Attempting to convert to numeric dtype...")

                try:    # Try float conversion by default
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
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py



# --- REINFORCE Agent ---
class Likelihood:
<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
    def __init__(self, feature_extractor, output_dim,no_of_inputs=3, lr=1e-3, device="cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = ObjectPolicy(feature_extractor, output_dim, no_of_inputs).to(self.device)
=======
    def __init__(self, feature_extractor, no_of_outputs,no_of_inputs=3, lr=1e-5, device="cpu"):
        self.device = torch.device(device if getattr(torch, device).is_available() else "cpu")

        self.policy = Policy(feature_extractor, no_of_outputs, no_of_inputs,self.device).to(self.device)
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        criterion = nn.CrossEntropyLoss()
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
<<<<<<< Updated upstream:rl_models/ReinLikelihood.py

        # clear memory
=======
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py
        self.memory = []

    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
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



=======
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py

    def train_supervised(self , inputs, labels): #labels -> crct obj index
        criterion = nn.CrossEntropyLoss()
<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
        self.optimizer.zero_grad()
        
=======

>>>>>>> Stashed changes:dl_models/ReinLikelihood.py
        all_scores = []
        for current_grid, objects, target_grid in inputs:

<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
            current_grid_t = self._preprocess_to_tensor(current_grid)
            target_grid_t = self._preprocess_to_tensor(target_grid)
            
            scores_per_sample = []
            for obj in objects:
                obj_grid_t = self._preprocess_to_tensor(obj['grid'])

                score = self.policy([current_grid_t, obj_grid_t, target_grid_t])
                scores_per_sample.append(score.squeeze())

            # display(current_grid,target_grid,objects[max(enumerate(scores_per_sample), key=lambda x: x[1])[0]])

            all_scores.append(torch.stack(scores_per_sample))

        scores_batch = torch.stack(all_scores).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)

        loss = criterion(scores_batch, labels)
        
=======
        for current_grid, objects, target_grid , _ in inputs:

            scores_per_sample = []
            for obj in objects:

                # Default position if not specified
                if obj['position']  == (None,None):
                    obj['position']=(target_grid.shape[0]//2,target_grid.shape[1]//2)
                 
                # place obj in a target grid shaped grid to encode position information
                obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])

                # Predict score for each the object
                score = self.policy([current_grid, obj_grid, target_grid])
                scores_per_sample.append(score.squeeze())

            all_scores.append(torch.stack(scores_per_sample))


            # # Visualization (optional) in   folder 'likelihood_predictions'
            obj=objects[max(enumerate(scores_per_sample), key=lambda x: x[1])[0]]
            obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])
            display(current_grid,target_grid,obj_grid,folder='likelihood_predictions') 


        scores_batch = torch.stack(all_scores).to(self.device)
        obj_labels = torch.tensor(obj_labels, dtype=torch.long, device=self.device)

        loss = criterion(scores_batch, obj_labels)

        self.optimizer.zero_grad()
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            # calculate accuracy
            preds = scores_batch.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        return loss.item(), acc

        
    def predict_supervised(self, input_data):

        current_grid, objects, target_grid = input_data
        
        # Set the self to evaluation mode
        self.policy.eval()
        with torch.no_grad():
            current_grid_t = self._preprocess_to_tensor(current_grid)
            target_grid_t = self._preprocess_to_tensor(target_grid)
            
            scores = []
            for obj in objects:
<<<<<<< Updated upstream:rl_models/ReinLikelihood.py
                obj_grid_t = self._preprocess_to_tensor(obj['grid'])
                score = self.policy([current_grid_t, obj_grid_t, target_grid_t])
=======
                # Default position if not specified
                if obj['position']  == (None,None):
                    obj['position']=(target_grid.shape[0]//2,target_grid.shape[1]//2)
               
                # place obj in a target grid shaped grid to encode position information
                obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])

                # Predict score for each the object
                score = self.policy([current_grid, obj_grid, target_grid])
>>>>>>> Stashed changes:dl_models/ReinLikelihood.py
                scores.append(score.squeeze())
            
            scores_t = torch.stack(scores)
            # Choose the action with the highest score
            best_action = torch.argmax(scores_t).item()
        
        # Set the self back to training mode
        self.policy.train()
        return best_action


    def load(self):
        self.policy.load_state_dict(torch.load(f'weights/{self.__class__.__name__}.pth'))
    def save(self):
        torch.save(self.policy.state_dict(), f'weights/{self.__class__.__name__}.pth')

    def show_structure(self):
        for name, param in self.policy.state_dict().items():
            print(name, param.shape)
