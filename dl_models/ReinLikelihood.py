import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import Categorical

from helper_env import placement , place_object
from helper_arc import display , clear
from dl_models.BaseDL import BaseDL
import random  
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/likelihood.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False


class Policy(nn.Module):


    def __init__(self, feature_extractor, no_of_outputs, no_of_inputs=3,device='cpu'):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.device=device
        # The size of the feature vector after being processed by the feature_extractor
        combined_feature_size = 64 * no_of_inputs
        
        # Network layers
        self.layer1 = nn.Linear(combined_feature_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 64)
        self.q_head = nn.Linear(64, no_of_outputs)

    def forward(self, inputs, x=-1):
        
        input_tensors=[self._preprocess_to_tensor(i) for i in inputs]

        features = [self.feature_extractor(tensor) for tensor in input_tensors]

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


# --- Main Agent ---
class Likelihood(BaseDL):
    def __init__(self, feature_extractor, no_of_outputs,no_of_inputs=3, lr=1e-5, device="cpu"):
        super().__init__( device=device, )

        
        self.policy = Policy(feature_extractor, no_of_outputs, no_of_inputs,self.device).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def select_action(self, input):
        predicted_grid, objects, target_grid = input

    
        scores=[]
        for obj in objects:
            if obj['position']  == (None,None):
                obj['position']=(target_grid.shape[0]//2,target_grid.shape[1]//2)

            obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])
            
            score= self.policy([predicted_grid,obj_grid,target_grid])
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
            logger.debug("Warning: NaN values found in returns. Replacing with 0.")
            returns = torch.nan_to_num(returns, nan=0.0)

        loss = 0
        for (logp, _), R in zip(self.memory, returns):
            loss += -logp * R


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []



    def train_supervised(self , inputs, obj_labels , printing=False): #obj_labels -> crct obj index
        criterion = nn.CrossEntropyLoss()

        all_scores = []

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


            if printing:
                # # Visualization (optional) in   folder 'likelihood_predictions'
                obj=objects[max(enumerate(scores_per_sample), key=lambda x: x[1])[0]]
                obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])
                display(current_grid,target_grid,obj_grid,folder='likelihood_predictions') 


        scores_batch = torch.stack(all_scores).to(self.device)
        print('score_batch: ',scores_batch.shape)
        obj_labels = torch.tensor(obj_labels, dtype=torch.long, device=self.device)
        print('obj_labels :',obj_labels.shape)
        loss = criterion(scores_batch, obj_labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            # calculate accuracy
            preds = scores_batch.argmax(dim=1)
            acc = (preds == obj_labels).float().mean().item()

        return loss.item(), acc

        
    def predict_supervised(self, input_data):

        current_grid, objects, target_grid = input_data
        
        # Set the self to evaluation mode
        self.policy.eval()
        with torch.no_grad():

            

            scores = []
            for obj in objects:
                # Default position if not specified
                if obj['position']  == (None,None):
                    obj['position']=(target_grid.shape[0]//2,target_grid.shape[1]//2)
               
                # place obj in a target grid shaped grid to encode position information
                obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])

                # Predict score for each the object
                score = self.policy([current_grid, obj_grid, target_grid])
                scores.append(score.squeeze())

            
            scores_t = torch.stack(scores)
            
            best_action = torch.argmax(scores_t).item()
        
        # Set the self back to training mode
        self.policy.train()
        return best_action
