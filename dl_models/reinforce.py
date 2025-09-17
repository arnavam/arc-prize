
import numpy as np
import torch


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from itertools import product
from A_arc import train 
from dsl import PRIMITIVE
import functools ,collections,time
PRIMITIVE_NAMES = list(PRIMITIVE.keys())
import random
# --- Neural Feature Extractor (PyTorch) ---
class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))  # Safe pooling
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 32)  # final feature dim
        self.primitive_names= list(PRIMITIVE.keys())


    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x



# --- Policy Network ---
class PolicyNetwork(nn.Module):
    """
    The policy network decides which action to take.
    It takes the state (current grid + target grid) and outputs action probabilities.
    """
    def __init__(self, feature_extractor, num_primitives):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Combined feature vector size is 128 (current) + 128 (target)
        self.policy_head = nn.Linear(32 * 2, num_primitives)
        
    def forward(self, inputs):
        current_grid_tensor, target_grid_tensor = inputs
        current_feat = self.feature_extractor(current_grid_tensor)
        target_feat = self.feature_extractor(target_grid_tensor)
        
        combined = torch.cat([current_feat, target_feat], dim=-1)
        # Output logits, which will be converted to probabilities
        action_logits = self.policy_head(combined)
        return F.softmax(action_logits, dim=-1)



    

# --- Neural-Symbolic RL Solver ---
class NeuralSymbolicSolverRL:
    def __init__(self,PRIMITIVE_NAMES,feature_extractor =FeatureExtractor(), gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')#("mps" if torch.mps.is_available() else "cpu"))
        self.feature_extractor = feature_extractor
        self.policy = PolicyNetwork(self.feature_extractor, len(PRIMITIVE_NAMES)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        
        self.gamma = gamma # Discount factor for future rewards
        self.memory=[()] # stores (state, action, reward)
        self.states=[]
        self.actions=[]
        self.log_probs=[]
        self.rewards=[]


        

    def select_action(self, state):
        """
        Selects an action by sampling from the policy distribution 
        and stores the log probability.
        """
        current, target = state
        
        # No need for torch.no_grad() here since we need the graph for log_prob
        action_probs = self.policy([current, target])
        
        # Create a categorical distribution to sample from
        m = Categorical(action_probs)
        action = m.sample() # 🎲 Sample an action!
        
        # Store the state and the log probability of the sampled action
        self.states.append(state)
        self.log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        self.rewards.append(reward)

    def update_policy(self, gamma=0.99):
        if not self.states:
            return
            
        # Calculate returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        # Compute loss
        policy_loss = torch.stack([
            -lp * R for lp, R in zip(self.log_probs, returns)
        ]).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Reset storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

    def _preprocess_to_tensor(self, grid, size=30):
        """Preprocess grid and convert to a tensor on the correct device."""
        h, w = grid.shape
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:h, :w] = grid
        # Add batch and channel dimensions (N, C, H, W)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)


    def train(self, train_data, episodes=1000, max_steps_per_episode=10):
        """Train the agent using the REINFORCE algorithm."""
        self.policy.train()
        
        for episode in range(episodes):
            # Select a random problem for this episode
            problem = train_data[episode % len(train_data)]
            input_grid_np = np.array(problem['input'])
            target_grid_np = np.array(problem['output'])
            
            current_grid_np = input_grid_np.copy()
            
            saved_log_probs = []
            rewards = []
            
            # --- Run one episode ---
            for t in range(max_steps_per_episode):
                # Prepare state tensors
                current_grid_tensor = self._preprocess_to_tensor(current_grid_np)
                target_grid_tensor = self._preprocess_to_tensor(target_grid_np)

                # Get action probabilities from the policy network
                action_probs = self.policy([current_grid_tensor, target_grid_tensor])
                dist = Categorical(action_probs)
                
                # Sample an action to explore
                action_index = dist.sample()
                saved_log_probs.append(dist.log_prob(action_index))

                # convert name to number using primitive-name list.
                primitive_name = PRIMITIVE_NAMES[action_index]
                #execute the primitive from the globally defined primitives in dsl file

                current_grid_np = PRIMITIVE[primitive_name](current_grid_np)



                # Determine reward
                if np.array_equal(current_grid_np, target_grid_np):
                    reward = 10.0  # High reward for solving
                    rewards.append(reward)
                    break
                else:
                    reward = -0.1 # Small penalty for each step
                    rewards.append(reward)
            
            # --- Update the policy after the episode ---
            if not saved_log_probs:
                continue

            returns = self.policy(rewards)
            
            policy_loss = []
            for log_prob, R in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * R)
            
            self.optimizer.zero_grad()
            # Sum the loss for all steps in the episode
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Last episode length: {len(rewards)}, Total Loss: {loss.item():.2f}")
    
    def solve(self, input_grid, max_steps=10):
        """Solve a new problem by greedily following the learned policy."""
        self.policy.eval()
        input_grid_np = np.array(input_grid)
        target_grid_np = np.array(input_grid) # Placeholder, not used for solving
        
        current_grid_np = input_grid_np.copy()
        solution_program = []
        
        with torch.no_grad():
            for _ in range(max_steps):
                current_grid_tensor = self._preprocess_to_tensor(current_grid_np)
                # For solving, we use a dummy target, as it's part of the model input
                # A more advanced model might not require the target during inference
                dummy_target_tensor = self._preprocess_to_tensor(target_grid_np)
                
                action_probs = self.policy([current_grid_tensor, dummy_target_tensor])
                # Choose the best action (greedy) instead of sampling
                action_index = torch.argmax(action_probs).item()
 

                # convert name to number using primitive-name list.
                primitive_name = PRIMITIVE_NAMES[action_index]
                #execute the primitive from the globally defined primitives in dsl file
                


                solution_program.append(primitive_name)
                current_grid_np = PRIMITIVE[primitive_name](current_grid_np)
                
                # In a real scenario, you'd have a way to check if it's solved
                # Here we just return the generated program after max_steps
                
        return solution_program, current_grid_np



        
    def _calculate_discounted_returns(self, rewards):
        """Calculate the discounted reward-to-go for each step of an episode."""
        R = 0
        returns = []
        # Iterate backwards through the rewards
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns for more stable training
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
    

# --- Main Execution ---
if __name__ == "__main__":
    

    training_examples = []
    for case_data in train.values():
        training_examples.extend(case_data['train'])
    
    solver_rl = NeuralSymbolicSolverRL()
    print("--- Starting RL Training ---")
    # Using a small number of episodes for a quick demonstration
    solver_rl.train(training_examples, episodes=500, max_steps_per_episode=5)
    print("--- RL Training Finished ---")
    
    # Test on a sample case
    sample_input = training_examples[0]['input']
    
    print("\n--- Solving a Sample Case ---")
    solution_program, result_grid = solver_rl.solve(sample_input)
    
    print(f"Solution Program Found: {solution_program}")
    print("\nOriginal Input:")
    print(np.array(sample_input))
    print("\nFinal Output Grid:")
    print(result_grid)
