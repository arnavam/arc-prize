
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import json
from itertools import product
from neurosymbolic

from  DSL import find_objects

# --- Neural Feature Extractor (PyTorch) ---


# --- Symbolic Program Generator ---
# (This section remains the same)
PRIMITIVES = [
    'rotate', 'mirrorlr', 'mirrorud', 'lcrop', 'rcrop', 'ucrop', 'dcrop',
    'recolor', 'select', 'fill', 'overlay', 'resize'
]

# --- Policy Network (Replaces ProgramExecutor) ---
class PolicyNetwork(nn.Module):
    """
    The policy network decides which action to take.
    It takes the state (current grid + target grid) and outputs action probabilities.
    """
    def __init__(self, feature_extractor, num_primitives):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Combined feature vector size is 128 (current) + 128 (target)
        self.policy_head = nn.Linear(128 * 2, num_primitives)
        
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
    def __init__(self, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.policy = PolicyNetwork(self.feature_extractor, len(PRIMITIVES)).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = gamma # Discount factor for future rewards
        
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
                
                # Execute the action
                action = PRIMITIVES[action_index.item()]
                current_grid_np = self.execute_action(current_grid_np, action)
                
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

            returns = self._calculate_discounted_returns(rewards)
            
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
                action = PRIMITIVES[action_index]
                
                solution_program.append(action)
                current_grid_np = self.execute_action(current_grid_np, action)
                
                # In a real scenario, you'd have a way to check if it's solved
                # Here we just return the generated program after max_steps
                
        return solution_program, current_grid_np

    def execute_action(self, grid, action):
        """Execute a single primitive action on a grid."""
        current = grid.copy()
        if action == 'rotate': current = np.rot90(current, k=-1)
        elif action == 'mirrorlr': current = np.fliplr(current)
        elif action == 'mirrorud': current = np.flipud(current)
        elif action == 'lcrop': current = current[:, 1:] if current.shape[1] > 1 else current
        elif action == 'rcrop': current = current[:, :-1] if current.shape[1] > 1 else current
        elif action == 'ucrop': current = current[1:, :] if current.shape[0] > 1 else current
        elif action == 'dcrop': current = current[:-1, :] if current.shape[0] > 1 else current
        return current
        
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
    
    def _preprocess_to_tensor(self, grid, size=30):
        """Preprocess grid and convert to a tensor on the correct device."""
        h, w = grid.shape
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:h, :w] = grid
        # Add batch and channel dimensions (N, C, H, W)
        tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

# --- Main Execution ---
if __name__ == "__main__":
    # A dummy JSON file path for demonstration purposes
    json_path = 'arc-prize-2025/arc-agi_training_challenges.json'
    
    try:
        with open(json_path, 'r') as f:
            train_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: JSON file not found. Using dummy data.")
        train_data = {
            "case_001": {"train": [{"input": [[5,0],[0,0]], "output": [[0,5],[0,0]]}]},
            "case_002": {"train": [{"input": [[0,8],[0,0]], "output": [[0,0],[0,8]]}]}
        }

    training_examples = []
    for case_data in train_data.values():
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
