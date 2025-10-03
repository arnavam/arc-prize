import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    A combined Actor-Critic network.

    This network shares a common backbone and has two output heads:
    1. Actor Head: Outputs a probability distribution over actions (policy).
    2. Critic Head: Outputs an estimate of the state value.
    """
    def __init__(self, shared_network, action_dim, n_latent_var=128):
        super(ActorCritic, self).__init__()

        # Shared layers for feature extraction
        self.shared_layers =shared_network

        # Actor head: outputs action logits
        self.actor_head = nn.Linear(n_latent_var, action_dim)

        # Critic head: outputs a single value
        self.critic_head = nn.Linear(n_latent_var, 1)

    def forward(self, state):

        x = self.shared_layers(state)
        
        # Get state value from the critic head
        state_value = self.critic_head(x)

        # Get action logits from the actor head
        action_logits = self.actor_head(x)
        action_dist = Categorical(logits=F.log_softmax(action_logits, dim=-1))
        
        return state_value, action_dist
    



class A2CAgent:
    def __init__(self, shared_network, action_dim, lr=0.001, gamma=0.99, entropy_beta=0.01, device='cpu'):
        self.device = torch.device(device)
        self.gamma = gamma
        self.entropy_beta = entropy_beta

        self.policy = ActorCritic(shared_network, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory buffer to store trajectory data for one episode
        self.memory = {
            "log_probs": [],
            "values": [],
            "rewards": [],
            "masks": [] # To handle terminal states
        }

    def select_action(self, state):
        """
        Selects an action based on the current policy and state.
        
        Args:
            state (np.ndarray): The current environment state.
            
        Returns:
            (int): The action to take.
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        value, dist = self.policy(state_tensor)
        
        action = dist.sample()
        
        # Store necessary info for the update step
        self.memory["log_probs"].append(dist.log_prob(action))
        self.memory["values"].append(value)
        
        return action.item()

    def store_reward(self, reward, done):
        """Stores the reward and done mask for the last step."""
        self.memory["rewards"].append(reward)
        self.memory["masks"].append(1.0 - done)

    def update(self):
        """
        Updates the policy and value networks using the collected trajectory.
        """
        # --- 1. Calculate Discounted Returns (G_t) ---
        returns = []
        discounted_return = 0
        # Iterate backwards through rewards and masks
        for reward, mask in zip(reversed(self.memory["rewards"]), reversed(self.memory["masks"])):
            discounted_return = reward + self.gamma * discounted_return * mask
            returns.insert(0, discounted_return)
        
        # Convert lists to tensors
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(self.memory["log_probs"])
        values = torch.cat(self.memory["values"])

        # --- 2. Calculate Advantage ---
        # A(s_t, a_t) = G_t - V(s_t)
        advantages = returns - values
        
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 3. Calculate Losses ---
        # Actor Loss (Policy Gradient)
        # We use .detach() on advantages as they should be treated as constants for the policy update
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic Loss (MSE or Smooth L1 Loss between returns and values)
        critic_loss = F.smooth_l1_loss(values.squeeze(), returns)
        
        # Entropy Loss (to encourage exploration)
        # We need to re-compute the distribution to get the entropy
        _, current_dist = self.policy(torch.FloatTensor(self.memory['states']).to(self.device))
        entropy_loss = current_dist.entropy().mean()

        # Total Loss
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_beta * entropy_loss

        # --- 4. Perform Optimization Step ---
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient Clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()

        # --- 5. Clear Memory for Next Episode ---
        self.clear_memory()

    def clear_memory(self):
        self.memory = {key: [] for key in self.memory}

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))import gymnasium as gym

