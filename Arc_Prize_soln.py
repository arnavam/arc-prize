import gymnasium as gym
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from tqdm import tqdm
import math
from typing import  Tuple


from dl_models.mamba import MambaBlock ,ModelArgs
from helper_arc import get_module_logger , plot_metrics
from A2C import A2CAgent
def train():
    # --- Hyperparameters ---
    env_name = 'CartPole-v1'
    max_episodes = 2000
    max_timesteps = 500
    lr = 0.002
    gamma = 0.99
    entropy_beta = 0.01
    log_interval = 10
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup ---
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = A2CAgent(state_dim, action_dim, lr, gamma, entropy_beta, device)
    
    # For logging
    rewards_deque = deque(maxlen=100)
    running_reward = 0

    # --- Training ---
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            # The agent needs the raw state to compute the policy distribution later for entropy
            agent.memory['states'].append(state)

            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            
            agent.store_reward(reward, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Perform the update at the end of the episode
        agent.update()

        # --- Logging ---
        rewards_deque.append(episode_reward)
        running_reward = np.mean(rewards_deque)
        
        if episode % log_interval == 0:
            print(f'Episode {episode} \t Last Reward: {episode_reward:.2f} \t Average Reward: {running_reward:.2f}')
        
        # Check for solving condition
        if running_reward > env.spec.reward_threshold:
            print(f"\nEnvironment solved in {episode} episodes! Average reward: {running_reward:.2f}")
            agent.save_model(f'{env_name}-a2c-solved.pth')
            break
            
    env.close()

if __name__ == '__main__':
    train()