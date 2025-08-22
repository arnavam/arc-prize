import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
import time
import numpy as np
import torch
if torch.mps.is_available(): torch.mps.empty_cache() 
elif torch.cuda.is_available(): torch.cuda.empty_cache()
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from A_arc import  loader , display
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to a file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)



import json
import math
import random

from matplotlib  import colors 
from dsl2 import convert_np_to_native
from dsl import find_objects , PRIMITIVE , ACTIONS , extract_target_region
from env import placement , matrix_similarity
from RL_alg.reinforce import NeuralSymbolicSolverRL ,FeatureExtractor
# from neurosymbolic_RL_A2C import NeuralSymbolicSolverRL_A2C as neural
from RL_alg.BaseDQN import BaseDQN  as neural

import torch.nn as nn
import torch.nn.functional as F
REWARDS=[]

    
# MCTS Node for arrangement search
class MCTSNode:
    def __init__(self, objects, output_grid, background, parent=None):
        self.objects = objects  
        self.arrangement = {}  # {object_id: {"position": (r,c), "grid": np.array, "transformed": False}}
        self.output_grid = output_grid
        self.background = background
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_score = 0.0
        # self.scorer = parent.scorer if parent else ArrangementScorer()
        self.depth = parent.depth + 1 if parent else 0 
        
    def expand(self):
        # Option 1: Skip placement
        skip_node = MCTSNode(
            self.objects.copy(),
            self.output_grid.copy(),
            self.background,
            self
        )
        skip_node.arrangement = self.arrangement.copy()
        self.children.append(skip_node)
        num_random_placements = 1 # You can tune this number


        # Option 2: Place each object at valid positions
        for obj in self.objects:
            obj_grid = obj['grid']
            th, tw = obj_grid.shape
            for _ in range(num_random_placements):
                # Check if any valid placement is possible
                    if self.output_grid.shape[0] - th < 0 or self.output_grid.shape[1] - tw < 0:
                        continue
                # Generate a random valid position
                    r = random.randrange(self.output_grid.shape[0] - th + 1)
                    c = random.randrange(self.output_grid.shape[1] - tw + 1)
                    # Create new node with object placed
                    new_objects = self.objects#[o for o in self.objects if o != obj]
                    new_node = MCTSNode(
                        new_objects,
                        self.output_grid.copy(),
                        self.background,
                        self
                    )
                    new_node.arrangement = self.arrangement.copy()
                    new_node.arrangement[id(obj)] = {
                        "position": (r, c),
                        "grid": obj_grid,
                        "transformed": False
                    }
                    # Update grid with object
                    new_node.output_grid[r:r+th, c:c+tw] = np.where(
                        obj_grid != self.background, 
                        obj_grid, 
                        new_node.output_grid[r:r+th, c:c+tw]
                    )
                    self.children.append(new_node)

    def simulate(self, neuro, spacial, target_grid, max_steps=4):
        """Run transformation and movement simulations"""
        total_reward = 0
        
        for obj_id, obj_info in self.arrangement.items():
        # if not obj_info["transformed"]:


            # Adjust position using spacial agent
            for _ in range(max_steps):
                # Get movement action


                action_idx = spacial.select_action([
                    self.output_grid,
                    target_grid
                ])
                func=spacial.actions[spacial.action_names[action_idx]]
                # print(func)
                old_sim = matrix_similarity(
                    self.output_grid,
                    target_grid, 
                )

                new_obj_info = obj_info.copy()
                new_obj_info['position']=func(obj_info['position'])
                # Calculate reward
                new_grid =placement(self.output_grid, obj_info, new_obj_info, background=0)
                if  new_grid is None:
                    max_steps+=1
                    continue

                new_sim = matrix_similarity(
                    self.output_grid,
                    target_grid
                )
                reward = new_sim - old_sim
                total_reward += reward
                # Store experience for spacial agent
                # spacial.store_reward(reward)
                state = (self.output_grid.copy(), target_grid.copy())
                next_state = (new_grid.copy(), target_grid.copy())

                # Pass the tuples to the store_experience method
                spacial.store_experience(state, action_idx, reward, next_state)

                # print(action,total_reward)

                current_pos = new_obj_info['position']
                self.output_grid = new_grid



            current_obj = obj_info["grid"]

            target_region = extract_target_region(target_grid, obj_info)
            for _ in range(max_steps):
                # Get transformation action
                
                action_idx = neuro.select_action([
                    current_obj,
                    target_region
                ])
                func=neuro.actions[neuro.action_names[action_idx]]
                new_obj_info =obj_info.copy()
                new_obj_info['grid']=func(obj_info['grid'])
                # Calculate reward
                old_sim = matrix_similarity(current_obj, target_region)
                new_grid =placement(self.output_grid, obj_info, new_obj_info, background=0)
                if  new_grid is None:
                    max_steps+=1
                    continue


                current_obj=extract_target_region(target_grid,new_obj_info)

                new_sim = matrix_similarity(new_obj_info['grid'], current_obj)
                reward = new_sim - old_sim
                total_reward += reward

                # Define the state and next_state as tuples
                state = (current_obj.copy(), target_region.copy())
                next_state = (new_obj_info['grid'].copy(), target_region.copy())

                # Pass the tuples to the store_experience method
                neuro.store_experience(state, action_idx, reward, next_state)

                # print(primitive,reward)
                current_obj = new_obj_info['grid']
                
             
            # Update object info

        print('total reward',total_reward)
        REWARDS.append(total_reward)
        return total_reward
            
    def ucb_score(self, exploration=0.4):# param
            if self.visits == 0:
                return float('inf')
            return (self.total_score / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        
    def is_terminal(self):
            if len(self.objects) == 0:
                print('true')
                return True


    
    def backpropagate(self, score):
        node = self
        while node:
            node.visits += 1
            node.total_score += score
            node = node.parent



def arrange_objects_mcts(input_grid, output_grid,device='cpu',save=False,load=False, iterations=500):
    # Initialize
    objects = find_objects(input_grid)

    output_grid=np.array(output_grid)
    input_grid=np.array(input_grid)

    if input_grid.size < output_grid.size:
        objects.append({
                    'grid': input_grid,
                    'color': 0,
                    'position': (output_grid.shape[0]//2, output_grid.shape[1]//2),
                    'size': (len(input_grid), len(input_grid[0]))
                })

    
    flat_output = np.array(output_grid).flatten()
    background = np.bincount(flat_output).argmax()    
    root = MCTSNode(objects, np.zeros_like(output_grid), background)

    # model = FeatureExtractor(input_channels=1)

    neuro_feature_extractor = FeatureExtractor(input_channels=1)
    spacial_feature_extractor = FeatureExtractor(input_channels=1)

    # Initialize each agent with its own, un-shared model
    neuro = neural(PRIMITIVE, neuro_feature_extractor,device=device)
    spacial = neural(ACTIONS, spacial_feature_extractor,device=device)
    if load:
     neuro.load()
     spacial.load()
        # Save trained policy networks

    # MCTS loop
    
    best_terminal_node = None
    best_score = -float('inf')
    # MCTS loop
    for _ in range(iterations):
        count =0

        # if best_terminal_node is None:
        node = root         
        # else:
            # node=best_terminal_node
        # Selection - traverse until leaf
        while node.children:
            count +=1
            node = max(node.children, key=lambda n: n.ucb_score())
        print('count', count )
        
        # Expansion
        if not node.is_terminal():
            node.expand()
            if node.children:  # Only move to child if expansion succeeded
                node = random.choice(node.children)
        
        # Simulation - only evaluate terminal nodes
        if node.is_terminal():
            print('node reached terminal!!')
        reward = node.simulate(neuro, spacial, output_grid)
        
        # Track best terminal node
        node_score = reward #/ (1 + node.visits)  # Prevent division by zero
        if node_score > best_score:
            best_score = node_score
            best_terminal_node = node

        neuro.update_policy()
        spacial.update_policy() 
        # Backpropagation
        current = node
        while current:
            current.visits += 1
            current.total_score += reward
            current = current.parent
    
    # Update policies

    
    # Fallback if no terminal node found
    if best_terminal_node is None:
        # Find deepest node as fallback
        best_terminal_node = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.depth > best_terminal_node.depth:
                best_terminal_node = node
            stack.extend(node.children)
    if save:
     neuro.save()    
     spacial.save()
    
    print(count)
    return best_terminal_node.output_grid, best_terminal_node



def solve(input_grid, background, max_size=30,):
    # Create blank target grid
    
    target_grid = np.zeros((max_size, max_size)) + background
    
    # Run MCTS with blank target
    solution, _ = arrange_objects_mcts(input_grid, target_grid)
    
    # Crop to content
    non_empty = np.where(solution != background)
    min_r, max_r = np.min(non_empty[0]), np.max(non_empty[0])
    min_c, max_c = np.min(non_empty[1]), np.max(non_empty[1])
    return solution[min_r:max_r+1, min_c:max_c+1]



def display(a, b, solved_grid):
    cmap = 'coolwarm'  # Example colormap

            # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot each heatmap on a separate subplot
    sns.heatmap(a, cmap=cmap, ax=axes[0], cbar=False)
    axes[0].set_title('Input')

    sns.heatmap(b, cmap=cmap, ax=axes[1], cbar=False)
    axes[1].set_title('Original')

    sns.heatmap(solved_grid, cmap=cmap, ax=axes[2], cbar=False)
    axes[2].set_title('Predicted')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"heatmap_{timestamp}.png"

    # Save the figure
    plt.savefig(f'fig/{filename}')

    # Optionally, print the filename to confirm
    print(f"Figure saved as {filename}")

    # Close the figure to free up memory
    plt.close()



if __name__ == '__main__':
    train_path='arc-prize-2025/arc-agi_training_challenges.json'

    train , ids = loader(train_path) 
    count=0
    win=0
    for case_id in train:
        count +=1
        if count ==2 :
            break
        for i in range(2):

            for j in ('input','output'):
                a=train[case_id]['train'][i]['input']
                b=train[case_id]['train'][i]['output']

        


        # Solve the puzzle using the new method
        start_time = time.time()
        solved_grid ,_= arrange_objects_mcts(a, b,device='cuda',save=True,load=True,iterations=10)
        solved_grid = convert_np_to_native(solved_grid)
        print(time.time()-start_time)
        logging.debug(f"count: {count}")
        logging.debug("Elapsed time: %f seconds", time.time() - start_time)
        logging.debug(f'original,{b}')
        logging.debug(f'predicted,{solved_grid}')
        
        display(a, b, solved_grid)

            # Verify if the solution is correct
        is_correct = np.array_equal(solved_grid, b)
        if is_correct:
            print(f"\nSolution is correct: {is_correct}")
            win +=1
            display(a, b, solved_grid)


            logging.debug(f'win,{win}')



