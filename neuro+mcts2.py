import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import numpy as np
import torch
torch.mps.empty_cache()

import json
import math
import seaborn as sns
import random

from sklearn.ensemble import RandomForestRegressor  # Simple neurosymbolic model
from matplotlib  import colors 
from matplotlib import pyplot as plt
from dsl2 import convert_np_to_native
from dsl import find_objects , PRIMITIVE
from neurosymbolic_reinforce import NeuralSymbolicSolverRL ,FeatureExtractor
# from neurosymbolic_RL_A2C import NeuralSymbolicSolverRL_A2C as neural
from nuerosymb_q_learning import DQN_Solver  as neural

import torch.nn as nn
import torch.nn.functional as F
REWARDS=[]

def move_up(grid, position):
    grid = grid.copy()
    x, y = position
    if x > 0 and grid[x - 1, y] == 0:
        grid[x, y], grid[x - 1, y] = 0, grid[x, y]
        return grid, (x - 1, y)
    return grid, position

def move_down(grid, position):
    grid = grid.copy()
    x, y = position
    if x < grid.shape[0] - 1 and grid[x + 1, y] == 0:
        grid[x, y], grid[x + 1, y] = 0, grid[x, y]
        return grid, (x + 1, y)
    return grid, position

def move_left(grid, position):
    grid = grid.copy()
    x, y = position
    if y > 0 and grid[x, y - 1] == 0:
        grid[x, y], grid[x, y - 1] = 0, grid[x, y]
        return grid, (x, y - 1)
    return grid, position

def c(grid, position):
    grid = grid.copy()
    x, y = position
    if y < grid.shape[1] - 1 and grid[x, y + 1] == 0:
        grid[x, y], grid[x, y + 1] = 0, grid[x, y]
        return grid, (x, y + 1)
    return grid, position


def idle (grid,position):
    return grid ,position

ACTIONS = {
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_left": move_left ,
    'idle':idle
}



# This gives you a list of names to index from
ACTION_NAMES = list(ACTIONS.keys())
num_action=len(ACTIONS)
PRIMITIVE_NAMES= list(PRIMITIVE.keys())




    
    
# MCTS Node for arrangement search
class MCTSNode:
    def __init__(self, objects, output_grid, background, parent=None):
        self.objects = objects  # Unplaced objects
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
                    while self.output_grid.shape[0] - th < 0 or self.output_grid.shape[1] - tw < 0:
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

    def simulate(self, neuro, spacial, target_grid, max_steps=3):
        """Run transformation and movement simulations"""
        total_reward = 0
        
        for obj_id, obj_info in self.arrangement.items():
        # if not obj_info["transformed"]:
            # Transform object using neuro agent
            current_grid = obj_info["grid"]
            target_region = extract_target_region(target_grid, obj_info)
              

            # Adjust position using spacial agent
            current_pos = obj_info["position"]
            for _ in range(max_steps):
                # Get movement action
                action_idx = spacial.select_action([
                    spacial._preprocess_to_tensor(self.output_grid),
                    spacial._preprocess_to_tensor(target_grid)
                ])

                action = ACTION_NAMES[action_idx]
                # Apply movement
                new_grid, new_pos = ACTIONS[action](self.output_grid.copy(), current_pos)
                # problm occur sometimes?
                # Calculate reward
                old_sim = matrix_similarity(
                    self.output_grid,
                    target_grid, 
                )
                new_sim = matrix_similarity(
                    new_grid,
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

                current_pos = new_pos
                self.output_grid = new_grid

            h, w = obj_info["grid"].shape
            r, c = obj_info["position"]

            # Ensure the slice won't exceed the output grid size
            max_r = min(r + h, self.output_grid.shape[0])
            max_c = min(c + w, self.output_grid.shape[1])

            # Crop the object grid if it would overflow
            obj_crop = obj_info["grid"][:max_r - r, :max_c - c]
            region = self.output_grid[r:max_r, c:max_c]

            self.output_grid[r:max_r, c:max_c] = np.where(
                obj_crop != self.background,
                obj_crop,
                region
            )


            for _ in range(max_steps):
                # Get transformation action
                action_idx = neuro.select_action([
                    neuro._preprocess_to_tensor(current_grid),
                    neuro._preprocess_to_tensor(target_region)
                ])
                # print(action_idx)
                primitive = PRIMITIVE_NAMES[action_idx]
                
                # Apply transformation
                new_grid = PRIMITIVE[primitive](current_grid.copy())
                
                # Calculate reward
                old_sim = matrix_similarity(current_grid, target_region)
                new_sim = matrix_similarity(new_grid, target_region)
                reward = new_sim - old_sim
                total_reward += reward
                
                # Store experience for neuro agent
                # neuro.store_reward(reward)
                # Define the state and next_state as tuples
                state = (current_grid.copy(), target_region.copy())
                next_state = (new_grid.copy(), target_region.copy())

                # Pass the tuples to the store_experience method
                neuro.store_experience(state, action_idx, reward, next_state)

                # print(primitive,reward)
                current_grid = new_grid
                
             
            # Update object info
            obj_info["grid"] = current_grid
            obj_info["transformed"] = True

            h, w = obj_info["grid"].shape
            r, c = obj_info["position"]

            self.output_grid[r:r+h, c:c+w] = np.where(
                obj_info["grid"] != self.background,
                obj_info["grid"],
                self.output_grid[r:r+h, c:c+w]
            )
        print(total_reward)
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

def extract_target_region(target_grid, obj_info):
    r, c = obj_info['position']
    obj_h, obj_w = obj_info['grid'].shape
    
    # Handle edge cases where object extends beyond target grid
    pad_h = max(0, r + obj_h - target_grid.shape[0])
    pad_w = max(0, c + obj_w - target_grid.shape[1])
    
    if pad_h > 0 or pad_w > 0:
        padded_target = np.pad(target_grid, 
                              ((0, pad_h), (0, pad_w)),
                              mode='constant',
                              constant_values=0)#use background
        return padded_target[r:r+obj_h, c:c+obj_w]
    return target_grid[r:r+obj_h, c:c+obj_w]



def extract_region(grid, position, obj_shape):
    grid = np.atleast_2d(grid)
    r, c = position
    h, w = obj_shape

    grid_rows, grid_cols = grid.shape
    max_r = min(r + h, grid_rows)
    max_c = min(c + w, grid_cols)
    return grid[r:max_r, c:max_c]

def get_object_position(node, obj):
    obj_id = id(obj)
    if obj_id in node.arrangement:
        return node.arrangement[obj_id]
    else:
        return None  
    
def matrix_similarity(a, b): #param

    if a.shape != b.shape:
        return 0.0
    
    value_sim = np.sum(a == b) / a.size  # Fraction of identical elements
    
    # Handle edge cases for small matrices (gradient requires ≥2 elements per axis)
    if a.shape[0] >= 2 and a.shape[1] >= 2:
        grad_a = np.abs(np.gradient(a)[0]) + np.abs(np.gradient(a)[1])  # Gradient magnitude for `a`
        grad_b = np.abs(np.gradient(b)[0]) + np.abs(np.gradient(b)[1])  # Gradient magnitude for `b`
    else:
        grad_a = np.zeros_like(a)
        grad_b = np.zeros_like(b)
    
    struct_sim = 1.0 - np.abs(grad_a - grad_b).mean()
    
    return 0.7 * value_sim + 0.3 * struct_sim



def arrange_objects_mcts(input_grid, output_grid,save=False,load=False, iterations=500):
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

    # neuro = NeuralSymbolicSolverRL_A2C(PRIMITIVE_NAMES,model)
    # spacial = NeuralSymbolicSolverRL_A2C(ACTION_NAMES,model)

    neuro_feature_extractor = FeatureExtractor(input_channels=1)
    spacial_feature_extractor = FeatureExtractor(input_channels=1)

    # Initialize each agent with its own, un-shared model
    neuro = neural(PRIMITIVE_NAMES, neuro_feature_extractor)
    spacial = neural(ACTION_NAMES, spacial_feature_extractor)
    if load:
     neuro.policy_net.load_state_dict(torch.load('neuro_model.pth'))
     spacial.policy_net.load_state_dict(torch.load('spacial_model.pth'))
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
        
        # Expansion
        if not node.is_terminal():
            node.expand()
            if node.children:  # Only move to child if expansion succeeded
                node = random.choice(node.children)
        
        # Simulation - only evaluate terminal nodes
        if node.is_terminal():
            a=1
        reward = node.simulate(neuro, spacial, output_grid)
        
        # Track best terminal node
        node_score = reward #/ (1 + node.visits)  # Prevent division by zero
        if node_score > best_score:
            best_score = node_score
            best_terminal_node = node
        # else:
        #     reward = 0  # Non-terminal nodes get neutral reward
        
        # Backpropagation
        current = node
        while current:
            current.visits += 1
            current.total_score += reward
            current = current.parent
    
    # Update policies
    neuro.update_policy()
    spacial.update_policy()
    
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
     torch.save(neuro.policy_net.state_dict(), 'neuro_model.pth')    
     torch.save(spacial.policy_net.state_dict(), 'spacial_model.pth')
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



if __name__ == '__main__':
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    ids=[]
    train_path='arc-prize-2025/arc-agi_training_challenges.json'
    with open(train_path, 'r') as f:
        train = json.load(f)

    for case_id in train:
        ids.append(case_id) 
    count=0
    for case_id in train:
        # count +=1
        # if count ==2 :
            # break
        for i in range(2):

            for j in ('input','output'):
                a=train[case_id]['train'][i]['input']
                b=train[case_id]['train'][i]['output']
        # print('input')
        # sns.heatmap(a,cmap=cmap)
        # plt.show()

        # a=np.array(a)
        # b=np.array(b)



        # Solve the puzzle using the new method
        solved_grid ,_= arrange_objects_mcts(a, b,save=True,load=True,iterations=10000)
        solved_grid = convert_np_to_native(solved_grid)

        print('original',b)
        print('predicted',solved_grid)

        # print('original')
        # sns.heatmap(b,cmap=cmap)
        # plt.show()

        # print('predicted')
        # sns.heatmap(solved_grid,cmap=cmap)
        # plt.show()
        
        # plt.plot(REWARDS)



            # Verify if the solution is correct
        is_correct = np.array_equal(solved_grid, b)
        if is_correct:
            print(f"\nSolution is correct: {is_correct}")   



