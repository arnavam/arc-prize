import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import pickle
import time
from typing import List, Tuple, Any
import itertools
from collections import Counter
import logging
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from helper_env import place_object, coordinate_converter
from helper_arc import display, clear
from helper_env import placement
from dsl import ALL_ACTIONS, SHIFT_ACTIONS, TRANSFORM_ACTIONS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/dataset_generator.log', mode='w')
logger.addHandler(handler)
logger.propagate = False

Shift_Actions = SHIFT_ACTIONS.keys()
Transform_Actions = TRANSFORM_ACTIONS.keys()
action_names = list(ALL_ACTIONS.keys())
action_counter = Counter()

@dataclass
class GridObject:
    grid: np.ndarray
    color: int
    size: tuple
    position: tuple = (0, 0)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

def create_random_object(max_size=3, max_color=9) -> GridObject:
    size = (random.randint(1, max_size), random.randint(1, max_size))
    color = random.randint(1, max_color)
    obj_grid = np.full(size, color)
    return GridObject(grid=obj_grid, color=color, size=size, position=(0, 0))

def find_empty_spot(grid, obj_size) -> Tuple[int, int]:
    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_size
    possible_spots = []
    for y in range(grid_h - obj_h + 1):
        for x in range(grid_w - obj_w + 1):
            if np.all(grid[y:y+obj_h, x:x+obj_w] == 0):
                possible_spots.append((y, x))
    return random.choice(possible_spots) if possible_spots else None

def all_pair_combinations(a, b):
    pool = list(itertools.product(range(a), range(b)))
    while True:
        for pair in np.random.permutation(pool):
            yield pair

def generate_tasks(num_simple_tasks=10, num_intermediate_tasks=10, grid_size=(10, 10), 
                   num_bg_objects=5, simple_examples_per_task=4, intermediate_examples_per_task=16):
    """Generate both simple and intermediate tasks combined - using original approach"""
    all_input_grids = []
    all_obj_grids = []
    all_target_grids = []
    all_obj_positions = []
    all_action_labels = []
    
    # Generate simple tasks (same as original generate_simple_task but with dataclass)
    for _ in range(num_simple_tasks):
        objects = []
        obj_labels = []
        action_labels = []
        target_grid = np.zeros(grid_size, dtype=int)

        # Create background objects
        for _ in range(num_bg_objects):
            objects.append(create_random_object())
        
        i = 0
        while i < simple_examples_per_task:
            obj_idx = random.randint(0, num_bg_objects - 1)
            obj = objects[obj_idx]
            
            position = find_empty_spot(target_grid, obj.size)
            
            if position:
                current_grid = target_grid.copy()
                # Use the original approach - pass obj.grid directly to place_object
                target_grid = place_object(target_grid, obj.grid, position)
               
                if not np.array_equal(target_grid, current_grid):
                    # Convert to DataLoader format directly
                    obj_grid_for_dataloader = place_object(np.zeros_like(target_grid), obj.grid, position)
                    
                    all_input_grids.append(current_grid)
                    all_obj_grids.append(obj_grid_for_dataloader)
                    all_target_grids.append(target_grid.copy())
                    all_obj_positions.append(position)
                    all_action_labels.append(action_names.index('place'))
                    
                    obj.position = position
                    i += 1

    # Generate intermediate tasks (same as original generate_intermediate_task but with dataclass)
    for _ in range(num_intermediate_tasks):
        objects = []
        obj_labels = []
        action_labels = []
        target_grid = np.zeros(grid_size, dtype=int)
        
        # Place initial objects
        i = 0
        while i < num_bg_objects:
            obj = create_random_object()
            pos = find_empty_spot(target_grid, obj.size)
            if pos:
                target_grid = place_object(target_grid, obj.grid, pos)
                obj.position = pos
                objects.append(obj)
                i += 1

        object_action_combinations = all_pair_combinations(num_bg_objects, len(action_names))
        i = 0

        while i < intermediate_examples_per_task:
            obj_idx, action_idx = next(object_action_combinations)
            obj = objects[obj_idx]
            action_name = action_names[action_idx]
            new_obj = copy.deepcopy(obj)

            if action_name in ['place', 'remove']:
                continue

            new_target_grid = None
            if action_name in Transform_Actions:
                # Use the original approach - apply action to grid directly
                new_obj.grid = ALL_ACTIONS[action_name](obj.grid)
                # Use placement function as in original code
                if new_obj.grid.size == 0 or 0 in new_obj.grid.shape:
                        continue  
                new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)
            elif action_name in Shift_Actions:
                # Use the original approach - apply action to position directly
                new_obj.position = ALL_ACTIONS[action_name](obj.position)
                new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)

            if new_target_grid is not None:
                # Convert to DataLoader format directly
                obj_grid_for_dataloader = place_object(np.zeros_like(target_grid), new_obj.grid, new_obj.position)
                
                all_input_grids.append(normalize(new_target_grid))
                all_obj_grids.append(normalize(obj_grid_for_dataloader))
                all_target_grids.append(normalize(target_grid.copy()))
                all_obj_positions.append(new_obj.position)
                all_action_labels.append(action_idx)
                i += 1

    return all_input_grids, all_obj_grids, all_target_grids, all_obj_positions, all_action_labels

def normalize_grid(grid_data):
    if isinstance(grid_data, list):
        grid_data = np.array(grid_data)  # Convert list to numpy array first
    grid_tensor = torch.tensor(grid_data, dtype=torch.float32)
    return grid_tensor / 10.0  

def create_dataset(create=True, **kwargs):
    """Create or load the complete dataset with both simple and intermediate tasks"""
    if create:
        tasks = generate_tasks(**kwargs)
        
        with open("generated_training_data.pkl", "wb") as f:
            pickle.dump((tasks), f)
    else:
        with open("generated_training_data.pkl", "rb") as f:
            tasks = pickle.load(f)
    
    return tasks

class GridDataset(Dataset):
    def __init__(self, input_grids, obj_grids, target_grids, obj_positions, action_labels):
        self.input_grids = input_grids
        self.obj_grids = obj_grids
        self.target_grids = target_grids
        self.obj_positions = obj_positions
        self.action_labels = action_labels
        
    def __len__(self):
        return len(self.action_labels)
    
    def __getitem__(self, idx):
        input_grid = torch.tensor(self.input_grids[idx], dtype=torch.float32)
        obj_grid = torch.tensor(self.obj_grids[idx], dtype=torch.float32)
        target_grid = torch.tensor(self.target_grids[idx], dtype=torch.float32)
        obj_position = torch.tensor(self.obj_positions[idx], dtype=torch.long)
        action_label = torch.tensor(self.action_labels[idx], dtype=torch.long)
        
        return input_grid, obj_grid, target_grid, obj_position, action_label

if __name__ == "__main__":
    # Create dataset with both simple and intermediate tasks
    dataset = create_dataset(
        create=True,
        num_simple_tasks=10,
        num_intermediate_tasks=30,
        grid_size=(10, 10),
        num_bg_objects=5,
        simple_examples_per_task=5,
        intermediate_examples_per_task=10
    )
    

            # action_labels = torch.tensor(action_labels).to(device)
            # pos_labels = torch.tensor(pos_labels).to(device)
            # current_grids = normalize_grid(current_grids).to(device)
            # obj_grids = normalize_grid(obj_grids).to(device)
            # target_grids = normalize_grid(target_grids).to(device)

    print(f"Input grids shape example: {len(dataset[0])}")
    print(f"Action labels distribution: {Counter(dataset[4])}")