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
from helper_arc import display, clear , get_module_logger
from helper_env import placement 
from dsl import ALL_ACTIONS, SHIFT_ACTIONS, TRANSFORM_ACTIONS

logger=get_module_logger('dataset_generator')

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

def is_valid_position(grid, obj_size, position):
    """Check if position is valid for placing object"""
    y, x = position
    obj_h, obj_w = obj_size
    grid_h, grid_w = grid.shape
    
    # Check bounds
    if y < 0 or x < 0 or y + obj_h > grid_h or x + obj_w > grid_w:
        return False
    
    # Check if area is empty
    return np.all(grid[y:y+obj_h, x:x+obj_w] == 0)

def normalize(grid):
    """Normalize grid values between 0 and 1"""
    return grid / 10.0

def generate_tasks(num_simple_tasks=10, num_intermediate_tasks=10, grid_size=(10, 10), 
                   num_bg_objects=5, simple_examples_per_task=4, intermediate_examples_per_task=16):
    """Generate both simple and intermediate tasks combined - with fixes for duplicates"""
    all_input_grids = []
    all_obj_grids = []
    all_target_grids = []
    all_obj_positions = []
    all_action_labels = []
    
    # Track seen examples to avoid duplicates
    seen_examples = set()
    
    # Generate simple tasks
    for task_idx in range(num_simple_tasks):
        objects = []
        input_grid = np.zeros(grid_size, dtype=int)

        # Create background objects
        for _ in range(num_bg_objects):
            objects.append(create_random_object())
        
        examples_generated = 0
        attempts = 0
        max_attempts = simple_examples_per_task * 10  # Prevent infinite loop
        
        while examples_generated < simple_examples_per_task and attempts < max_attempts:
            obj_idx = random.randint(0, num_bg_objects - 1)
            obj = objects[obj_idx]
            
            position = find_empty_spot(input_grid, obj.size)
            
            if position:
               

                target_grid = place_object(input_grid.copy(), obj.grid, position)
               
                if not np.array_equal(input_grid, target_grid):
                    # Create unique identifier for this example
                    example_id = (tuple(input_grid.flatten()), tuple(obj.grid.flatten()), position, 'place')
                    
                    if example_id not in seen_examples:
                        # Convert to DataLoader format directly
                        obj_grid_for_dataloader = place_object(np.zeros_like(input_grid), obj.grid, position)
                        
                        all_input_grids.append(input_grid.copy())
                        all_obj_grids.append(obj_grid_for_dataloader)
                        all_target_grids.append(target_grid.copy())
                        all_obj_positions.append(position)
                        all_action_labels.append(action_names.index('place'))
                        
                        seen_examples.add(example_id)
                        examples_generated += 1
                        input_grid = target_grid.copy()  # Update target grid
                        
                        # Update object position
                        obj.position = position
            
            attempts += 1

    # Generate intermediate tasks
    for task_idx in range(num_intermediate_tasks):
        objects = []
        input_grid = np.zeros(grid_size, dtype=int)
        
        # Place initial objects
        placed_objects = 0
        attempts = 0
        max_placement_attempts = num_bg_objects * 10
        
        while placed_objects < num_bg_objects and attempts < max_placement_attempts:
            obj = create_random_object()
            pos = find_empty_spot(input_grid, obj.size)
            if pos:
                input_grid = place_object(input_grid, obj.grid, pos)
                obj.position = pos
                objects.append(obj)
                placed_objects += 1
            attempts += 1

        examples_generated = 0
        attempts = 0
        max_attempts = intermediate_examples_per_task * 20

        while examples_generated < intermediate_examples_per_task and attempts < max_attempts:
            obj_idx = random.randint(0, len(objects) - 1)
            action_idx = random.randint(0, len(action_names) - 1)
            
            obj = objects[obj_idx]
            action_name = action_names[action_idx]
            new_obj = copy.deepcopy(obj)
            

            # Skip place and remove actions for intermediate tasks
            if action_name in ['place', 'remove']:
                attempts += 1
                continue

            elif action_name in Transform_Actions:
                # Apply transformation
                new_obj.grid = ALL_ACTIONS[action_name](obj.grid)
                
                # Skip if transformation results in invalid object
                if new_obj.grid.size == 0 or 0 in new_obj.grid.shape:
                    print('happend')
                    attempts += 1
                    continue
                    
                # Try to place transformed object
                target_grid = placement(input_grid.copy(), obj, new_obj, background=0)
                
            elif action_name in Shift_Actions:
                # Apply shift
                new_pos = ALL_ACTIONS[action_name](obj.position)
                new_obj.position = new_pos
                
                # Check if new position is valid
                # if is_valid_position(target_grid, new_obj.size, new_pos):
                target_grid = placement(input_grid.copy(), obj, new_obj, background=0)

            if np.array_equal(input_grid, target_grid) :
                display(input_grid,place_object(np.zeros_like(target_grid), obj.grid, obj.position),target_grid,input_title='image',predicted_title=new_obj.position,target_title=action_names[action_idx],folder='wrong_patterns')

            if (target_grid is not None and 
                not np.array_equal(input_grid, target_grid)):
                
                # Create unique identifier
                example_id = (tuple(target_grid.flatten()), 
                             tuple(new_obj.grid.flatten()), 
                             new_obj.position, 
                             action_name)
                
                if example_id not in seen_examples:
                    # Convert to DataLoader format
                    obj_grid = place_object(np.zeros_like(target_grid), obj.grid, obj.position)

                    display(input_grid,obj_grid,target_grid,input_title='image',predicted_title=new_obj.position,target_title=action_names[action_idx])
                    all_input_grids.append(input_grid.copy())
                    all_obj_grids.append(obj_grid)
                    all_target_grids.append(target_grid.copy())
                    all_obj_positions.append(new_obj.position)
                    all_action_labels.append(action_idx)
                    
                    seen_examples.add(example_id)
                    examples_generated += 1
                    # Update the target grid for next iteration
                    input_grid = target_grid.copy()
                    # Update the object in our list
                    objects[obj_idx] = new_obj
            
            attempts += 1

        logger.info(f"Intermediate task {task_idx}: Generated {examples_generated}/{intermediate_examples_per_task} examples")

    return all_input_grids, all_obj_grids, all_target_grids, all_obj_positions, all_action_labels

def normalize_grid(grid_data):
    if isinstance(grid_data, list):
        grid_data = np.array(grid_data)
    grid_tensor = torch.tensor(grid_data, dtype=torch.float32)
    return grid_tensor / 10.0  

def create_dataset(create=True, **kwargs):
    """Create or load the complete dataset with both simple and intermediate tasks"""
    if create:
        tasks = generate_tasks(**kwargs)
        
        with open("generated_training_data.pkl", "wb") as f:
            pickle.dump(tasks, f)
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

    input_grids, obj_grids, target_grids, obj_positions, action_labels = dataset
    # for x,y,z in zip(input_grids, obj_grids , target_grids):
    #     display(x,y,z)
    print(f"Total examples generated: {len(action_labels)}")
    print(f"Input grids shape example: {input_grids[0].shape if len(input_grids) > 0 else 'No examples'}")
    print(f"Action labels distribution: {Counter(action_labels)}")
    
    # Check for duplicates and same input/target
    same_input_target = 0
    for i in range(len(input_grids)):
        if np.array_equal(input_grids[i], target_grids[i]):
            same_input_target += 1
    
    print(f"Examples where input == target: {same_input_target}")
    
    # Check for duplicate examples
    example_set = set()
    duplicates = 0
    for i in range(len(input_grids)):
        example_id = (tuple(input_grids[i].flatten()), 
                     tuple(obj_grids[i].flatten()), 
                     obj_positions[i], 
                     action_labels[i])
        if example_id in example_set:
            duplicates += 1
        else:
            example_set.add(example_id)
    
    print(f"Duplicate examples: {duplicates}")