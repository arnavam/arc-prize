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
from helper_env import place_object , coordinate_converter
action_counter = Counter()

from helper_arc import display,clear
from dl_models.DQNAction_Classifier import DQN_Classifier
from dl_models.feature_extractor import FeatureExtractor
from dl_models.ReinLikelihood import Likelihood
# --- Helper Functions for Data Generation ---
from dsl import ALL_ACTIONS , SHIFT_ACTIONS , TRANSFORM_ACTIONS
from helper_env import placement

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/Arc_Prize_pretraining.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False


Shift_Actions = SHIFT_ACTIONS.keys()
Transform_Actions = TRANSFORM_ACTIONS.keys()
action_names=list(ALL_ACTIONS.keys())


def create_random_object(max_size=3, max_color=9):

    size = (random.randint(1, max_size), random.randint(1, max_size))
    color = random.randint(1, max_color)
    obj_grid = np.full(size, color)
    return {'grid': obj_grid, 'color': color, 'size': size,'position':(0,0)}
    # return obj_grid

def find_empty_spot(grid, obj_size):

    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_size
    possible_spots = []
    for y in range(grid_h - obj_h + 1):
        for x in range(grid_w - obj_w + 1):
            if np.all(grid[y:y+obj_h, x:x+obj_w] == 0):
                possible_spots.append((y, x))
    return random.choice(possible_spots) if possible_spots else None


 
# --- Task-Specific Generation Functions ---
def generate_simple_task(grid_size=(10, 10), num_bg_objects=3, training_eg=4):
    inputs = []
    objects=[]
    obj_labels=[]
    action_labels=[]
    target_grid = np.zeros(grid_size, dtype=int)

    for _ in range(num_bg_objects):
        objects.append(create_random_object())
    
    i=0
    while i < training_eg:

        obj_idx= random.randint(0,num_bg_objects-1) # choose one random object from the lsit
        obj=objects[obj_idx]
        
        position = find_empty_spot(target_grid, obj['size']) # looks if the obj can be placed on target
        
        if position: # if can be placed

            current_grid=target_grid.copy()
            target_grid = place_object(target_grid, obj['grid'], position)
           
            if np.array_equal(target_grid, current_grid):
                print('grid: \n' , current_grid,'\n',target_grid)
                continue
                # raise ValueError("Both grids can't be the same")
        
            inputs.append((current_grid,objects,target_grid.copy(),position))
            obj_labels.append(obj_idx)       
            action_labels.append(action_names.index('place'))

            obj['position']=position
            



            i +=1

    return inputs , obj_labels , action_labels


def generate_intermediate_task(grid_size=(10, 10), num_bg_objects=5, training_eg=4):
    inputs = []
    objects=[]
    labels=[]
    funcs=[]
    target_grid = np.zeros(grid_size, dtype=int)
    
    i = 0
    while i < num_bg_objects: # place random objects  in  target_grid
        
        obj=create_random_object()
        pos = find_empty_spot(target_grid, obj['size'])

        if pos:
            target_grid = place_object(target_grid, obj, pos)
            obj['position']=pos
            objects.append(obj)
            i+=1

    object_action_combinations = all_pair_combinations(num_bg_objects,len(action_names) ) 
    i = 0

    while i < training_eg:

        new_target_grid=None

        obj_idx,action_idx=next(object_action_combinations)
        obj=objects[obj_idx]

        action_name = random.choice(action_names)
        action_idx = action_names.index(action_name)
        new_obj = copy.deepcopy(obj)


        if action_name in ['place', 'remove']: # do nothing for this two actions
            continue 


        elif action_name in Transform_Actions:
            new_obj['grid'] = ALL_ACTIONS[action_name](obj['grid']) # performs action on the grid
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0) # update the obj on the grid

        elif action_name in Shift_Actions:
            new_obj['position'] = ALL_ACTIONS[action_name](obj['position']) # peforms action on the position
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)# update the obj on the grid

        if new_target_grid is not  None: # the update has happend
            i+=1
            inputs.append((new_target_grid,objects,target_grid.copy(),new_obj['position']))
            
            obj_labels.append(obj_idx)
            action_labels.append(action_idx)

    return inputs, obj_labels, action_labels


def all_pair_combinations(a, b):
    pool = list(itertools.product(range(a), range(b)))
    while True:
        for pair in np.random.permutation(pool):
            yield pair



def create_data_loader(tasks: List[Tuple], batch_size: int, shuffle: bool = True,printing=False):
 
    
    all_examples:      List[Any] = []
    all_input_grids =[]
    all_obj_grids=[]
    all_target_grids=[]
    all_obj_labels:    List[int] = []
    all_action_labels: List[int] = []
    inputs=[]
    for all_inputs, obj_labels, action_labels in tasks:
        for (input_grid, objects, target_grid , obj_pos) ,obj_label ,action_idx in zip(all_inputs, obj_labels ,action_labels):
            
            # place object inside a target shaped input 
            obj = objects[obj_label]
            obj_grid = place_object(np.zeros_like(target_grid.copy()),obj['grid'],obj['position'])

            # all_examples.append([input_grid,obj_grid,target_grid])
            all_input_grids.append(input_grid)
            all_obj_grids.append(obj_grid)
            all_target_grids.append(target_grid)
            all_obj_labels.append(obj_pos)
            all_action_labels.append(action_idx)

            obj=objects[obj_label]
            action_counter[action_idx] += 1


            if printing == True:
                logger.debug(f"dataset: \n {input_grid} ,\n {obj['grid']} ,\n{target_grid},{action_names[action_idx]},")
                obj_grid= placement(np.zeros_like(target_grid),obj,obj,0)
                display(input_grid,obj_grid,target_grid,'data_loader') 

                

    # for finding no of each actions each timed it used it train
    print("Function counts:")
    for action_idx, count in action_counter.items():
        print(f"{action_names[action_idx]}: {count}")

    indices = list(range(len(all_action_labels))) #create and shuffle indices
    if shuffle:
        random.shuffle(indices)

    # Yield mini-batches one by one
    for i in range(0, len(indices), batch_size):
        # Get the indices for the current batch
        batch_indices = indices[i:i + batch_size]
        
        # Use the indices to get the data for the batch
        # batch_examples = [all_examples[j] for j in batch_indices]
        batch_target_grids = [all_target_grids[j] for j in batch_indices]
        batch_input_grids = [all_input_grids[j] for j in batch_indices]
        batch_obj_grids = [all_obj_grids[j] for j in batch_indices]
        batch_obj_labels = [all_obj_labels[j] for j in batch_indices]
        batch_action_labels = [all_action_labels[j] for j in batch_indices]

        yield batch_input_grids,batch_obj_grids,batch_target_grids, batch_obj_labels, batch_action_labels




def dataset_creater( create):
    if create==True:

        tasks1 = [      generate_simple_task(grid_size=(10, 10), num_bg_objects=5 , training_eg=4 ) for _ in range(10)] 
        tasks2 = [generate_intermediate_task(grid_size=(10, 10), num_bg_objects=5 , training_eg=16) for _ in range(10)]

        tasks1.extend(tasks2)

        with open("generated_training_data.pkl", "wb") as f:
            pickle.dump(tasks1, f) 

    else:
        with open("generated_training_data.pkl", "rb") as f:
            tasks1 = pickle.load(f)
    return tasks1

