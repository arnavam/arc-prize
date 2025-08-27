import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import pickle
import time

from A_arc import display,clear
from rl_models.DQNAction_Classifier import DQN_Classifier
from rl_models.feature_extractor import FeatureExtractor
from rl_models.ReinLikelihood import Likelihood
# --- Helper Functions for Data Generation ---
from dsl import COMB , SHIFT , TRANSFORM
from env import placement

import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='Arc_Prize_pretraining.log',  # Log output to current_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

Shift = SHIFT.keys()
Transform = TRANSFORM.keys()
func_names=list(COMB.keys())
num_actions = len(func_names)


def create_random_object(max_size=3, max_color=9):

    size = (random.randint(1, max_size), random.randint(1, max_size))
    color = random.randint(1, max_color)
    obj_grid = np.full(size, color)
    return {'grid': obj_grid, 'color': color, 'size': size,'position':(0,0)}
    # return obj_grid

def find_empty_spot(grid, obj_size):
    """Finds a valid top-left (y, x) coordinate to place an object without overlap."""
    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_size
    possible_spots = []
    for y in range(grid_h - obj_h + 1):
        for x in range(grid_w - obj_w + 1):
            if np.all(grid[y:y+obj_h, x:x+obj_w] == 0):
                possible_spots.append((y, x))
    return random.choice(possible_spots) if possible_spots else None

def place_object(grid, obj, pos):
    """Places an object's grid onto the main grid at a given position."""
    y, x = pos
    obj_h, obj_w = obj['size']
    grid[y:y+obj_h, x:x+obj_w] = obj['grid']
    return grid

# --- Task-Specific Generation Functions ---

def generate_simple_task(grid_size=(10, 10), num_bg_objects=3, training_eg=4):

    datasets = []
    objects=[]
    labels=[]
    funcs=[]
    target_grid = np.zeros(grid_size, dtype=int)

    for _ in range(num_bg_objects):
        objects.append(create_random_object())

    for _ in range(training_eg):

        obj_idx= random.randint(0,num_bg_objects-1)
        obj=objects[obj_idx]
        current_grid=target_grid
        pos = find_empty_spot(target_grid, obj['size'])
        if pos:
            target_grid = place_object(target_grid, obj, pos)
            datasets.append((current_grid,objects,target_grid))
            labels.append(obj_idx)
            funcs.append(func_names.index('place'))

        else:
            training_eg +=1

    return datasets , labels , funcs

def generate_intermediate_task(grid_size=(10, 10), num_bg_objects=5, training_eg=4):
    datasets = []
    objects=[]
    labels=[]
    funcs=[]
    target_grid = np.zeros(grid_size, dtype=int)

    i = 0
    while i < num_bg_objects:
        
        obj=create_random_object()
        pos = find_empty_spot(target_grid, obj['size'])

        if pos:
            target_grid = place_object(target_grid, obj, pos)
            obj['position']=pos
            objects.append(obj)
            i+=1
            
    i = 0
    while i < training_eg:

        new_target_grid=None

        obj_idx = random.randint(0, num_bg_objects - 1)
        obj=objects[obj_idx]

        func = random.choice(func_names)
        func_idx = func_names.index(func)
        new_obj = copy.deepcopy(obj)

        
        if func in ['place', 'remove']:
            continue 
    #     new_obj_info.update({
    #         'placed': True,
    #         'position': pos_values
    #     })
    #         is_place_action = True
    #         objects.append(new_obj_info)
    #         obj_info = new_obj_info

        elif func in Transform:
            new_obj['grid'] = COMB[func](obj['grid'])
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)

        elif func in Shift:
            new_obj['position'] = COMB[func](obj['position'])
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)

        if new_target_grid is not  None:
            i+=1
            logging.debug(f"dataset: {new_target_grid},{target_grid},{obj['grid']},{obj_idx},{i}")
            display(new_target_grid,target_grid,obj['grid'],'train_examples')
            datasets.append((new_target_grid,objects,target_grid))
            labels.append(obj_idx)
            funcs.append(func_idx)

    return datasets,labels, funcs




# --- Main  Function ---

if __name__ == '__main__':
    clear('train_examples')

    ft1 = FeatureExtractor(input_channels=1)
    likelihood_predictor = Likelihood(feature_extractor=ft1, output_dim=1,no_of_inputs=3)
    likelihood_predictor.show_structure()
    nuero_classifier = DQN_Classifier(ft1,len(COMB),3)
    # likelihood_predictor.load()
    # nuero_classifier.load()
    tasks1 = [generate_intermediate_task(grid_size=(10, 10), num_bg_objects=10 , training_eg=16) for _ in range(10)]
    tasks2= [generate_simple_task(grid_size=(10, 10), num_bg_objects=4 , training_eg=16) for _ in range(10)]

    tasks1.extend(tasks2)
    random.shuffle(tasks1)
# Save
    with open("generated_training_data.pkl", "wb") as f:
        pickle.dump(tasks1, f)

    # Load
    # with open("data.pkl", "rb") as f:
    #     tasks1 = pickle.load(f)

    l_losses=[]
    l_accs=[]
    # Then train over those
    for epoch in range(100):
        for datasets, labels , funcs in tasks1:
            start_time=time.time()
            loss,acc=likelihood_predictor.train_supervised(datasets,labels)
            print(f"Step {epoch+1}: losses = {loss:.4f}, Accuracy = {acc:.2%} , time= {time.time()-start_times}")
            
            l_losses.append(loss)
            l_accs.append(acc)
            loss,acc= nuero_classifier.train_supervised(datasets,labels,funcs)

    likelihood_predictor.save()
    plt.plot(l_accs, label='Accuracy')
    plt.plot(l_losses, label='l_losses')
    plt.xlabel("Step")
    plt.legend()
    plt.show()

    nuero_classifier.save()
    plt.plot(l_accs, label='Accuracy')
    plt.plot(l_losses, label='losses')
    plt.xlabel("Step")
    plt.legend()
    plt.show()
