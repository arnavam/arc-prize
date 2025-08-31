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
from dl_models.Feature_Extractor import FeatureExtractor
from dl_models.ReinLikelihood import Likelihood
# --- Helper Functions for Data Generation ---
from dsl import ALL_ACTIONS , SHIFT_ACTIONS , TRANSFORM_ACTIONS
from helper_env import placement

import logging
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/Arc_Prize_pretraining.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False


Shift_Actions = SHIFT_ACTIONS.keys()
Transform_Actions = TRANSFORM_ACTIONS.keys()
action_names=list(ALL_ACTIONS.keys())
num_actions = len(action_names)



def create_random_object(max_size=3, max_color=9):
    size = (random.randint(1, max_size), random.randint(1, max_size))
    
    # Decide how many colors to use (1 to 3)
    num_colors = random.randint(2, min(3, max_color))
    
    # Randomly choose distinct colors
    colors = random.sample(range(1, max_color + 1), num_colors)
    
    # create grid with randomly assigned colors from the chosen set
    obj_grid = np.random.choice(colors, size=size)
    # print(obj_grid)
    return {
        'grid': obj_grid,
        'colors': colors,
        'size': size,
        'position': (None, None)
    }


def find_empty_spot(grid, obj_size):
    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_size
    possible_spots = []

    for y in range(grid_h - obj_h + 1):
        for x in range(grid_w - obj_w + 1):
            if np.all(grid[y:y+obj_h, x:x+obj_w] == 0):
                # Convert top-left to center position
                center_pos = coordinate_converter((y, x), obj_size, is_center=False)
                possible_spots.append(center_pos)

    return random.choice(possible_spots) if possible_spots else None


# --- Task-Specific Generation Functions ---
def generate_simple_task(grid_size=(10, 10), num_bg_objects=3, training_eg=4):
    datasets = []
    objects=[]
    obj_labels=[]
    action_labels=[]
    target_grid = np.zeros(grid_size, dtype=int)

    for _ in range(num_bg_objects):
        objects.append(create_random_object())
    i=0
    while i < training_eg:

        obj_idx= random.randint(0,num_bg_objects-1)
        obj=objects[obj_idx]
        
        position = find_empty_spot(target_grid, obj['size'])
        if position:

            current_grid=target_grid.copy()

            target_grid = place_object(target_grid, obj['grid'], position)
           
            if np.array_equal(target_grid, current_grid):
                print('grid: \n' , current_grid,'\n',target_grid)
                continue
                # raise ValueError("Both grids can't be the same")
        
            datasets.append((current_grid,objects,target_grid.copy(),position))
            obj_labels.append(obj_idx)       
            action_labels.append(action_names.index('place'))

            obj['position']=position
            



            i +=1

    return datasets , obj_labels , action_labels





def generate_intermediate_task(grid_size=(10, 10), num_bg_objects=5, training_eg=4):
    datasets = []
    objects=[]
    obj_labels=[]
    action_labels=[]
    target_grid = np.zeros(grid_size, dtype=int)
    i = 0
    while i < num_bg_objects:
        
        obj=create_random_object()
        position = find_empty_spot(target_grid, obj['size'])

        if position:
            logger.debug('generated_ineter')
            target_grid = place_object(target_grid, obj['grid'], position)
            obj['position']=position
            objects.append(obj)
            i+=1

    gen = infinite_random_pairs(num_bg_objects,len(action_names) )
    i = 0
    while i < training_eg:

        new_target_grid=None

        obj_idx,action_idx=next(gen)
        obj=objects[obj_idx]

        action_name = action_names[action_idx]
        new_obj = copy.deepcopy(obj)


        if action_name in ['place', 'remove']:
            continue 

        elif action_name in Transform_Actions:
            new_obj['grid'] = ALL_ACTIONS[action_name](obj['grid'])
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)

        elif action_name in Shift_Actions:
            new_obj['position'] = ALL_ACTIONS[action_name](obj['position'])
            new_target_grid = placement(target_grid.copy(), obj, new_obj, background=0)

        if new_target_grid is not  None:
            i+=1
            datasets.append((new_target_grid,objects,target_grid.copy(),new_obj['position']))
            
            obj_labels.append(obj_idx)
            action_labels.append(action_idx)

    return datasets, obj_labels, action_labels

def infinite_random_pairs(a, b):
    pool = list(itertools.product(range(a), range(b)))
    while True:
        for pair in np.random.permutation(pool):
            yield pair



def create_data_loader(tasks: List[Tuple], batch_size: int, shuffle: bool = True,printing=False):
    if printing ==True:
        clear('data_loader')    
    # 1. Pool all examples into a single dataset
    all_examples: List[Any] = []
    all_obj_labels:   List[int] = []
    all_action_labels:     List[int] = []

    for datasets, obj_labels, action_labels in tasks:
        for (input_grid,objects,target_grid,_position) , label , action_idx in zip(datasets,obj_labels,action_labels):
            obj=objects[label]
            action_counter[action_idx] += 1


            if printing == True:
                logger.debug(f"dataset: \n {input_grid} ,\n {obj['grid']} ,\n{target_grid},{action_names[action_idx]},")
                obj_grid= placement(np.zeros_like(target_grid),obj,obj,0)
                display(input_grid,obj_grid,target_grid,'data_loader') 

                
        all_examples.extend(datasets)
        all_obj_labels.extend(obj_labels)
        all_action_labels.extend(action_labels)

    # After the loop
    print("Function counts:")
    for action_idx, count in action_counter.items():
        print(f"{action_names[action_idx]}: {count}")
    # 2. create and shuffle indices
    indices = list(range(len(all_examples)))
    if shuffle:
        random.shuffle(indices)

    # 3. Yield mini-batches one by one
    for i in range(0, len(indices), batch_size):
        # Get the indices for the current batch
        batch_indices = indices[i:i + batch_size]
        
        # Use the indices to get the data for the batch
        batch_examples = [all_examples[j] for j in batch_indices]
        batch_obj_labels = [all_obj_labels[j] for j in batch_indices]
        batch_action_labels = [all_action_labels[j] for j in batch_indices]

        yield batch_examples, batch_obj_labels, batch_action_labels




def dataset_creater( create):
    if create==True:
        # clear('train_examples')
        tasks1 = [      generate_simple_task(grid_size=(10, 10), num_bg_objects=5 , training_eg=4 ) for _ in range(10)]
        tasks2 = [generate_intermediate_task(grid_size=(10, 10), num_bg_objects=5 , training_eg=16) for _ in range(10)]

        tasks1.extend(tasks2)

        with open("generated_training_data.pkl", "wb") as f:
            pickle.dump(tasks1, f)

    else:
        with open("generated_training_data.pkl", "rb") as f:
            tasks1 = pickle.load(f)
    return tasks1


# --- Main  Function ---
if __name__ == '__main__':
    # clear('likelihood_predictions')

    # clear('classifier_predictions')

    ft1 = FeatureExtractor(input_channels=1)
    likelihood_predictor = Likelihood(feature_extractor=ft1, output_dim=1,no_of_inputs=3)
    likelihood_predictor.show_structure()
    action_classifier = DQN_Classifier(ft1,len(ALL_ACTIONS),3)
    # likelihood_predictor.load()
    # action_classifier.load()

    tasks = dataset_creater(create=False)

    likelihood_losses=[]
    likelihood_accuracies=[]
    action_losses=[]
    action_accuracies=[]
    position_losses=[]

    # data_loader = create_data_loader(tasks, batch_size=10, shuffle=True)
    # datasets, obj_labels , action_labels = next(iter(data_loader))

    for epoch in range(10):

        data_loader = create_data_loader(tasks, batch_size=10, shuffle=True,printing=False)
        for datasets, obj_labels , action_labels in data_loader:
            start_time=time.time()

            loss,acc=likelihood_predictor.train_supervised(datasets,obj_labels)
            print(f"l Step {epoch+1}: losses = {loss:.4f}, Accuracy = {acc:.2%} , time= {time.time()-start_time}")
            likelihood_losses.append(loss)
            likelihood_accuracies.append(acc)

            a_loss,pos_loss,acc= action_classifier.train_supervised(datasets,obj_labels,action_labels)
            print(f"a Step {epoch+1}: losses = {loss:.4f}, Accuracy = {acc:.2%} , time= {time.time()-start_time}")
            action_losses.append(a_loss)
            position_losses.append(pos_loss)
            action_accuracies.append(acc)

    # likelihood_predictor.save()
    # action_classifier.save()

    plt.title('likelihood_predictor')
    plt.plot(likelihood_accuracies, label='Accuracy')
    plt.plot(likelihood_losses, label='losses')
    plt.xlabel("Step")
    plt.legend()
    plt.show()

    plt.title('action_classifier')
    plt.plot(action_accuracies, label='Accuracy')
    plt.plot(action_losses, label='action_loss')
    plt.plot(position_losses,label='position_loss')
    plt.xlabel("Step")
    plt.legend()
    plt.show()
