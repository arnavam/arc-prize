import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from rl_models.DQNAction_Classifier import DQN_Classifier
from rl_models.feature_extractor import FeatureExtractor
from rl_models.ReinLikelihood import Likelihood
# --- Helper Functions for Data Generation ---
from dsl import COMB

num_actions = len(list(COMB.keys()))
criterion = nn.CrossEntropyLoss()

def create_random_object(max_size=3, max_color=9):

    size = (random.randint(1, max_size), random.randint(1, max_size))
    color = random.randint(1, max_color)
    obj_grid = np.full(size, color)
    return {'grid': obj_grid, 'color': color, 'size': size}
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

def generate_simple_task(grid_size=(10, 10), num_bg_objects=5, training_eg=4):

    datasets = []
    objects=[]
    labels=[]

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
            

        else:
            training_eg +=1

    return datasets , labels


def generate_movement_task(grid_size=(10, 10), num_total_objects=4):
    """
    Generates a task where an object is in the wrong place and must be moved.
    """
    target_grid = np.zeros(grid_size, dtype=int)
    placed_objects = []
    object_positions = {}

    # 1. Populate the target grid
    for _ in range(num_total_objects):
        obj = create_random_object()
        pos = find_empty_spot(target_grid, obj['size'])
        if pos:
            target_grid = place_object(target_grid, obj, pos)
            placed_objects.append(obj)
            object_positions[obj['color']] = pos # Store original position
        
    if len(placed_objects) < 2: return None, None # Need at least two to move

    # 2. Choose one object to be moved
    object_to_move = random.choice(placed_objects)

    # 3. Create the current_grid by moving the object
    current_grid = target_grid.copy()
    # Erase from its correct final position
    y_t, x_t = object_positions[object_to_move['color']]
    h, w = object_to_move['size']
    current_grid[y_t:y_t+h, x_t:x_t+w] = 0
    
    # Find a new, incorrect spot to place it
    new_pos = find_empty_spot(current_grid, object_to_move['size'])
    if not new_pos: return None, None # Failed to find a new spot
    current_grid = place_object(current_grid, object_to_move, new_pos)
    
    # 4. Candidate objects are ALL objects present in the current_grid
    # Note: We just use the original list of objects, which is simpler here
    candidate_objects = placed_objects
    random.shuffle(candidate_objects)
    
    correct_action_index = candidate_objects.index(object_to_move)

    inputs = (current_grid, candidate_objects, target_grid)
    return inputs, correct_action_index

# --- Main  Function ---



def train_supervised(model, inputs, correct_action_indices):
    """
    Performs a single step of supervised training.
    
    Args:
        inputs (list): A list of tuples, where each tuple is (current_grid, objects, target_grid).
        correct_action_indices (torch.Tensor): A tensor of correct object indices (the labels).
    """
    model.optimizer.zero_grad()
    
    all_scores = []
    for current_grid, objects, target_grid in inputs:

        current_grid_t = model._preprocess_to_tensor(current_grid)
        target_grid_t = model._preprocess_to_tensor(target_grid)
        
        scores_per_sample = []
        for obj in objects:
            obj_grid_t = model._preprocess_to_tensor(obj['grid'])

            score = model.policy([current_grid_t, obj_grid_t, target_grid_t])
            scores_per_sample.append(score.squeeze())

        all_scores.append(torch.stack(scores_per_sample))

    scores_batch = torch.stack(all_scores).to(model.device)
    correct_action_indices = torch.tensor(correct_action_indices, dtype=torch.long, device=model.device)

    loss = criterion(scores_batch, correct_action_indices)
    
    loss.backward()
    model.optimizer.step()
    
    with torch.no_grad():
        preds = scores_batch.argmax(dim=1)
        acc = (preds == correct_action_indices).float().mean().item()

    return loss.item(), acc

    # --- NEW METHOD for Supervised Prediction ---
def predict_supervised(model, input_data):
    """
    Makes a deterministic prediction based on the highest score.
    """
    current_grid, objects, target_grid = input_data
    
    # Set the model to evaluation mode
    model.policy.eval()
    with torch.no_grad():
        current_grid_t = model._preprocess_to_tensor(current_grid)
        target_grid_t = model._preprocess_to_tensor(target_grid)
        
        scores = []
        for obj in objects:
            obj_grid_t = model._preprocess_to_tensor(obj['grid'])
            score = model.policy([current_grid_t, obj_grid_t, target_grid_t])
            scores.append(score.squeeze())
        
        scores_t = torch.stack(scores)
        # Choose the action with the highest score
        best_action = torch.argmax(scores_t).item()
    
    # Set the model back to training mode
    model.policy.train()
    return best_action


if __name__ == '__main__':

    ft1 = FeatureExtractor(input_channels=1)
    likelihood_predictor = Likelihood(feature_extractor=ft1, output_dim=1,no_of_inputs=3)
    likelihood_predictor.load()
    likelihood_predictor.show_structure()
    tasks = [generate_simple_task(grid_size=(10, 10), num_bg_objects=5 , training_eg=8) for _ in range(100)]
    losses=[]
    accs=[]
    # Then train over those
    for epoch in range(1):
        for datasets, labels in tasks:
            loss,acc=train_supervised(likelihood_predictor,datasets,labels)
            print(f"Step {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.2%}")
            losses.append(loss)
            accs.append(acc)
    likelihood_predictor.save()
    plt.plot(accs, label='Accuracy')
    plt.plot(losses, label='Loss')
    plt.xlabel("Step")
    plt.legend()
    plt.show()

