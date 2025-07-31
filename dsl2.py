import numpy as np
import math
from collections import deque
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from dsl import find_objects
def convert_np_to_native(obj):
    if isinstance(obj, list):
        return [convert_np_to_native(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        # If it's a scalar array, extract the value
        if obj.ndim == 0:
            return obj.item()
        # If it's a 1-element array, convert to scalar
        elif obj.size == 1:
            return obj.item()
        else:
            # For larger arrays, convert recursively
            return [convert_np_to_native(x) for x in obj]
    elif isinstance(obj, (np.generic,)):  # Covers np.int32, np.float64, etc.
        return obj.item()
    else:
        return obj
        


# --- NEW: Helper function for transformations ---
def generate_transformations(obj_grid):
    """
    Generates all 8 unique transformations (rotations and flips) for an object grid.
    """
    transforms = set()
    current = np.array(obj_grid)
    
    for _ in range(2): # Original and flipped
        for _ in range(4): # 4 rotations
            current = np.rot90(current)
            # Convert to tuple of tuples to make it hashable for the set
            transforms.add(tuple(map(tuple, current)))
        current = np.fliplr(current)
        
    # Convert back to list of lists
    return [list(map(list, t)) for t in transforms]


# --- NEW: Helper function to score a potential move ---
def calculate_fit_score(canvas, target_grid, obj_grid, pos):
    """
    Calculates how well placing an object fits the target.
    - Positive points for matching non-background pixels.
    - Infinite penalty for creating a mismatch.
    """
    obj_h, obj_w = len(obj_grid), len(obj_grid[0])
    r_start, c_start = pos
    
    score = 0
    for r_offset in range(obj_h):
        for c_offset in range(obj_w):
            obj_pixel = obj_grid[r_offset][c_offset]
            if obj_pixel == 0: # Skip background pixels of the object itself
                continue

            canvas_r, canvas_c = r_start + r_offset, c_start + c_offset
            
            # Check for mismatches
            # If the canvas already has something different, it's a collision
            if canvas[canvas_r][canvas_c] != 0 and canvas[canvas_r][canvas_c] != target_grid[canvas_r][canvas_c]:
                return -1 # Invalid move (already a wrong pixel there)

            # If placing this object creates a pixel that doesn't match the target
            if obj_pixel != target_grid[canvas_r][canvas_c]:
                return -1 # Invalid move (creates a wrong pixel)
                
            # If the spot on the canvas is empty and this placement is correct
            if canvas[canvas_r][canvas_c] == 0 and obj_pixel == target_grid[canvas_r][canvas_c]:
                score += 1 # Add a point for each correctly placed pixel
                
    return score


# --- NEW: Main Solver Function ---
def solve_with_transformations(input_grid, output_grid):
    """
    Solves a puzzle by iteratively placing transformed input objects onto a canvas.
    """
    # 1. Extract unique objects from the input to create a "palette"
    input_objects = find_objects(input_grid)
    
    # 2. Generate all transformations for each unique object
    transformed_palette = []
    for obj in input_objects:
        transformed_palette.extend(generate_transformations(obj))
        
    # 3. Initialize a blank canvas of the output size
    # Assuming the background color is 0
    output_h, output_w = len(output_grid), len(output_grid[0])
    canvas = [[0] * output_w for _ in range(output_h)]
    
    # 4. Iteratively find the best move and "freeze" it onto the canvas
    while True:
        best_move_this_iteration = {'score': 0, 'obj': None, 'pos': None}
        
        # Iterate through every available piece and every possible position
        for obj_to_place in transformed_palette:
            obj_h, obj_w = len(obj_to_place), len(obj_to_place[0])
            for r in range(output_h - obj_h + 1):
                for c in range(output_w - obj_w + 1):
                    pos = (r, c)
                    score = calculate_fit_score(canvas, output_grid, obj_to_place, pos)
                    
                    if score > best_move_this_iteration['score']:
                        best_move_this_iteration = {'score': score, 'obj': obj_to_place, 'pos': pos}
                        
        # 5. If a good move was found, apply it to the canvas
        if best_move_this_iteration['score'] > 0:
            best_obj = best_move_this_iteration['obj']
            r_start, c_start = best_move_this_iteration['pos']
            
            # "Freeze" the piece onto the canvas
            for r_offset in range(len(best_obj)):
                for c_offset in range(len(best_obj[0])):
                    if best_obj[r_offset][c_offset] != 0:
                         canvas[r_start + r_offset][c_start + c_offset] = best_obj[r_offset][c_offset]
        else:
            # If no move improves the score, we are done
            break
            
    return canvas


# --- Example Usage ---
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

    for case_id in train:
        for i in range(2):
            for j in ('input','output'):
                a=train[case_id]['train'][i]['input']
                b=train[case_id]['train'][i]['output']
                # print(a)
                # sns.heatmap(a,cmap=cmap)
                # plt.show()

        # a=np.array(a)
        # b=np.array(b)



        # Solve the puzzle using the new method
        solved_grid = solve_with_transformations(a, b)
        solved_grid = convert_np_to_native(solved_grid)




        print("\nSolved Grid:")
        for row in solved_grid:
            print(row )
            # Verify if the solution is correct
        is_correct = (np.array(solved_grid) == np.array(b)).all()
        print(f"\nSolution is correct: {is_correct}")





