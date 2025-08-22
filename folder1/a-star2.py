import collections
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import heapq
# --- Your Primitive Functions (same as before) ---
def rotate(data):
    """Rotates the grid 90 degrees clockwise."""
    return np.rot90(data, k=-1)

def mirrorlr(data):
    """Mirrors the grid left-to-right."""
    return np.fliplr(data)

def mirrorud(data):
    """Mirrors the grid up-to-down."""
    return np.flipud(data)

def lcrop(data):
    """Crops the leftmost column."""
    if data.shape[1] > 1: return data[:, 1:]
    return data

def rcrop(data):
    """Crops the rightmost column."""
    if data.shape[1] > 1: return data[:, :-1]
    return data

def ucrop(data):
    """Crops the top row."""
    if data.shape[0] > 1: return data[1:, :]
    return data

def dcrop(data):
    """Crops the bottom row."""
    if data.shape[0] > 1: return data[:-1, :]
    return data

# --- User's Convolution and Normalization Functions ---
def conv(input_matrix, kernel):
    """
    Performs a 'valid' convolution, checking where the kernel matches the input.
    Returns a grid of 0s (match) and 1s (mismatch).
    """
    input_matrix = np.array(input_matrix)
    kernel = np.array(kernel)

    if kernel.shape[0] > input_matrix.shape[0] or kernel.shape[1] > input_matrix.shape[1]:
        # Kernel is bigger than the matrix, so convolution is not possible.
        # We must return a new grid. Returning the original is a safe "no-op".
        return input_matrix

    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output = np.ones((output_h, output_w), dtype=int) # Default to 1 (mismatch)

    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            if np.array_equal(region, kernel):
                output[i, j] = 0 # Set to 0 on exact match
    return output

def normalize(mat):
    """Normalizes the color values in a grid to be contiguous integers starting from 0."""
    mat = np.array(mat)
    if mat.size == 0:
        return mat
    unique_vals, inverse = np.unique(mat, return_inverse=True)
    normalized_mat = inverse.reshape(mat.shape)
    return normalized_mat

# --- A* Heuristic Function ---
def heuristic(current_grid, goal_grid):
    """
    Estimates the cost to get from the current grid to the goal grid.
    A simple heuristic is the number of differing pixels.
    """
    if current_grid.shape != goal_grid.shape:
        return float('inf')
    return np.sum(current_grid != goal_grid)

# --- The A* Solver ---
def solve_with_a_star(input_grid, output_grid):
    """
    Finds a sequence of primitive operations using the A* search algorithm.
    """
    start_node = np.array(input_grid)
    goal_node = np.array(output_grid)

    # --- Define Primitives, including a wrapper for conv ---
    geometric_primitives = [rotate, mirrorlr, mirrorud, lcrop, rcrop, ucrop, dcrop]

    def create_conv_primitive(kernel):
        """Creates a convolution function that uses a fixed kernel."""
        normalized_kernel = normalize(kernel)
        def conv_with_fixed_kernel(data):
            normalized_data = normalize(data)
            return conv(normalized_data, normalized_kernel)
        # Give the function a descriptive name for printing the solution path
        conv_with_fixed_kernel.__name__ = f"conv_with_goal_as_kernel"
        return conv_with_fixed_kernel

    primitives = geometric_primitives
    if goal_node.size > 0 and start_node.size > 0:
        conv_primitive = create_conv_primitive(goal_node)
        primitives = primitives + [conv_primitive]
    
    counter = itertools.count()
    g_score = 0
    h_score = heuristic(start_node, goal_node)
    f_score = g_score + h_score
    
    open_set = [(f_score, g_score, next(counter), [], start_node)]
    visited_costs = {start_node.tobytes(): 0}
    max_path_length = 100

    while open_set:
        current_f_score, current_g_score, _, path, current_grid = heapq.heappop(open_set)

        if len(path) > max_path_length:
            continue

        if np.array_equal(current_grid, goal_node):
            # print(f"✅ Solution Found! Path: {[func.__name__ for func in path]}")
            return path

        for func in primitives:
            try:
                new_grid = func(current_grid.copy())
                new_grid_bytes = new_grid.tobytes()
                new_g_score = current_g_score + 1

                if new_grid_bytes in visited_costs and visited_costs[new_grid_bytes] <= new_g_score:
                    continue
                
                visited_costs[new_grid_bytes] = new_g_score
                h = heuristic(new_grid, goal_node)
                if h == float('inf'):
                    continue
                    
                new_f_score = new_g_score + h
                new_path = path + [func]
                heapq.heappush(open_set, (new_f_score, new_g_score, next(counter), new_path, new_grid))

            except Exception as e:
                continue

    # print("❌ No solution found within the search limit.")
    return None


train_path='arc-prize-2025/arc-agi_training_challenges.json'
with open(train_path, 'r') as f:
    train = json.load(f)

ids=[]
ways=[]
count=0
for case_id in train:
    count += 1
    print(count)
    ids.append(case_id)
    a = train[case_id]['train'][0]['input']
    b = train[case_id]['train'][0]['output']
    solution_path = solve_with_a_star(a, b)
    if solution_path != None:
        ways.append((solution_path,case_id))
        print(solution_path)

len(ways)

# for i in ways:
#     id= i[1]
#     a = train[id]['train'][0]['input']
#     b = train[id]['train'][0]['output']
#     print(a)
#     sns.heatmap(a,cmap=cmap)
#     plt.show()
#     print(b)
#     sns.heatmap(b,cmap=cmap)
#     plt.show()