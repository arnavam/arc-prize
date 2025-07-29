import collections
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import functools # Import the functools library

# --- Matplotlib and Seaborn Setup ---
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


# --- Your Primitive Functions ---
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

def normalize(mat):
    """Normalizes color values to be contiguous integers starting from 0."""
    mat = np.array(mat)
    if mat.size == 0:
        return mat
    # Find unique values and map them to 0, 1, 2, ...
    unique_vals, inverse = np.unique(mat, return_inverse=True)
    normalized_mat = inverse.reshape(mat.shape)
    return normalized_mat

def color(matrix):
    max_val = matrix.max()
    transformed = (matrix % max_val) + 1
    return transformed

def conv(input_matrix, kernel):
    """
    Performs a 'valid' convolution. Returns a grid of 0s and 1s.
    0 where the kernel matches, 1 otherwise.
    """
    # Safety check: if kernel is larger than the input, the operation is impossible.
    if kernel.shape[0] > input_matrix.shape[0] or kernel.shape[1] > input_matrix.shape[1]:
        return input_matrix # Return original matrix to avoid crashing the search

    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output = np.zeros((output_h, output_w), dtype=int)

    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            if np.array_equal(region, kernel):
                output[i, j] = 0 # Match found
            else:
                output[i, j] = 1 # No match
    return output


# --- The BFS Solver (Updated) ---
def solve_with_bfs(input_grid, output_grid):
    """
    Attempts to find a sequence of primitive operations to transform the
    input_grid into the output_grid using Breadth-First Search.
    """
    input_node = np.array(input_grid)
    output_node = np.array(output_grid)

    # We create the primitive map inside the function. This allows us to create
    # convolution functions that are specific to the current puzzle's input/output.
    primitive_map = {
        "rotate": rotate,
        "mirrorlr": mirrorlr,
        "mirrorud": mirrorud,
        "lcrop": lcrop,
        "rcrop": rcrop,
        "ucrop": ucrop,
        "dcrop": dcrop,
        "normalize": normalize,
        # Use functools.partial to create one-argument functions from 'conv'.
        # We "freeze" the 'kernel' argument with the start and goal nodes.
        "conv_with_input_kernel": functools.partial(conv, kernel=input_node),
    }

    # The queue will store tuples of (current_grid, path_of_function_names)
    queue = collections.deque([(input_node, [])])
    visited = {input_node.tobytes()} # Use bytes for hashable set keys

    max_path_length = 100 # Safety break to prevent searching forever
    start_time=time.time()
    while queue:
        current_grid, path = queue.popleft()

        if len(path) > max_path_length:
            continue

        # --- GOAL CHECK ---
        if np.array_equal(current_grid, output_node):
            # print(f"✅ Solution Found! Path: {path}")
            return path
        
        # --- GENERATE NEW STATES ---
        # Iterate through our named primitives
        for name, func in primitive_map.items():
            try:
                # Operate on a copy to avoid modifying the grid shared by other branches
                new_grid = func(current_grid.copy())
                new_grid_bytes = new_grid.tobytes()

                if new_grid_bytes not in visited:
                    visited.add(new_grid_bytes)
                    new_path = path + [name] # Add the function's name to the path
                    queue.append((new_grid, new_path))
            except Exception:
                # Some operations might fail (e.g., cropping a 1x1 grid).
                # We can safely ignore these and continue the search.
                continue
        if time.time()-start_time > 40:
            break

    return None

# --- Main Execution Logic ---
# Make sure the path to your JSON file is correct
try:
    train_path = 'arc-prize-2025/arc-agi_training_challenges.json'
    with open(train_path, 'r') as f:
        train_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{train_path}' was not found.")
    print("Please make sure the JSON file is in the correct directory.")
    train_data = {} # Assign empty dict to prevent crash

start_time = time.time()
solutions = []
case_count = 0
MAX_CASES_TO_TEST = 100

for case_id, case_data in train_data.items():
    case_count += 1
    if case_count > MAX_CASES_TO_TEST:
        break
    
    print(f"[{case_count}/{MAX_CASES_TO_TEST}] Processing case: {case_id}")
    
    # Using the first training pair
    input_matrix = case_data['train'][0]['input']
    output_matrix = case_data['train'][0]['output']
    
    solution_path = solve_with_bfs(input_matrix, output_matrix)
    
    if solution_path is not None:
        solutions.append({'id': case_id, 'path': solution_path})
        print(f"  -> Solution found for {case_id}: {solution_path}")

print("\n" + "="*30)
print("      SEARCH COMPLETE")
print("="*30)
print(f"Time taken: {time.time() - start_time:.2f} seconds")
print(f"Found solutions for {len(solutions)} out of {case_count-1} cases tested.")
print("="*30 + "\n")


# --- Visualization of Found Solutions ---
if not solutions:
    print("No solutions were found to visualize.")
else:
    print("Visualizing found solutions...")

for sol in solutions:
    case_id = sol['id']
    path = sol['path']
    
    input_m = train_data[case_id]['train'][0]['input']
    output_m = train_data[case_id]['train'][0]['output']
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(f"Solution for: {case_id}\nPath: {' -> '.join(path)}", fontsize=12)
    
    # Plot Input
    sns.heatmap(input_m, cmap=cmap, norm=norm, ax=axes[0], cbar=False, annot=True, fmt='d', linewidths=.5, linecolor='gray')
    axes[0].set_title("Input")
    axes[0].set_aspect('equal')


    # Plot Output
    sns.heatmap(output_m, cmap=cmap, norm=norm, ax=axes[1], cbar=False, annot=True, fmt='d', linewidths=.5, linecolor='gray')
    axes[1].set_title("Output")
    axes[1].set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
