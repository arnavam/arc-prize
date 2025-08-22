import collections
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


# --- Your Primitive Functions ---
def rotate(data):
    return np.rot90(data, k=-1)
def mirrorlr(data):
    return np.fliplr(data)
def mirrorud(data):
    return np.flipud(data)
def lcrop(data):
    if data.shape[1] > 1: return data[:, 1:]
    return data
def rcrop(data):
    if data.shape[1] > 1: return data[:, :-1]
    return data
def ucrop(data):
    if data.shape[0] > 1: return data[1:, :]
    return data
def dcrop(data):
    if data.shape[0] > 1: return data[:-1, :]
    return data

def normalize(mat):
    """Normalizes the color values in a grid to be contiguous integers starting from 0."""
    mat = np.array(mat)
    if mat.size == 0:
        return mat
    unique_vals, inverse = np.unique(mat, return_inverse=True)
    normalized_mat = inverse.reshape(mat.shape)
    return normalized_mat


def conv(input_matrix, kernel):
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Output size for 'valid' convolution
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output = np.zeros((output_h, output_w), dtype=int)

    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            # Check if region matches the kernel exactly
            if np.array_equal(region, kernel):
                output[i, j] = 0
            else:
                output[i, j] = 1
    if np.all(output == 0):
        print('won')

    return output

# --- The BFS Solver ---
def solve_with_bfs(input_grid, output_grid):

    primitives = [ normalize ,rotate, mirrorlr, mirrorud, lcrop, rcrop, ucrop, dcrop]

    start_node = np.array(input_grid)
    goal_node = np.array(output_grid)

    queue = collections.deque([(start_node, [])])# store (current_grid, path_of_functions)


    visited = {start_node.tobytes()}# NumPy arrays are not hashable,

    max_path_length = 100 # Safety break to prevent searching forever
    
    while queue:
        current_grid, path = queue.popleft()

        # Stop if the search gets too deep
        if len(path) > max_path_length:
            continue

        # --- GOAL CHECK ---
        # Check if the current grid matches the goal grid
        if np.array_equal(current_grid, goal_node):
            # print(f"✅ Solution Found! Path: {[func.__name__ for func in path]}")
            return path

         # Apply each primitive function to generate new states
        for func in primitives:
            try:
                # Important: operate on a copy to not modify the grid in the path
                new_grid = func(current_grid.copy())
                new_grid_bytes = new_grid.tobytes()

                # If we haven't seen this grid configuration before...
                if new_grid_bytes not in visited:
                    visited.add(new_grid_bytes)
                    new_path = path + [func]
                    queue.append((new_grid, new_path))
            except Exception as e:
                # Some operations might fail (e.g., cropping a 1x1 grid).
                # We can just ignore these failures and continue.
                # print(f"Operation {func.__name__} failed: {e}")
                continue

    # print("❌ No solution found within the search limit.")
    return None

train_path='arc-prize-2025/arc-agi_training_challenges.json'
with open(train_path, 'r') as f:
    train = json.load(f)

start= time.time()
ids=[]
ways=[]
count=0
for case_id in train:
    count += 1
    if count == 100:
        break
    print(count)
    ids.append(case_id)
    a = train[case_id]['train'][0]['input']
    b = train[case_id]['train'][0]['output']
    solution_path = solve_with_bfs(a, b)
    if solution_path != None:
        ways.append((solution_path,case_id))
        print(solution_path,case_id)
print(time.time()-start)
len(ways)

for i in ways:
    id= i[1]
    a = train[id]['train'][0]['input']
    b = train[id]['train'][0]['output']
    print(a)
    sns.heatmap(a,cmap=cmap)
    plt.show()
    print(b)
    sns.heatmap(b,cmap=cmap)
    plt.show()