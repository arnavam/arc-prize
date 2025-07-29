import collections
import numpy as np
import heapq # Used for the priority queue
import json

import collections
import numpy as np
import heapq # Used for the priority queue
import itertools # Used for the unique counter

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

# --- A* Heuristic Function ---
def heuristic(current_grid, goal_grid):
    """
    Estimates the cost to get from the current grid to the goal grid.
    A simple heuristic is the number of differing pixels.
    This heuristic is "admissible" because it never overestimates the true cost.
    """
    # If shapes don't match, the distance is effectively infinite for a simple comparison.
    # A more advanced heuristic might try to align or pad them.
    if current_grid.shape != goal_grid.shape:
        return float('inf')
    
    # Calculate the number of pixels that are different
    return np.sum(current_grid != goal_grid)

# --- The A* Solver ---
def solve_with_a_star(input_grid, output_grid):
    """
    Finds a sequence of primitive operations using the A* search algorithm.

    Args:
        input_grid (list of lists): The starting grid.
        output_grid (list of lists): The target grid.

    Returns:
        A list of function names representing the solution path, or None if no solution is found.
    """
    primitives = [rotate, mirrorlr, mirrorud, lcrop, rcrop, ucrop, dcrop]
    
    start_node = np.array(input_grid)
    goal_node = np.array(output_grid)

    # The priority queue stores tuples of: (priority, cost, counter, path, grid)
    # priority = cost + heuristic
    # cost (g_score) = number of steps taken so far
    # counter = a unique value to break ties
    # path = list of functions applied
    # grid = the current numpy array state
    
    # Initialize a unique counter for tie-breaking
    counter = itertools.count()

    # Initial state
    g_score = 0
    h_score = heuristic(start_node, goal_node)
    f_score = g_score + h_score
    
    # The priority queue (min-heap)
    open_set = [(f_score, g_score, next(counter), [], start_node)]
    
    # A dictionary to store the minimum cost (g_score) found so far to reach a state.
    # Key: grid state in bytes. Value: minimum g_score.
    visited_costs = {start_node.tobytes(): 0}

    max_path_length = 100 # Safety break

    while open_set:
        # Get the node with the lowest f_score from the priority queue
        current_f_score, current_g_score, _, path, current_grid = heapq.heappop(open_set)

        # Stop if the search gets too deep
        if len(path) > max_path_length:
            continue

        # --- GOAL CHECK ---
        if np.array_equal(current_grid, goal_node):
            # print(f"✅ Solution Found! Path: {[func.__name__ for func in path]}")
            return path

        # --- EXPLORE NEIGHBORS ---
        for func in primitives:
            try:
                new_grid = func(current_grid.copy())
                new_grid_bytes = new_grid.tobytes()
                
                # Calculate the cost (g_score) for this new path
                new_g_score = current_g_score + 1

                # If we've seen this state before but with a lower or equal cost, skip it.
                if new_grid_bytes in visited_costs and visited_costs[new_grid_bytes] <= new_g_score:
                    continue
                
                # This is a better path to this state, so update its cost
                visited_costs[new_grid_bytes] = new_g_score
                
                # Calculate the heuristic and total priority (f_score)
                h = heuristic(new_grid, goal_node)
                if h == float('inf'): # Don't explore paths with mismatched shapes
                    continue
                    
                new_f_score = new_g_score + h
                
                # Add the new state to the priority queue
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
