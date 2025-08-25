import collections
import numpy as np
import json
import pandas as pd
from pathlib import Path


def rotate(data):
    """Rotates the grid 90 degrees clockwise."""
    return np.rot90(data, k=-1)

def mirrorlr(data):
    """Mirrors the grid left-to-right."""
    return np.fliplr(data)

def mirrorud(data):
    """Mirrors the grid up-down."""
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

# --- The BFS Solver (from your original code) ---
# This function finds the sequence of transformations to get from input to output.
def solve_with_bfs(input_grid, output_grid):
    """
    Performs a Breadth-First Search to find a sequence of primitive functions
    that transform the input_grid into the output_grid.
    """
    primitives = [rotate, mirrorlr, mirrorud, lcrop, rcrop, ucrop, dcrop]

    start_node = np.array(input_grid)
    goal_node = np.array(output_grid)

    # The queue stores tuples of (current_grid, path_of_functions)
    queue = collections.deque([(start_node, [])])

    # Use a set to keep track of visited grid states to avoid cycles and redundant work.
    # NumPy arrays are not hashable, so we convert them to bytes.
    visited = {start_node.tobytes()}

    max_path_length = 10 # Safety break for complex problems

    while queue:
        current_grid, path = queue.popleft()

        # Stop if the search gets too deep
        if len(path) > max_path_length:
            continue

        # --- GOAL CHECK ---
        if np.array_equal(current_grid, goal_node):
            # print(f"✅ Solution Found! Path: {[func.__name__ for func in path]}")
            return path

        # --- EXPLORE NEIGHBORS ---
        # Apply each primitive function to generate new states
        for func in primitives:
            try:
                # Operate on a copy to not modify the grid in the current path
                new_grid = func(current_grid.copy())
                new_grid_bytes = new_grid.tobytes()

                if new_grid_bytes not in visited:
                    visited.add(new_grid_bytes)
                    new_path = path + [func]
                    queue.append((new_grid, new_path))
            except Exception as e:
                # Some operations might fail (e.g., cropping a 1x1 grid).
                # We can just ignore these failures and continue.
                continue

    return None


def apply_path(grid, path):
    """
    Applies a sequence of transformation functions to a given grid.

    Args:
        grid (np.array): The input grid to transform.
        path (list): A list of functions (the solution path).

    Returns:
        np.array: The transformed grid.
    """
    transformed_grid = grid.copy()
    for func in path:
        transformed_grid = func(transformed_grid)
    return transformed_grid


if __name__ == '__main__':
    eval_path = 'arc-prize-2025/arc-agi_evaluation_challenges.json'
    output_path = 'arc-prize-2025/submission.json'

    try:
        with open(eval_path, 'r') as f:
            eval_tasks = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {eval_path} was not found.")

    predictions = {}
    print('hai')
    for task_id, task_data in eval_tasks.items():
        task_predictions = []

        # Train example to learn transformation
        train_input = np.array(task_data['train'][0]['input'])
        train_output = np.array(task_data['train'][0]['output'])

        solution_path = solve_with_bfs(train_input, train_output)

        for test_case in task_data['test']:
            test_input_grid = np.array(test_case['input'])

            if solution_path:
                predicted_output = apply_path(test_input_grid, solution_path)
                attempt_1 = predicted_output.tolist()
                # You could apply another transformation or just repeat
                attempt_2 = predicted_output.tolist()
            else:
                # Fallback: just use input as prediction
                attempt_1 = test_input_grid.tolist()
                attempt_2 = test_input_grid.tolist()

            task_predictions.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })

        predictions[task_id] = task_predictions

    # Save predictions as submission.json
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"✅ Submission file '{output_path}' created successfully!")
