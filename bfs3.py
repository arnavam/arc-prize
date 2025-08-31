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

# --- Enhanced Primitive Functions ---
def rotate(data):
    return np.rot90(data.copy(), k=-1)

def mirrorlr(data):
    return np.fliplr(data.copy())

def mirrorud(data):
    return np.flipud(data.copy())

def lcrop(data):
    return data[:, 1:].copy() if data.shape[1] > 1 else data.copy()

def rcrop(data):
    return data[:, :-1].copy() if data.shape[1] > 1 else data.copy()

def ucrop(data):
    return data[1:, :].copy() if data.shape[0] > 1 else data.copy()

def dcrop(data):
    return data[:-1, :].copy() if data.shape[0] > 1 else data.copy()

def pad_left(data, value=0):
    return np.pad(data, ((0, 0), (1, 0)), constant_values=value)

def pad_right(data, value=0):
    return np.pad(data, ((0, 0), (0, 1)), constant_values=value)

def pad_up(data, value=0):
    return np.pad(data, ((1, 0), (0, 0)), constant_values=value)

def pad_down(data, value=0):
    return np.pad(data, ((0, 1), (0, 0)), constant_values=value)

def normalize(mat):
    """Normalizes the color values to contiguous integers starting from 0."""
    if mat.size == 0:
        return mat
    unique_vals, inverse = np.unique(mat, return_inverse=True)
    return inverse.reshape(mat.shape)

# --- Pattern Matching & Grid Filling ---
def find_pattern(input_matrix, pattern):
    """Finds all occurrences of pattern in input_matrix."""
    input_h, input_w = input_matrix.shape
    pattern_h, pattern_w = pattern.shape
    positions = []
    
    for i in range(input_h - pattern_h + 1):
        for j in range(input_w - pattern_w + 1):
            region = input_matrix[i:i+pattern_h, j:j+pattern_w]
            if np.array_equal(region, pattern):
                positions.append((i, j))
    return positions

def fill_template(blank_grid, pattern, position):
    """Fills the template with pattern at specified position."""
    y, x = position
    ph, pw = pattern.shape
    new_grid = blank_grid.copy()
    new_grid[y:y+ph, x:x+pw] = pattern
    return new_grid

def generate_patterns(input_grid, max_depth=3):
    """Generates transformed patterns from input using BFS."""
    primitives = [rotate, mirrorlr, mirrorud, lcrop, rcrop, ucrop, dcrop,
                  pad_left, pad_right, pad_up, pad_down]
    
    queue = collections.deque([(input_grid, [])])
    visited = {input_grid.tobytes()}
    patterns = [(input_grid, [])]
    
    while queue:
        current_grid, path = queue.popleft()
        
        if len(path) >= max_depth:
            continue
            
        for func in primitives:
            try:
                new_grid = func(current_grid)
                new_bytes = new_grid.tobytes()
                
                if new_bytes not in visited:
                    visited.add(new_bytes)
                    new_path = path + [func]
                    patterns.append((new_grid, new_path))
                    queue.append((new_grid, new_path))
            except Exception:
                continue
    return patterns

# --- Recursive Template Filling ---
def recursive_fill(input_grid, output_grid, template=None, mask=None):
    """Recursively fills output template using transformed input patterns."""
    if template is None:
        template = np.zeros_like(output_grid)
    if mask is None:
        mask = np.zeros(output_grid.shape, dtype=bool)
    
    # Base case: entire output is filled
    if np.all(mask):
        return template, []
    
    # Find the first unfilled position
    unfilled_pos = np.argwhere(~mask)
    if len(unfilled_pos) == 0:
        return template, []
    start_y, start_x = unfilled_pos[0]
    
    # Generate and try patterns
    patterns = generate_patterns(input_grid)
    for pattern, transform_seq in patterns:
        ph, pw = pattern.shape
        
        # Skip patterns that are too big or extend beyond output bounds
        if (start_y + ph > output_grid.shape[0] or 
            start_x + pw > output_grid.shape[1]):
            continue
            
        # Extract the target region from output
        target_region = output_grid[start_y:start_y+ph, start_x:start_x+pw]
        
        # Check if pattern matches target region
        if np.array_equal(pattern, target_region):
            # Create new template and mask
            new_template = template.copy()
            new_template[start_y:start_y+ph, start_x:start_x+pw] = pattern
            new_mask = mask.copy()
            new_mask[start_y:start_y+ph, start_x:start_x+pw] = True
            
            # Recursive call to fill remaining areas
            filled_template, remaining_transforms = recursive_fill(
                input_grid, output_grid, new_template, new_mask)
            
            if filled_template is not None:
                # Return successful path
                placement = [lambda g, p=pattern, pos=(start_y, start_x): 
                             fill_template(g, p, pos)]
                return filled_template, transform_seq + placement + remaining_transforms
    
    # No valid pattern found for this position
    return None, None

# --- Integrated Solver ---
def solve_with_bfs(input_node, output_node):

    
    # Case 1: Output is smaller or same size
    if input_node.shape >= output_node.shape:
        # ... (original BFS implementation for smaller outputs)
        # This part remains similar to your original BFS code
        pass
    # Case 2: Output is larger - use recursive filling
    else:
        result, transform_seq = recursive_fill(input_node, output_node)
        if result is not None and np.array_equal(result, output_node):
            return transform_seq
    return None

# --- Main Execution ---
def main():
    train_path = 'arc-prize-2025/arc-agi_training_challenges.json'
    
    with open(train_path, 'r') as f:
        train = json.load(f)
    
    solutions = []
    start_time = time.time()
    
    for count, case_id in enumerate(train.keys()):
        if count >= 10:  # Processing first 10 for testing
            break
            
        print(f"Processing case {count+1}: {case_id}")
        train_data = train[case_id]['train'][0]
        input_grid = np.array(train_data['input'])
        output_grid = np.array(train_data['output'])
        
        solution_path = solve_with_bfs(input_grid, output_grid)
        
        if solution_path is not None:
            solutions.append((solution_path, case_id))
            print(f"Solution found for {case_id}")
            # Verify solution
            current = input_grid.copy()
            for transform in solution_path:
                current = transform(current)
            if np.array_equal(current, output_grid):
                print("✅ Verification passed")
            else:
                print("❌ Verification failed")
        else:
            print(f"No solution for {case_id}")
    
    print(f"Processing time: {time.time()-start_time:.2f} seconds")
    print(f"Solved {len(solutions)} out of {min(10, len(train))} problems")
    
    # Visualization
    for solution, case_id in solutions:
        train_data = train[case_id]['train'][0]
        input_grid = np.array(train_data['input'])
        output_grid = np.array(train_data['output'])
        
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        sns.heatmap(input_grid, cmap=cmap, norm=norm, cbar=False)
        plt.title(f"Input\n{case_id}")
        
        plt.subplot(122)
        sns.heatmap(output_grid, cmap=cmap, norm=norm, cbar=False)
        plt.title("Output")
        plt.show()

if __name__ == "__main__":
    main()