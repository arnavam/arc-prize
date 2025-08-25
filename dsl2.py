import numpy as np
import math
from collections import deque
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors

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
    

def find_objects(grid, max_objects=10):
    # 1. Ensure the input is a NumPy array from the start.
    if not isinstance(grid, np.ndarray):
        grid_np = np.array(grid)
    else:
        # Create a copy to avoid modifying the original array passed to the function.
        grid_np = grid.copy() 
    print('grid_np',grid_np)
    if grid_np.ndim == 3:
        # If the grid is 3D, try to squeeze it into a 2D grid.
        # This works if the shape is (rows, cols, 1).
        print(f"Warning: Input grid is 3D with shape {grid_np.shape}. Attempting to squeeze it to 2D.")
        grid_np = np.squeeze(grid_np)

    # After squeezing, verify it is now 2D. If not, we can't proceed.
    if grid_np.ndim != 2:
        raise ValueError(f"Input grid must be 2-dimensional, but has {grid_np.ndim} dimensions.")

    if grid_np.size == 0:
        return []
    
    rows, cols = grid_np.shape
    objects = []
    # Start with the entire grid as one object
    initial_object = {
        'grid': np.array([row[:] for row in grid]),
        'color': 'mixed',  # Special value indicating multiple colors
        'position': (0, 0),
        'placed':False,
        'size': (rows, cols)
    }
    
    queue = [initial_object]
    
    while queue and len(objects) < max_objects:
        current_obj = queue.pop(0)
        
        # If the object is uniform color or we've reached the limit, add to results
        if is_uniform(current_obj['grid']) or len(objects) + len(queue) >= max_objects - 1:
            if is_uniform(current_obj['grid']):
                current_obj['color'] = current_obj['grid'][0][0]
            objects.append(current_obj)
            continue
        
        # Otherwise, split into connected components by color
        color_components = split_by_color(current_obj['grid'])
        
        for component in color_components:
            if len(objects) + len(queue) + len(color_components) - 1 >= max_objects:
                # If adding these would exceed max, just add as is
                if is_uniform(component['grid']):
                    component['color'] = component['grid'][0][0]
                objects.append(component)
            else:
                queue.append(component)
    print(len(objects))
    return objects[:max_objects]  # Ensure we don't exceed max_objects

def is_uniform(grid):
    first_color = grid[0][0]
    return np.all(grid == first_color)

def split_by_color(grid):
    rows, cols = grid.shape
    visited = np.zeros((rows, cols), dtype=bool)
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                color = grid[r][c]
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-connectivity
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols 
                            and not visited[nr][nc] 
                            and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                if component:
                    # Extract bounding box
                    min_r = min(x[0] for x in component)
                    max_r = max(x[0] for x in component)
                    min_c = min(x[1] for x in component)
                    max_c = max(x[1] for x in component)
                    
                    # Create object grid
                    obj_grid = grid[min_r:max_r+1, min_c:max_c+1].copy()
                    
                    components.append({
                        'grid': obj_grid,
                        'color': color,
                        'position': (min_r, min_c),
                        'placed':False,
                        'size': (obj_grid.shape[0], obj_grid.shape[1])
                    })
    
    return components
#-----------------------------------------------------------
def extract_target_region(target_grid, obj_info):
    r, c = obj_info['position']
    obj_h, obj_w = obj_info['grid'].shape
    
    # Handle edge cases where object extends beyond target grid
    pad_h = max(0, r + obj_h - target_grid.shape[0])
    pad_w = max(0, c + obj_w - target_grid.shape[1])
    
    if pad_h > 0 or pad_w > 0:
        padded_target = np.pad(target_grid, 
                              ((0, pad_h), (0, pad_w)),
                              mode='constant',
                              constant_values=0)#use background
        return padded_target[r:r+obj_h, c:c+obj_w]
    return target_grid[r:r+obj_h, c:c+obj_w]


def pad_matrix(a, target_shape, direction):
    pad_height = target_shape[0] - a.shape[0]
    pad_width = target_shape[1] - a.shape[1]

    # Default padding: [top, bottom], [left, right]
    if direction == 'top':
        padding = ((pad_height, 0), (0, 0))
    elif direction == 'bottom':
        padding = ((0, pad_height), (0, 0))
    elif direction == 'left':
        padding = ((0, 0), (pad_width, 0))
    elif direction == 'right':
        padding = ((0, 0), (0, pad_width))
    else:
        raise ValueError("Direction must be one of: 'top', 'bottom', 'left', 'right'")

    return np.pad(a, padding, mode='constant', constant_values=0)


#-------------------------------------------------



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





