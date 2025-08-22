from collections import deque

import numpy as np

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

def concat_h(grids, background=0):
    if not grids:
        return []
    
    # Calculate dimensions
    heights = [len(g) for g in grids if g]
    widths = [len(g[0]) for g in grids if g and g[0]]
    if not heights or not widths:
        return []
    
    total_height = max(heights)
    total_width = sum(widths)
    result = [[background] * total_width for _ in range(total_height)]
    
    # Paste grids side-by-side
    col_offset = 0
    for g in grids:
        if not g or not g[0]:
            continue
        h, w = len(g), len(g[0])
        for r in range(h):
            for c in range(w):
                result[r][col_offset + c] = g[r][c]
        col_offset += w
    return result

def concat_v(grids, background=0):
    if not grids:
        return []
    
    # Calculate dimensions
    heights = [len(g) for g in grids if g]
    widths = [len(g[0]) for g in grids if g and g[0]]
    if not heights or not widths:
        return []
    
    total_height = sum(heights)
    total_width = max(widths)
    result = [[background] * total_width for _ in range(total_height)]
    
    # Paste grids top-to-bottom
    row_offset = 0
    for g in grids:
        if not g or not g[0]:
            continue
        h, w = len(g), len(g[0])
        for r in range(h):
            for c in range(w):
                result[row_offset + r][c] = g[r][c]
        row_offset += h
    return result


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
def idle(data):
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
    unique_vals = np.unique(matrix)
    shifted_vals = np.roll(unique_vals, -1)  # shift left, wrap around
    
    mapping = dict(zip(unique_vals, shifted_vals))
    
    # Vectorized mapping using np.vectorize
    transform = np.vectorize(mapping.get)
    
    return transform(matrix)

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


PRIMITIVE = {
        "rotate": rotate,
        "mirrorlr": mirrorlr,
        "mirrorud": mirrorud,
        # "lcrop": lcrop,
        # "rcrop": rcrop,
        # "ucrop": ucrop,
        # "dcrop": dcrop,
        "color":color,
        'idle':idle
        # We "freeze" the 'kernel' argument with the start and goal nodes.
        # "conv_with_input_kernel": functools.partial(conv, kernel=input_node),
        # "conv":conv
    }


def move_up(position):
    x, y = position
    return  (x - 1, y)


def move_down(position):
    x, y = position

    return  (x + 1, y)

def move_left(position):
    x, y = position
    return  (x, y - 1)

def move_right( position):
    x, y = position

    return (x, y + 1)


def idle (position):
    return position

ACTIONS = {
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_right": move_right ,
    'idle':idle
}

COMB = PRIMITIVE | ACTIONS
