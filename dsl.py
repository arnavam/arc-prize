from collections import deque

import numpy as np



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
def place(data):
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

def remove(data):
    data=np.zeros_like(data) 
    return data

PRIMITIVE = {
        "rotate": rotate,
        "mirrorlr": mirrorlr,
        "mirrorud": mirrorud,
        # "lcrop": lcrop,
        # "rcrop": rcrop,
        # "ucrop": ucrop,
        # "dcrop": dcrop,
        'remove':remove,

        "color":color,
        'place':place,
        # We "freeze" the 'kernel' argument with the start and goal nodes.
        # "conv_with_input_kernel": functools.partial(conv, kernel=input_node),
        # "conv":conv

    }

#remove , 

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


def place (position):
    return position

ACTIONS = {
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_right": move_right ,
}

COMB = PRIMITIVE | ACTIONS
