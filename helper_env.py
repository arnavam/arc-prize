import numpy as np
import logging

import random 
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to current_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

def matrix_similarity(a, b, direction=None):
    if a.shape == b.shape:
        padded_a = a

    else:
        # Make sure a is smaller or equal in shape
        if a.shape[0] > b.shape[0] or a.shape[1] > b.shape[1]:
            # Cut down `b` to the shape of `a`
            min_rows = min(a.shape[0], b.shape[0])
            min_cols = min(a.shape[1], b.shape[1])

            a = a[:min_rows, :min_cols]
            b = b[:min_rows, :min_cols]
            padded_a = a
        else:
            padded_a = pad_matrix(a, b.shape, direction)

    # Ensure shapes now match
    if padded_a.shape != b.shape:
        logging.warning(a)
        logging.warning(b)

        raise ValueError("Shapes do not match after padding.")

    # Pixel-by-pixel comparison
    matches = np.sum(padded_a == b)
    total = b.size
    score = matches / total  # percentage match

    return score  # Returns between 0.0 and 1.0

def pad_matrix(a, target_shape, direction):
    pad_height = target_shape[0] - a.shape[0]
    pad_width = target_shape[1] - a.shape[1]

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

#--------------------------------------------------

    
def placement(canvas, old_obj, new_obj, background=0):
    result = canvas.copy()

    # print(f"Old object top-left: {old_top_left}")
    old_canvas = place_object(np.full_like(canvas, background), old_obj["grid"], old_obj["position"])
    
    # print(f"New object top-left: {new_top_left}")
    new_canvas = place_object(np.full_like(canvas, background), new_obj["grid"], new_obj["position"])
    

    canvas_without_old = result.copy()
    canvas_without_old = np.where(old_canvas != background, background, canvas_without_old)
    
    overlap_mask = (new_canvas != background) & (canvas_without_old != background)
    
    if np.any(overlap_mask):
        print("Overlap detected with other objects")

        old_object_mask = (old_canvas != background)
        if not np.all(overlap_mask <= old_object_mask):
            return None
    
    # No overlap detected, proceed to update the canvas
    # Remove the old object by restoring the background
    result = np.where(old_canvas != background, background, result)
    
    # Place the new object
    result = np.where(new_canvas != background, new_canvas, result)
    
    return result

def place_object(grid, obj_grid, pos):
    """
    Places an object onto a grid, correctly handling partial or no overlap.
    """
    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_grid.shape

    # Get the top-left coordinate for the object on the grid
    y_top, x_left = coordinate_converter(pos, obj_grid.shape, is_center=True)

    # --- Calculate the overlapping area ---

    # Find the starting y-coordinate on both grids
    grid_y_start = max(0, y_top)
    obj_y_start = max(0, -y_top)

    # Find the starting x-coordinate on both grids
    grid_x_start = max(0, x_left)
    obj_x_start = max(0, -x_left)

    # Find the height of the overlap
    # This is the distance from the top of the overlap to the bottom
    overlap_h = max(0, min(y_top + obj_h, grid_h) - grid_y_start)

    # Find the width of the overlap
    overlap_w = max(0, min(x_left + obj_w, grid_w) - grid_x_start)

    # --- If there is an overlap, perform the copy ---
    if overlap_h > 0 and overlap_w > 0:
        # Define the destination slice on the main grid
        dest_slice = grid[grid_y_start : grid_y_start + overlap_h,
                          grid_x_start : grid_x_start + overlap_w]

        # Define the source slice from the object grid
        src_slice = obj_grid[obj_y_start : obj_y_start + overlap_h,
                             obj_x_start : obj_x_start + overlap_w]
        
        # Copy the source slice to the destination
        dest_slice[:] = src_slice
    
    # The grid is modified in-place, but returning it is good practice
    return grid


def coordinate_converter(position, obj_size, is_center=True):
    y, x = position
    obj_h, obj_w = obj_size
    
    if is_center:
        # Use floor division to get integer coordinates
        y_top_left = int(np.floor(y - obj_h / 2))
        x_top_left = int(np.floor(x - obj_w / 2))
        return (y_top_left, x_top_left)
    else:
        y_center = int(np.floor(y + obj_h / 2))
        x_center = int(np.floor(x + obj_w / 2))

        return (y_center, x_center)
    
def find_empty_spot(grid, obj_size):
    grid_h, grid_w = grid.shape
    obj_h, obj_w = obj_size
    possible_spots = []

    for y in range(grid_h - obj_h + 1):
        for x in range(grid_w - obj_w + 1):
            if np.all(grid[y:y+obj_h, x:x+obj_w] == 0):
                # Convert top-left to center position
                center_pos = coordinate_converter((y, x), obj_size, is_center=False)
                # center_pos=y,x
                possible_spots.append(center_pos)

    return random.choice(possible_spots) if possible_spots else None



# Example usage
if __name__ == '__main__':
    # Original canvas (5x5 grid)
    canvas = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    # Old object (3x2 rectangle)
    old_obj = {
        "grid": np.array([

            [1, 5],
            [5, 1]
        ]),
        "position": (2, 2)  # Center position
    }

    # New object (rotated version: 2x3 rectangle)
    new_obj = {
        "grid": np.array([
            [1, 1, 1],
            [1, 1, 1]
        ]),
        "position": (1, 2)  # New center position
    }
    
    pos=find_empty_spot(canvas.copy(),old_obj['grid'].shape)


    while pos is not None:
        new_canvas = place_object(canvas, old_obj['grid'], pos)

        print("Original canvas:")
        print(canvas)
        print("\nUpdated canvas:")
        print(new_canvas)

        # Update canvas for the next iteration
        canvas = new_canvas
        # Now search for the next empty spot in the updated grid
        pos = find_empty_spot(canvas, old_obj['grid'].shape)