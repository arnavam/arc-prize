import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log output to current_grid file named app.log
    filemode='w'  # Overwrite the log file each time the program runs
)

#--------------------------------------------------


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

#--------------------------------------------------

def placement(output_grid, old_obj_info, new_obj_info, background=0):
    new_grid = new_obj_info["grid"]
    new_r, new_c = new_obj_info["position"]
    new_h, new_w = new_grid.shape

    # Determine overlapping region with output grid (for truncation)
    max_new_r = min(new_r + new_h, output_grid.shape[0])
    max_new_c = min(new_c + new_w, output_grid.shape[1])
    start_r = max(new_r, 0)
    start_c = max(new_c, 0)

    if max_new_r <= start_r or max_new_c <= start_c:
        # New object is completely out of grid (no overlap)
        pass  # Allow placement (nothing to draw)
    else:
        # Crop the new object grid according to the overlap with output grid
        obj_crop_r_start = start_r - new_r
        obj_crop_c_start = start_c - new_c
        obj_crop_r_end = max_new_r - new_r
        obj_crop_c_end = max_new_c - new_c
        new_crop = new_grid[obj_crop_r_start:obj_crop_r_end, obj_crop_c_start:obj_crop_c_end]

        # Corresponding region on output_grid
        grid_region = output_grid[start_r:max_new_r, start_c:max_new_c]

        # Check overlap: if any pixel in new_crop != background AND grid_region != background, reject
        overlap_mask = (new_crop != background) & (grid_region != background)
        if np.any(overlap_mask):
            print('overlap happened')
            return None  # Reject placement due to overlap

    # No overlap detected, proceed to clear old object and draw new one
    # Clear old object
    old_grid = old_obj_info["grid"]
    old_r, old_c = old_obj_info["position"]
    old_h, old_w = old_grid.shape
    max_old_r = min(old_r + old_h, output_grid.shape[0])
    max_old_c = min(old_c + old_w, output_grid.shape[1])
    start_old_r = max(old_r, 0)
    start_old_c = max(old_c, 0)

    if max_old_r > start_old_r and max_old_c > start_old_c:
        old_crop_r_start = start_old_r - old_r
        old_crop_c_start = start_old_c - old_c
        old_crop_r_end = max_old_r - old_r
        old_crop_c_end = max_old_c - old_c
        old_crop = old_grid[old_crop_r_start:old_crop_r_end, old_crop_c_start:old_crop_c_end]
        region_to_clear = output_grid[start_old_r:max_old_r, start_old_c:max_old_c]
        output_grid[start_old_r:max_old_r, start_old_c:max_old_c] = np.where(
            old_crop != background,
            background,
            region_to_clear
        )

    # Draw new object
    if max_new_r > start_r and max_new_c > start_c:
        # Recalculate new_crop if necessary (if not already calculated in overlap check)
        if max_new_r <= start_r or max_new_c <= start_c:
            # Object was completely out of bounds, nothing to draw
            pass
        else:
            region_to_draw = output_grid[start_r:max_new_r, start_c:max_new_c]
            output_grid[start_r:max_new_r, start_c:max_new_c] = np.where(
                new_crop != background,
                new_crop,
                region_to_draw
            )

    return output_grid


