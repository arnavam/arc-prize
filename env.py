import numpy as np


def placement(output_grid, old_obj_info, new_obj_info, background=0):
    """
    Attempts to move/transform an object on the grid safely:
    - Checks if the new placement overlaps existing non-background pixels
    - Allows truncation at edges (partial placement)
    - If invalid (overlaps), returns None and does not modify grid
    - If valid, clears old position and draws new shape, then returns updated new_obj_info

    Parameters:
        output_grid (ndarray): The grid.
        old_obj_info (dict): Contains old "grid" and "position".
        new_obj_info (dict): Contains new "grid" and "position".
        background: Background value.

    Returns:
        new_obj_info if successful, None otherwise.
    """

    new_grid = new_obj_info["grid"]
    # print(new_grid)
    new_r, new_c = new_obj_info["position"]
    new_h, new_w = new_grid.shape

    # Determine overlapping region with output grid (for truncation)
    max_new_r = min(new_r + new_h, output_grid.shape[0])
    max_new_c = min(new_c + new_w, output_grid.shape[1])
    start_r = max(new_r, 0)
    start_c = max(new_c, 0)

    if max_new_r <= start_r or max_new_c <= start_c:
        # New object is completely out of grid (no overlap)
        # Placement is trivially valid, but nothing to draw
        # You might consider returning None or allowing empty placement
        pass  # allow

    else:
        # Crop the new object grid according to the overlap with output grid
        obj_crop_r_start = start_r - new_r  # >=0
        obj_crop_c_start = start_c - new_c  # >=0
        obj_crop_r_end = max_new_r - new_r
        obj_crop_c_end = max_new_c - new_c
        new_crop = new_grid[obj_crop_r_start:obj_crop_r_end, obj_crop_c_start:obj_crop_c_end]

        # Corresponding region on output_grid
        grid_region = output_grid[start_r:max_new_r, start_c:max_new_c]

        # Check overlap: if any pixel in new_crop != background AND grid_region != background, reject
        overlap_mask = (new_crop != background) & (grid_region != background)
        if np.any(overlap_mask):
            return None  # Cannot place object here due to overlap

    # No overlap detected, proceed to clear old object and draw new one

    # Clear old object (similar to before)
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

    # Draw new object (same as overlap check crop)
    if max_new_r > start_r and max_new_c > start_c:
        region_to_draw = output_grid[start_r:max_new_r, start_c:max_new_c]
        output_grid[start_r:max_new_r, start_c:max_new_c] = np.where(
            new_crop != background,
            new_crop,
            region_to_draw
        )

    # Update position in new_obj_info
    return output_grid