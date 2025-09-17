import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors







cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



def display(a, b, solved_grid):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(a, cmap=cmap, ax=axes[0], cbar=False)
    axes[0].set_title('Input')

    sns.heatmap(b, cmap=cmap, ax=axes[1], cbar=False)
    axes[1].set_title('Original')

    sns.heatmap(solved_grid, cmap=cmap, ax=axes[2], cbar=False)
    axes[2].set_title('Predicted')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"heatmap_{timestamp}.png"

    plt.savefig(f'fig/{filename}')
    print(f"Figure saved as {filename}")
    plt.close()


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
        print(a)
        print(b)

        raise ValueError("Shapes do not match after padding.")

    # Pixel-by-pixel comparison
    matches = np.sum(padded_a == b)
    total = b.size
    score = matches / total  # percentage match

    return score  # Returns between 0.0 and 1.0


