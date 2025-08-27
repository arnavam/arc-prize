import numpy as np
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
def clear(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

def display(a, b, solved_grid,folder='train_ouputs'):
    cmap = 'coolwarm'  # Example colormap

            # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Plot each heatmap on a separate subplot
    sns.heatmap(a, cmap=cmap, ax=axes[0], cbar=False)
    axes[0].set_title('Input')

    sns.heatmap(b, cmap=cmap, ax=axes[1], cbar=False)
    axes[1].set_title('Target')

    sns.heatmap(solved_grid, cmap=cmap, ax=axes[2], cbar=False)
    axes[2].set_title('Predicted')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"heatmap_{timestamp}.png"

    # Save the figure
    plt.savefig(f'{folder}/{filename}')

    # Optionally, print the filename to confirm
    print(f"Figure saved as {filename}")

    # Close the figure to free up memory
    plt.close()


train_path='arc-prize-2025/arc-agi_training_challenges.json'
with open(train_path, 'r') as f:
    train = json.load(f)



def loader(train_path):

    with open(train_path, 'r') as f:
        train = json.load(f)
    ids=[]
    for case_id in train:
        ids.append(case_id)
    return train , ids 

ids=[]
for case_id in train:
    ids.append(case_id)


def rotate(data):

    rotated = np.rot90(data, k=-1)    
    return rotated
 
def mirrorlr(data):
    mirrored = np.fliplr(data)
    return mirrored

def mirrorud(data):
    mirrored = np.flipud(data)
    return mirrored

def lcrop(data):
    left = data[:, 1:] 
    return left

def rcrop(data):
    right = data[:, :-1] 
    return right

def ucrop(data):
    top = data[1:, :] 
    return top

def dcrop(data):
    bottom = data[:-1, :] 

    return bottom


    pass



def conv(input_matrix, kernel):
    input_h, input_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Output size for 'valid' convolution
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1
    output = np.zeros((output_h, output_w), dtype=int)

    for i in range(output_h):
        for j in range(output_w):
            region = input_matrix[i:i+kernel_h, j:j+kernel_w]
            # Check if region matches the kernel exactly
            if np.array_equal(region, kernel):
                output[i, j] = 0
            else:
                output[i, j] = 1
    if np.all(output == 0):
        print('won')

    return output



def normalize(mat):

    unique_vals, inverse = np.unique(mat, return_inverse=True)
    normalized_mat = inverse.reshape(mat.shape)
    return normalized_mat


def new_func(train, cmap, ids, data, i):
    for j in ('input','output'):
        a=train[ids[1]][data][i]['input']
        b=train[ids[1]][data][i]['output']
        print(a)
        sns.heatmap(a,cmap=cmap)
        plt.show()
    return a,b

if __name__ == '__main__':
    data='train'
    for i in range(2):
        a, b = new_func(train, cmap, ids, data, i)

    a=np.array(a)
    b=np.array(b)

    if b.size <= a.size:
        print('output')
        x=normalize(b)
        y=normalize(a)
        ans=(conv(y,x))
        


    else :
        print('input')
