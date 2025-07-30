import numpy as np
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns





train_path='arc-prize-2025/arc-agi_training_challenges.json'
with open(train_path, 'r') as f:
    train = json.load(f)


cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
ids=[]
for case_id in train:
    ids.append(case_id)

a=train[ids[1]]['train'][2]['output']


print(a)
sns.heatmap(a,cmap=cmap)




# for  case_id in train:
#     train[case_id]
# defining a handful of basic primitives



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




for i in range(2):
    for j in ('input','output'):
        a=train[ids[1]]['train'][i]['input']
        b=train[ids[1]]['train'][i]['output']
        print(a)
        sns.heatmap(a,cmap=cmap)
        plt.show()

a=np.array(a)
b=np.array(b)

if b.size <= a.size:
    print('output')
    x=normalize(b)
    y=normalize(a)
    ans=(conv(y,x))
    


else :
    print('input')
