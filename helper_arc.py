import numpy as np
import json
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime


import logging

plt.set_loglevel (level = 'warning')
pil_logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)



def get_module_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_filename = f"log/{name}.log"
    handler = logging.FileHandler(log_filename, mode='w')

    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger


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

def display(input ,predicted,target,folder='train_ouputs',printing=True):
    
    folder = 'visualizations/' + folder
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(input, cmap=cmap, ax=axes[0], cbar=False)
    axes[0].set_title('Input')

    sns.heatmap(predicted, cmap=cmap, ax=axes[1], cbar=False)
    axes[1].set_title('Predicted')

    sns.heatmap(target, cmap=cmap, ax=axes[2], cbar=False)
    axes[2].set_title('Target')



    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S,%f")[:-3]
    filename = f"heatmap_{timestamp}.png"

    plt.savefig(f'{folder}/{filename}')

    if printing == True:
        print(f"Figure saved as {filename}")

    plt.close()


def loader(train_path):

    with open(train_path, 'r') as f:
        train = json.load(f)
    ids=[]
    for case_id in train:
        ids.append(case_id)
    return train , ids 

