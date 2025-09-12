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
pil_logger.setLevel(logging.INFO) # override the logger logging level to INFO


cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



# Global set to keep track of cleared folders
cleared_folders = set()



def loader(dataset_path='arc-prize-2025/arc-agi_training_challenges.json'):

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    ids=[]
    for case_id in dataset:
        ids.append(case_id)
    return dataset , ids 


# could be used to clear a folder but mainly used to clear images in a folder
def clear(folder_path):

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# save  the  image of the input , predicted  and target in  'folder'
def display(input, predicted, target, folder='train_outputs', printing=True):

    if folder not in cleared_folders: # TO clear folder only once

        if os.path.exists(folder):
            clear(folder)
        cleared_folders.add(folder)

    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(input, cmap='viridis', ax=axes[0], cbar=False)
    axes[0].set_title('Input')

    sns.heatmap(predicted, cmap='viridis', ax=axes[1], cbar=False)
    axes[1].set_title('Predicted')

    sns.heatmap(target, cmap='viridis', ax=axes[2], cbar=False)
    axes[2].set_title('Target')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S,%f")[:-3]
    filename = f"heatmap_{timestamp}.png"

    plt.savefig(os.path.join(folder, filename))

    if printing:
        print(f"Figure saved as {filename}")

    plt.close()






