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

# Global set to keep track of cleared folders
cleared_folders = set()

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)



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
def display(input, predicted, target, folder='train_outputs', input_title='Input',predicted_title='Predicted',target_title='target',printing=True):

    if folder not in cleared_folders: # TO clear folder only once

        if os.path.exists(folder):
            clear(folder)
        cleared_folders.add(folder)

    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(input, cmap=cmap, norm=norm ,ax=axes[0], cbar=False)
    axes[0].set_title(input_title)

    sns.heatmap(predicted, cmap=cmap,norm=norm, ax=axes[1], cbar=False)
    axes[1].set_title(predicted_title)

    sns.heatmap(target, cmap=cmap, norm=norm,ax=axes[2], cbar=False)
    axes[2].set_title(target_title)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S,%f")[:-3]
    filename = f"heatmap_{timestamp}.png"

    plt.savefig(os.path.join(folder, filename))

    if printing:
        print(f"Figure saved as {filename}")

    plt.close()


def plot_metrics(train_losses: list[float], train_accuracies: list[float], folder='accuracy_and_loss_plot'):
    epochs = list(range(1, len(train_losses) + 1))
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color='tab:blue')
    ax2.plot(epochs, train_accuracies, label='Train Accuracy', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S,%f")[:-3]
    filename = f"heatmap_{timestamp}.png"

    plt.savefig(os.path.join(folder, filename))
    fig.tight_layout()
    plt.title('Training Loss and Accuracy Over Epochs')
    plt.grid(True)
    plt.show()





