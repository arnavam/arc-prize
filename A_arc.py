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
