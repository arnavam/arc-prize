import numpy as np
import math
from collections import deque
import random
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from dsl import find_objects
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
        



