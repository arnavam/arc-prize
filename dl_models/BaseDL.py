import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('log/base_dqn.log', mode='w')
# handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

import random
from collections import deque



# --- Parent/Base DQN Class ---
class BaseDL:

    def __init__(self, device):
        _device = torch.device(device if getattr(torch, device).is_available() else "cpu")
        print(f'Using device: {_device} for Multi-Head Solver')

        self.device = _device

    def select_action(self, state, epsilon=0.4):
            pass

    def update_policy(self):
            pass

    def _preprocess_to_tensor(self, grid, dtype=torch.float32, size=30):

        if isinstance(grid, torch.Tensor):  
            tensor = grid.to(device=self.device, dtype=dtype)
        else:
            array = np.asarray(grid)

            # If it's object dtype, try to convert to numeric
            if array.dtype == np.object_:
                logger.debug("WARNING: NumPy array has dtype=object. Attempting to convert to numeric dtype...")

                try:
                    # Try float conversion by default
                    array = array.astype(np.float32 if dtype.is_floating_point else np.int32)
                except Exception as e:
                    raise ValueError("ERROR: Failed to convert object array to numeric type.",array,"\nDetails:", e)

            # Convert to tensor
            try:
                tensor = torch.from_numpy(array).to(dtype)
            except Exception as e:
                raise ValueError("ERROR: torch.from_numpy failed.",array,"Details:", e)
            
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)     # shape becomes (1, 6, 6)

        return tensor.to(self.device)



    def load(self):
        self.policy_net.load_state_dict(torch.load(f'weights/{self.__class__.__name__}.pth'))
        self.policy_net.load_state_dict(torch.load(f'weights/{self.__class__.__name__}.pth'))

    def save(self):
        torch.save(self.policy_net.state_dict(), f'weights/{self.__class__.__name__}.pth')

    def show_structure(self):
        for name, param in self.policy_net.state_dict().items():
            print(name, param.shape)