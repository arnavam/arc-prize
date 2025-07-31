import numpy as np
import json
import math
import seaborn as sns
import random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor  # Simple neurosymbolic model
from matplotlib  import colors 
from matplotlib import pyplot as plt
from dsl2 import convert_np_to_native
from dsl import find_objects , PRIMITIVE
from neurosymbolic_torch import NeuralSymbolicSolverRL,FeatureExtractor
import torch
model = FeatureExtractor(input_channels=1)

def shift_up(grid):
    return np.roll(grid, shift=-1, axis=0)

def shift_down(grid):
    return np.roll(grid, shift=1, axis=0)

def shift_left(grid):
    return np.roll(grid, shift=-1, axis=1)

def shift_right(grid):
    return np.roll(grid, shift=1, axis=1)

ACTIONS = {
    "shift_up": shift_up,
    "shift_down": shift_down,
    "shift_left": shift_left,
    "shift_right": shift_right
}



# This gives you a list of names to index from
ACTION_NAMES = list(ACTIONS.keys())
num_action=len(ACTIONS)
class Spacial_Network(nn.Module):
    """
    The policy network decides which action to take.
    It takes the state (current grid + target grid) and outputs action probabilities.
    """
    def __init__(self, feature_extractor, num_action):
        super().__init__()
        self.feature_extractor = feature_extractor
        # Combined feature vector size is 128 (current) + 128 (target)
        self.policy_head = nn.Linear(128 * 2, num_action)
        
    def forward(self, inputs):
        current_grid_tensor, target_grid_tensor = inputs
        current_feat = self.feature_extractor(current_grid_tensor)
        target_feat = self.feature_extractor(target_grid_tensor)
        
        combined = torch.cat([current_feat, target_feat], dim=-1)
        # Output logits, which will be converted to probabilities
        action_logits = self.policy_head(combined)
        return F.softmax(action_logits, dim=-1)

    

# Neural network for arrangement scoring
class ArrangementScorer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def train(self, X, y):

        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, features):
        if not self.is_trained:
            return random.random()  # Random score if not trained
        return self.model.predict([features])[0]
    


# MCTS Node for arrangement search
class MCTSNode:
    def __init__(self, objects, output_grid, background, parent=None):
        self.objects = objects  # Unplaced objects
        self.arrangement = {}  # {object_id: (row, col)}
        self.output_grid = output_grid
        self.background = background
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_score = 0.0
        self.scorer = parent.scorer if parent else ArrangementScorer()
        
    def ucb_score(self, exploration=1.4):
        if self.visits == 0:
            return float('inf')
        return (self.total_score / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def is_terminal(self):
        return len(self.objects) == 0
    
def expand(self):
    if not self.objects:
        return

    obj = self.objects[0]
    H, W = len(self.output_grid), len(self.output_grid[0])
    
    # Step 1: Generate all transformations of the object
    transformations = PRIMITIVES(obj['grid'])
    
    for transformed_grid in transformations:
        th, tw = transformed_grid.shape  # Transformed height/width
        
        # Step 2: Generate candidate positions (grid-aligned)
        positions = [
            (r, c) 
            for r in range(H - th + 1) 
            for c in range(W - tw + 1)
        ]
        
        # Step 3: Random sampling for efficiency (optional)
        if len(positions) > 20:
            positions = random.sample(positions, 20)
        
        # Step 4: For each position, query the model to check compatibility
        for pos in positions:
            r, c = pos
            # Extract the region under the object
            region_under = [
                [self.output_grid[r + i][c + j] 
                for j in range(tw)]
                for i in range(th)
            ]
            
            # Step 5: Use a pre-trained model to predict compatibility
            # (Assuming `model.predict()` returns a score or True/False)
            is_compatible = model.predict(
                object=transformed_grid,
                region=region_under,
                background=self.background
            )
            
            if is_compatible:  # Only proceed if the model approves
                new_objects = self.objects[1:]
                new_node = MCTSNode(
                    new_objects, 
                    self.output_grid, 
                    self.background, 
                    parent=self
                )
                new_node.arrangement = self.arrangement.copy()
                new_node.arrangement[id(obj)] = {
                    'position': pos,
                    'grid': transformed_grid.tolist(),  # Store transformed grid
                    'size': (th, tw)
                }
                self.children.append(new_node)

    def simulate(self):
        # Create temporary arrangement with all objects placed
        temp_arrangement = self.arrangement.copy()
        remaining_objects = self.objects.copy()
        
        # Place remaining objects randomly
        H, W = len(self.output_grid), len(self.output_grid[0])
        for obj in remaining_objects:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                r = random.randint(0, H - obj['size'][0])
                c = random.randint(0, W - obj['size'][1])
                temp_arrangement[id(obj)] = (r, c)
                placed = True  # Simple version - skip collision detection
                attempts += 1
        



        input_tensor = torch.tensor(self.output_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
        # shape: (batch_size=1, channels=1, height, width)

        features = model(input_tensor)
        return self.scorer.predict(features.detach().cpu().numpy().flatten())
    
    def backpropagate(self, score):
        node = self
        while node:
            node.visits += 1
            node.total_score += score
            node = node.parent


# MCTS for object arrangement
def arrange_objects_mcts(input_grid, output_grid, iterations=500):
    # 1. Extract objects and background
    objects = find_objects(input_grid)
    flat_output = np.array(output_grid).flatten()
    background = np.bincount(flat_output).argmax()
    
    # 2. Initialize MCTS
    root = MCTSNode(objects, output_grid, background)
    root.expand()  # Initial expansion
    neuro = NeuralSymbolicSolverRL()
    nueuro(object,output_grid)
    # 3. Train scorer with initial samples
    X_train, y_train = [], []
    for _ in range(100):
        random_arrangement = {}
        for obj in objects:
            r = random.randint(0, len(output_grid) - obj['size'][0])
            c = random.randint(0, len(output_grid[0]) - obj['size'][1])
            random_arrangement[id(obj)] = (r, c)
        

        input_tensor = torch.tensor(output_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  
        # shape: (batch_size=1, channels=1, height, width)
        features = model(input_tensor).detach().cpu().numpy().flatten()


        # Simple scoring: coverage of non-background areas
        coverage_score = 0
        for obj in objects:
            r, c = random_arrangement[id(obj)]
            for i in range(obj['size'][0]):
                for j in range(obj['size'][1]):
                    if output_grid[r+i][c+j] != background:
                        coverage_score += 1
        y_train.append(coverage_score / (len(output_grid)*len(output_grid[0])))
        X_train.append(features)

    
    # root.scorer.train(X_train, y_train)
    
    # 4. Run MCTS
    for _ in range(iterations):
        node = root
        
        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score())
        
        # Expansion
        if not node.is_terminal() and not node.children:
            node.expand()
            if node.children:
                node = random.choice(node.children)
        
        # Simulation
        score = node.simulate()
        
        # Backpropagation
        node.backpropagate(score)
    
    # 5. Find best arrangement
    best_node = max(root.children, key=lambda n: n.visits)
    
    # 6. Create output grid from arrangement
    output = [[background for _ in range(len(output_grid[0]))] 
              for _ in range(len(output_grid))]
    
    for obj in objects:
        if id(obj) in best_node.arrangement:
            r, c = best_node.arrangement[id(obj)]
            for i in range(len(obj['grid'])):
                for j in range(len(obj['grid'][0])):
                    if obj['grid'][i][j] != background:  # Only place non-background
                        if 0 <= r+i < len(output) and 0 <= c+j < len(output[0]):
                            output[r+i][c+j] = obj['grid'][i][j]
    
    return output, best_node.total_score / best_node.visits



if __name__ == '__main__':
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
            '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    ids=[]
    train_path='arc-prize-2025/arc-agi_training_challenges.json'
    with open(train_path, 'r') as f:
        train = json.load(f)

    for case_id in train:
        ids.append(case_id) 
    count=0
    for case_id in train:
        count +=1
        if count ==2 :
            break
        for i in range(2):

            for j in ('input','output'):
                a=train[case_id]['train'][i]['input']
                b=train[case_id]['train'][i]['output']
                print('input')
                sns.heatmap(a,cmap=cmap)
                plt.show()

        # a=np.array(a)
        # b=np.array(b)



        # Solve the puzzle using the new method
        solved_grid ,_= arrange_objects_mcts(a, b,10)
        solved_grid = convert_np_to_native(solved_grid)
        print('original')

        sns.heatmap(b,cmap=cmap)
        plt.show()

        print('predicted')
        sns.heatmap(solved_grid,cmap=cmap)
        plt.show()



            # Verify if the solution is correct
        is_correct = np.array_equal(solved_grid, b)
        print(f"\nSolution is correct: {is_correct}")



