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
from RL_alg.reinforce import NeuralSymbolicSolverRL,FeatureExtractor
import torch
model = FeatureExtractor(input_channels=1)

import numpy as np

def move_up(grid, position):
    grid = grid.copy()
    x, y = position
    if x > 0 and grid[x - 1, y] == 0:
        grid[x, y], grid[x - 1, y] = 0, grid[x, y]
        return grid, (x - 1, y)
    return grid, position

def move_down(grid, position):
    grid = grid.copy()
    x, y = position
    if x < grid.shape[0] - 1 and grid[x + 1, y] == 0:
        grid[x, y], grid[x + 1, y] = 0, grid[x, y]
        return grid, (x + 1, y)
    return grid, position

def move_left(grid, position):
    grid = grid.copy()
    x, y = position
    if y > 0 and grid[x, y - 1] == 0:
        grid[x, y], grid[x, y - 1] = 0, grid[x, y]
        return grid, (x, y - 1)
    return grid, position

def c(grid, position):
    grid = grid.copy()
    x, y = position
    if y < grid.shape[1] - 1 and grid[x, y + 1] == 0:
        grid[x, y], grid[x, y + 1] = 0, grid[x, y]
        return grid, (x, y + 1)
    return grid, position


def idle (grid,object):
    return grid

ACTIONS = {
    "move_up": move_up,
    "move_down": move_down,
    "move_left": move_left,
    "move_left": move_left ,
    'idle':idle
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

    
    def expand(self,objects):
        if not self.objects:
            return

        # Step 1: Randomly decide whether to place an object or not (50% chance)
        if random.random() < 0.5:
            # Choose not to place anything - create one child node with same objects
            new_node = MCTSNode(
                objects.copy(),  # Keep same objects
                self.output_grid.copy(),
                self.background,
                parent=self
            )
            new_node.arrangement = self.arrangement.copy()
            self.children.append(new_node)
            return

        # Step 2: Randomly select an object to place
        obj = random.choice(self.objects)
        obj_grid = obj['grid']
        th, tw = len(obj_grid), len(obj_grid[0])  # Object height/width
        
        H, W = len(self.output_grid), len(self.output_grid[0])
        
        # Step 3: Generate all possible valid positions
        possible_positions = [
            (r, c) 
            for r in range(H - th + 1) 
            for c in range(W - tw + 1)
        ]
        
        # If no valid positions, skip this object
        if not possible_positions:
            return
        
        # Step 4: Randomly select one position to place it
        r, c = random.choice(possible_positions)
        
        # Step 5: Create new node with this placement
        new_objects = [o for o in self.objects if o != obj]  # Remove placed object
        new_node = MCTSNode(
            new_objects, 
            self.output_grid.copy(), 
            self.background, 
            parent=self
        )
        new_node.arrangement = self.arrangement.copy()
        new_node.arrangement[id(obj)] = {
            'position': (r, c),
            'grid': obj_grid,
            'size': (th, tw)
        }
        
        # Step 6: Actually place the object in the output grid
        for i in range(th):
            for j in range(tw):
                if obj_grid[i][j] != 0:  # Assuming 0 is empty/transparent
                    new_node.output_grid[r + i][c + j] = obj_grid[i][j]
        
        self.children.append(new_node)

    def simulate(self, neuro,spacial):
        temp_arrangement = self.arrangement.copy()
        for self.objects in self.objects:
            obj_id = id(self.objects)
            score = evaluate_placement(self, obj_id, neuro,spacial,self.node)  # Neuro-guided scoring
            if score is not None:
                temp_arrangement[obj_id]["score"] = score
        return score
        



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

def extract_region_under_object(output_grid, object_matrix, pos):
    x, y = pos  # Position (x, y) in the output_grid
    h, w = object_matrix.shape  # Height and width of the object
    
    # Extract the region from output_grid where the object is placed
    region = output_grid[x:x+h, y:y+w]
    
    return region

def get_object_position(node, obj):
    obj_id = id(obj)
    if obj_id in node.arrangement:
        return node.arrangement[obj_id]
    else:
        return None  
    
def matrix_similarity(predicted, original, weight_pattern=0.5, weight_color=0.5):
    if predicted.shape != original.shape:
        return 0.0  # Completely different
    
    # Pattern match (exact values match at same positions)
    pattern_match = (predicted == original).astype(np.float32)
    pattern_score = pattern_match.mean()  # Fraction of matching positions
    
    # Color similarity (normalized value difference)
    max_val = 9.0  # Assuming values range from 0 to 9
    diff = np.abs(predicted - original)
    color_similarity = 1 - (diff / max_val)  # 1 = perfect match, 0 = max difference
    color_score = color_similarity.mean()
    
    # Weighted total score
    total_score = weight_pattern * pattern_score + weight_color * color_score
    return total_score


# MCTS for object arrangement
def arrange_objects_mcts(input_grid, output_grid, iterations=500):
    # 1. Extract objects and background
    objects = find_objects(input_grid)
    output_grid=np.array(output_grid)
    input_grid=np.array(input_grid)
    flat_output = np.array(output_grid).flatten()
    background = np.bincount(flat_output).argmax()

    # 2. Initialize MCTS
    root = MCTSNode(objects, output_grid, background)
    root.expand(objects)  # Initial expansion
    feature_extractor=FeatureExtractor()
    neuro = NeuralSymbolicSolverRL(feature_extractor)
    spacial = Spacial_Network(feature_extractor,num_action)

    ## select random object - simultinously or not?
    objector = objects[1]
    ## select a random pos
    x=np.random.randint(output_grid.shape[0])
    y=np.random.randint(output_grid.shape[1])

    

    node= root
    # 4. Run MCTS
    for _ in range(iterations):

        
        # Selection
        while node.children:
            node = max(node.children, key=lambda n: n.ucb_score())
        
        # Expansion
        if not node.is_terminal() and not node.children:

            node.expand(objects)
  

            
            if node.children:
                node = random.choice(node.children)
                
        # Simulation
        score = node.simulate(neuro,spacial)
        
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

def evaluate_placement(output_grid, neuro, spacial, current_grid, node):
    selection = get_object_position(node,id)
    move=spacial.forward((current_grid,output_grid))
    current_grid= ACTIONS[ACTION_NAMES[move]](current_grid,selection['position'])
    selection['grid'] , action=neuro.select_action(selection['grid'],current_grid)
            
    frame = extract_region_under_object(output_grid,selection['grid'])
    score1=matrix_similarity(selection['grid'],frame) ##not yet implemented
    score2=matrix_similarity(current_grid,output_grid)
    neuro.store_reward(score1)
    spacial.store_reward(score2)



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
                # sns.heatmap(a,cmap=cmap)
                # plt.show()

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



