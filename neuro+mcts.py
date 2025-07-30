import numpy as np
import json
import math
from collections import deque
import random
from sklearn.ensemble import RandomForestRegressor  # Simple neurosymbolic model
from   matplotlib  import colors 
from dsl2 import convert_np_to_native
from DSL import find_objects
from neurosymbolic_torch import NeuralSymbolicSolverRL



# Neural network for arrangement scoring
class ArrangementScorer:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False
        
    def train(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, features):
        if not self.is_trained:
            return random.random()  # Random score if not trained
        return self.model.predict([features])[0]
    

# Feature extraction for neural network
def extract_features(objects, arrangement, output_grid):
    features = []
    H, W = len(output_grid), len(output_grid[0])
    
    # 1. Coverage ratio
    covered = sum(obj['size'][0] * obj['size'][1] for obj in objects)
    total_area = H * W
    features.append(covered / total_area)
    
    # 2. Positional variance
    avg_x = sum(pos[1] for _, pos in arrangement.items()) / len(arrangement)
    avg_y = sum(pos[0] for _, pos in arrangement.items()) / len(arrangement)
    var_x = sum((pos[1] - avg_x)**2 for _, pos in arrangement.items())
    var_y = sum((pos[0] - avg_y)**2 for _, pos in arrangement.items())
    features.append(var_x / (W**2))
    features.append(var_y / (H**2))
    
    # 3. Color distribution
    output_colors = set(np.array(output_grid).flatten())
    input_colors = set(obj['color'] for obj in objects)
    features.append(len(input_colors & output_colors) / len(output_colors | input_colors))
    
    # 4. Edge alignment
    edge_count = 0
    for _, pos in arrangement.items():
        if pos[0] == 0 or pos[0] == H-1 or pos[1] == 0 or pos[1] == W-1:
            edge_count += 1
    features.append(edge_count / len(arrangement))
    
    return features

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
        
        # Generate candidate positions (grid-aligned)
        positions = []
        for r in range(H - obj['size'][0] + 1):
            for c in range(W - obj['size'][1] + 1):
                positions.append((r, c))
        
        # Random sampling for efficiency
        if len(positions) > 20:
            positions = random.sample(positions, 20)
            
        for pos in positions:
            new_objects = self.objects[1:]
            new_node = MCTSNode(new_objects, self.output_grid, 
                                self.background, self)
            new_node.arrangement = self.arrangement.copy()
            new_node.arrangement[id(obj)] = pos
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
        
        # Score the arrangement
        features = extract_features(self.parent.objects if self.parent else self.objects,
                                   temp_arrangement, self.output_grid)
        return self.scorer.predict(features)
    
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
    
    # 3. Train scorer with initial samples
    X_train, y_train = [], []
    for _ in range(100):
        random_arrangement = {}
        for obj in objects:
            r = random.randint(0, len(output_grid) - obj['size'][0])
            c = random.randint(0, len(output_grid[0]) - obj['size'][1])
            random_arrangement[id(obj)] = (r, c)
        
        features = extract_features(objects, random_arrangement, output_grid)
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
    
    root.scorer.train(X_train, y_train)
    
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

    for case_id in train:
        for i in range(2):
            for j in ('input','output'):
                a=train[case_id]['train'][i]['input']
                b=train[case_id]['train'][i]['output']
                # print(a)
                # sns.heatmap(a,cmap=cmap)
                # plt.show()

        # a=np.array(a)
        # b=np.array(b)



        # Solve the puzzle using the new method
        solved_grid ,_= arrange_objects_mcts(a, b)
        # solved_grid = convert_np_to_native(solved_grid)
        solved_grid = convert_np_to_native(solved_grid[1])

        print(solved_grid)

        print("\nSolved Grid:")
        for row in solved_grid:
            print(row )
            # Verify if the solution is correct
        is_correct = np.array_equal(solved_grid, b)
        print(f"\nSolution is correct: {is_correct}")



