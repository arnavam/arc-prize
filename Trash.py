def find_objects(grid, connectivity=4):
    if not grid or not grid[0]:
        return []
    
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    objects = []
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connectivity
    if connectivity == 8:
        dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                # Use starting pixel color as object color
                obj_color = grid[r][c]
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc, grid[cr][cc]))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols 
                            and not visited[nr][nc] 
                            and grid[nr][nc] == obj_color):  # Only same-color pixels
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                if not component:  # Skip empty components
                    continue
                    
                # Extract bounding box
                min_r = min(x[0] for x in component)
                max_r = max(x[0] for x in component)
                min_c = min(x[1] for x in component)
                max_c = max(x[1] for x in component)
                
                # Create object grid with original colors
                obj_grid = [[None] * (max_c - min_c + 1) 
                           for _ in range(max_r - min_r + 1)]
                
                # Fill object and background
                for i in range(len(obj_grid)):
                    for j in range(len(obj_grid[0])):
                        abs_r, abs_c = min_r + i, min_c + j
                        if (abs_r, abs_c, grid[abs_r][abs_c]) in component:
                            obj_grid[i][j] = grid[abs_r][abs_c]
                        else:
                            obj_grid[i][j] = grid[abs_r][abs_c]  # Preserve background colors
                
                objects.append((obj_grid, min_r, min_c))
    return objects



###############################################################


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

# features = extract_features(objects, random_arrangement, output_grid)

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
    

# Neural network for arrangement scoring

    




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

# using prob based pos

        def select_action(self, input_grid, current_grid, ):
        self.policy.train()  # Optional

     
        input_tensor = self._preprocess_to_tensor(input_grid)     # "state" / input
        current_tensor = self._preprocess_to_tensor(current_grid) # "target" / current grid

       
        action_probs = self.policy([input_tensor, current_tensor])  # Shape: [num_actions]
        
        # Create a categorical distribution over actions
        dist = Categorical(action_probs)
        action_index = dist.sample()

        primitive_name = PRIMITIVE_NAMES[action_index.item()]
        new_grid = PRIMITIVE[primitive_name](current_grid.copy())  # Safe to copy before changing

        self.states.append(input_grid.copy(), current_grid.copy())
        self.actions.append(action_index.item())
        self.log_probs.append(dist.log_prob(action_index))

        return new_grid, action_index.item()
    

# def extract_region(grid, position, obj_shape):
#     grid = np.atleast_2d(grid)  # Ensure grid is at least 2D (converts 0D/1D to 2D)
#     r, c = position
#     h, w = obj_shape
    
#     # Handle cases where the region is out of bounds
#     grid_rows, grid_cols = grid.shape
#     if r + h > grid_rows or c + w > grid_cols:
#         raise ValueError("Requested region exceeds grid dimensions")

#     return grid[r:r+h, c:c+w].copy()



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
    
    