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
    
