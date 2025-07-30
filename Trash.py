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


