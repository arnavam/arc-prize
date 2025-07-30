from collections import deque

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
                            and grid[nr][nc] == obj_color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                
                if not component:
                    continue
                    
                # Extract bounding box
                min_r = min(x[0] for x in component)
                max_r = max(x[0] for x in component)
                min_c = min(x[1] for x in component)
                max_c = max(x[1] for x in component)
                
                # Create object grid preserving original colors
                obj_grid = [[grid[min_r + i][min_c + j] 
                            for j in range(max_c - min_c + 1)]
                            for i in range(max_r - min_r + 1)]
                
                objects.append({
                    'grid': obj_grid,
                    'color': obj_color,
                    'position': (min_r, min_c),
                    'size': (len(obj_grid), len(obj_grid[0]))
                })
    return objects

def concat_h(grids, background=0):
    if not grids:
        return []
    
    # Calculate dimensions
    heights = [len(g) for g in grids if g]
    widths = [len(g[0]) for g in grids if g and g[0]]
    if not heights or not widths:
        return []
    
    total_height = max(heights)
    total_width = sum(widths)
    result = [[background] * total_width for _ in range(total_height)]
    
    # Paste grids side-by-side
    col_offset = 0
    for g in grids:
        if not g or not g[0]:
            continue
        h, w = len(g), len(g[0])
        for r in range(h):
            for c in range(w):
                result[r][col_offset + c] = g[r][c]
        col_offset += w
    return result

def concat_v(grids, background=0):
    if not grids:
        return []
    
    # Calculate dimensions
    heights = [len(g) for g in grids if g]
    widths = [len(g[0]) for g in grids if g and g[0]]
    if not heights or not widths:
        return []
    
    total_height = sum(heights)
    total_width = max(widths)
    result = [[background] * total_width for _ in range(total_height)]
    
    # Paste grids top-to-bottom
    row_offset = 0
    for g in grids:
        if not g or not g[0]:
            continue
        h, w = len(g), len(g[0])
        for r in range(h):
            for c in range(w):
                result[row_offset + r][c] = g[r][c]
        row_offset += h
    return result