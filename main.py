import pygame
import random
import math
from queue import PriorityQueue, Queue

# --- CONFIG ---
WIDTH = 960
HEIGHT = 640
ROWS = 80
COLS = 120
BLOCKED_COUNT = 2000
LEGEND_WIDTH = 200
GRID_WIDTH = WIDTH - LEGEND_WIDTH
CELL_WIDTH = GRID_WIDTH // COLS
CELL_HEIGHT = HEIGHT // ROWS

# --- COLORS ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
SKY = (135, 206, 235)
PINK = (255, 105, 180)
BLUE = (0, 0, 255)
LIGHT_GRAY = (200, 200, 200) 
DARK_GRAY = (50, 50, 50)
BACKGROUND_COLOR = (240, 240, 240)
BORDER_COLOR = (80, 80, 80)
GRID_BG_COLOR = (245, 245, 245)
LEGEND_BG_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (255, 255, 0)

pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pathfinding Visualizer")

# Global variables to control algorithm execution
running_algorithm = False
algorithm_paused = False
current_algorithm = None
algorithm_thread = None

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_WIDTH
        self.y = row * CELL_HEIGHT
        self.color = WHITE
        self.neighbors = []
        self.blocked = False
        self.border_thickness = 1  # Added border thickness

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == GREY

    def is_open(self):
        return self.color == SKY

    def is_blocked(self):
        return self.blocked

    def is_start(self):
        return self.color == GREEN

    def is_end(self):
        return self.color == RED

    def reset(self):
        if not self.is_start() and not self.is_end():
            self.color = WHITE
        self.blocked = False

    def make_start(self):
        self.color = GREEN

    def make_closed(self):
        self.color = GREY

    def make_open(self):
        self.color = SKY

    def make_barrier(self):
        self.color = BLACK
        self.blocked = True

    def make_end(self):
        self.color = RED

    def make_path(self):
        self.color = PINK

    def draw(self, win):
        # Draw filled rectangle with a border
        pygame.draw.rect(win, self.color, (self.x, self.y, CELL_WIDTH, CELL_HEIGHT))
        pygame.draw.rect(win, DARK_GRAY, (self.x, self.y, CELL_WIDTH, CELL_HEIGHT), self.border_thickness)

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for d in directions:
            r, c = self.row + d[0], self.col + d[1]
            if 0 <= r < ROWS and 0 <= c < COLS and not grid[r][c].is_blocked():
                self.neighbors.append(grid[r][c])

def draw_grid_lines(win):
    # Draw a clean separator between grid and legend
    pygame.draw.line(win, BORDER_COLOR, (GRID_WIDTH, 0), (GRID_WIDTH, HEIGHT), 2)
    
    # Draw subtle grid lines only in the grid area
    for i in range(ROWS + 1):
        pygame.draw.line(win, LIGHT_GRAY, (0, i * CELL_HEIGHT), (GRID_WIDTH, i * CELL_HEIGHT))
    for j in range(COLS + 1):
        pygame.draw.line(win, LIGHT_GRAY, (j * CELL_WIDTH, 0), (j * CELL_WIDTH, HEIGHT))

def draw(win, grid, message=""):
    # Fill background
    win.fill(GRID_BG_COLOR, (0, 0, GRID_WIDTH, HEIGHT))
    win.fill(LEGEND_BG_COLOR, (GRID_WIDTH, 0, LEGEND_WIDTH, HEIGHT))
    
    # Draw grid nodes
    for row in grid:
        for node in row:
            node.draw(win)
    
    # Draw grid lines and borders
    draw_grid_lines(win)
    
    # Draw main border around the entire window
    pygame.draw.rect(win, BORDER_COLOR, (0, 0, WIDTH, HEIGHT), 3)
    
    # Draw grid border
    pygame.draw.rect(win, BORDER_COLOR, (0, 0, GRID_WIDTH, HEIGHT), 2)
    
    # Draw legend
    draw_legend(win, message)
    
    pygame.display.update()

def heuristic(a, b):
    return abs(a.row - b.row) + abs(a.col - b.col)

def reconstruct_path(came_from, current, draw_func):
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw_func()
        # Check if algorithm should stop
        if check_stop_algorithm():
            break

def reset_grid(grid, start, end):
    for row in grid:
        for node in row:
            if node != start and node != end and not node.is_blocked():
                node.color = WHITE

def make_grid():
    return [[Node(i, j) for j in range(COLS)] for i in range(ROWS)]

def check_stop_algorithm():
    global running_algorithm, algorithm_paused
    
    # Check for quit or stop events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_c:
                running_algorithm = False
                return True
            elif event.key == pygame.K_SPACE:
                algorithm_paused = not algorithm_paused
    
    # Pause the algorithm if needed
    while algorithm_paused and running_algorithm:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_c:
                    running_algorithm = False
                    return True
                elif event.key == pygame.K_SPACE:
                    algorithm_paused = not algorithm_paused
                    break
        pygame.time.delay(100)
    
    return not running_algorithm

def dijkstra(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    open_set_hash = {start}

    while not open_set.empty() and running_algorithm:
        if check_stop_algorithm():
            break

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            start.make_start()
            running_algorithm = False
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((g_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    running_algorithm = False
    return False

def astar(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    f_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score[start] = heuristic(start, end)
    open_set_hash = {start}

    while not open_set.empty() and running_algorithm:
        if check_stop_algorithm():
            break

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            start.make_start()
            running_algorithm = False
            return True

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    running_algorithm = False
    return False

def dfs(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    stack = [start]
    came_from = {}
    visited = set()
    visited.add(start)

    while stack and running_algorithm:
        if check_stop_algorithm():
            break

        current = stack.pop()

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            start.make_start()
            running_algorithm = False
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                came_from[neighbor] = current
                visited.add(neighbor)
                stack.append(neighbor)
                neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    running_algorithm = False
    return False

def greedy(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    count = 0
    open_set = PriorityQueue()
    open_set.put((heuristic(start, end), count, start))
    came_from = {}
    visited = {start}

    while not open_set.empty() and running_algorithm:
        if check_stop_algorithm():
            break

        current = open_set.get()[2]

        if current == end:
            reconstruct_path(came_from, end, draw_func)
            end.make_end()
            start.make_start()
            running_algorithm = False
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                count += 1
                open_set.put((heuristic(neighbor, end), count, neighbor))
                neighbor.make_open()

        draw_func()
        if current != start:
            current.make_closed()

    running_algorithm = False
    return False

def line_of_sight(grid, node1, node2):
    x0, y0 = node1.col, node1.row
    x1, y1 = node2.col, node2.row
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while x0 != x1 or y0 != y1:
        if y0 < 0 or y0 >= ROWS or x0 < 0 or x0 >= COLS or grid[y0][x0].is_blocked():
            return False
        
        # Avoid sneaky diagonal corner cuts
        if (x0 != x1 and y0 != y1):
            if (0 <= y0 < ROWS and 0 <= x0 - sx < COLS and grid[y0][x0 - sx].is_blocked() and 
                0 <= y0 - sy < ROWS and 0 <= x0 < COLS and grid[y0 - sy][x0].is_blocked()):
                return False

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return not grid[y1][x1].is_blocked()

def bidirectional_a_star(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)

    open_set_start = PriorityQueue()
    open_set_end = PriorityQueue()
    count = 0

    g_score_start = {node: float('inf') for row in grid for node in row}
    g_score_end = {node: float('inf') for row in grid for node in row}

    g_score_start[start] = 0
    g_score_end[end] = 0

    came_from_start = {}
    came_from_end = {}

    open_set_start.put((0, count, start))
    open_set_end.put((0, count, end))

    visited_start = set()
    visited_end = set()

    meeting_node = None

    while not open_set_start.empty() and not open_set_end.empty() and running_algorithm:
        if check_stop_algorithm():
            break

        _, _, current_start = open_set_start.get()
        _, _, current_end = open_set_end.get()

        visited_start.add(current_start)
        visited_end.add(current_end)

        if current_start in visited_end:
            meeting_node = current_start
            break
        if current_end in visited_start:
            meeting_node = current_end
            break

        # Expand from start side
        for neighbor in current_start.neighbors:
            temp_g = g_score_start[current_start] + 1
            if temp_g < g_score_start[neighbor]:
                came_from_start[neighbor] = current_start
                g_score_start[neighbor] = temp_g
                f_score = temp_g + heuristic(neighbor, end)
                count += 1
                open_set_start.put((f_score, count, neighbor))
                if neighbor not in visited_start:
                    neighbor.make_open()

        # Expand from end side
        for neighbor in current_end.neighbors:
            temp_g = g_score_end[current_end] + 1
            if temp_g < g_score_end[neighbor]:
                came_from_end[neighbor] = current_end
                g_score_end[neighbor] = temp_g
                f_score = temp_g + heuristic(neighbor, start)
                count += 1
                open_set_end.put((f_score, count, neighbor))
                if neighbor not in visited_end:
                    neighbor.make_open()

        if current_start != start:
            current_start.make_closed()
        if current_end != end:
            current_end.make_closed()

        draw_func()

    if meeting_node and running_algorithm:
        # Reconstruct both halves
        path1 = []
        node = meeting_node
        while node in came_from_start:
            path1.append(node)
            node = came_from_start[node]
        path1.append(start)
        path1.reverse()

        path2 = []
        node = meeting_node
        while node in came_from_end:
            node = came_from_end[node]
            path2.append(node)

        full_path = path1 + path2

        for node in full_path:
            if check_stop_algorithm():
                break
            if node != start and node != end:
                node.make_path()
            draw_func()
        
        running_algorithm = False
        return True

    running_algorithm = False
    return False

def theta_star(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    open_set = PriorityQueue()
    count = 0
    came_from = {}
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0
    parent = {start: start}

    open_set.put((0, count, start))

    while not open_set.empty() and running_algorithm:
        if check_stop_algorithm():
            break

        _, _, current = open_set.get()

        if current == end:
            # Reconstruct using parent to show Theta* jumps
            node = end
            while node != start:
                if check_stop_algorithm():
                    break
                if node != end:
                    node.make_path()
                node = parent[node]
                draw_func()
            
            running_algorithm = False
            return True

        for neighbor in current.neighbors:
            if line_of_sight(grid, parent[current], neighbor):
                tentative_g = g_score[parent[current]] + math.dist(parent[current].get_pos(), neighbor.get_pos())
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = parent[current]
                    parent[neighbor] = parent[current]
                    count += 1
                    open_set.put((tentative_g + heuristic(neighbor, end), count, neighbor))
                    neighbor.make_open()
            else:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    parent[neighbor] = current
                    count += 1
                    open_set.put((tentative_g + heuristic(neighbor, end), count, neighbor))
                    neighbor.make_open()

        if current != start:
            current.make_closed()
        draw_func()

    running_algorithm = False
    return False

def bidirectional_search(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    q_start = Queue()
    q_end = Queue()
    came_from_start = {start: None}
    came_from_end = {end: None}
    visited_start = {start}
    visited_end = {end}

    q_start.put(start)
    q_end.put(end)

    meet_node = None

    while not q_start.empty() and not q_end.empty() and not meet_node and running_algorithm:
        if check_stop_algorithm():
            break

        # Process from start
        current = q_start.get()
        if current in visited_end:
            meet_node = current
            break

        for neighbor in current.neighbors:
            if neighbor not in visited_start:
                visited_start.add(neighbor)
                came_from_start[neighbor] = current
                q_start.put(neighbor)
                neighbor.make_open()

        if current != start:
            current.make_closed()
        
        # Process from end
        if not meet_node:
            current = q_end.get()
            if current in visited_start:
                meet_node = current
                break

            for neighbor in current.neighbors:
                if neighbor not in visited_end:
                    visited_end.add(neighbor)
                    came_from_end[neighbor] = current
                    q_end.put(neighbor)
                    neighbor.make_open()

            if current != end:
                current.make_closed()
        
        draw_func()

    if meet_node and running_algorithm:
        # Reconstruct path from start to meet point
        current = meet_node
        while current in came_from_start and came_from_start[current] is not None:
            if check_stop_algorithm():
                break
            current.make_path()
            current = came_from_start[current]
            draw_func()

        # Reconstruct path from end to meet point
        current = meet_node
        while current in came_from_end and came_from_end[current] is not None:
            if check_stop_algorithm():
                break
            current.make_path()
            current = came_from_end[current]
            draw_func()
            
        end.make_end()
        start.make_start()
        running_algorithm = False
        return True

    running_algorithm = False
    return False

def flow_field(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    # Step 1: Compute cost-to-go from every node to the end using BFS
    cost = {node: float('inf') for row in grid for node in row}
    cost[end] = 0
    queue = Queue()
    queue.put(end)

    while not queue.empty() and running_algorithm:
        if check_stop_algorithm():
            break
            
        current = queue.get()
        for neighbor in current.neighbors:
            if cost[neighbor] == float('inf'):
                cost[neighbor] = cost[current] + 1
                queue.put(neighbor)
                neighbor.make_open()
        if current != end:
            current.make_closed()
        draw_func()

    # Step 2: Build flow directions
    flow_map = {}
    for row in grid:
        for node in row:
            if node != end and not node.is_blocked() and cost[node] != float('inf'):
                min_cost = cost[node]
                next_step = None
                for neighbor in node.neighbors:
                    if cost.get(neighbor, float('inf')) < min_cost:
                        min_cost = cost[neighbor]
                        next_step = neighbor
                if next_step:
                    flow_map[node] = next_step

    # Step 3: Follow flow from start
    current = start
    while current != end and running_algorithm:
        if check_stop_algorithm():
            break
            
        if current != start:
            current.make_path()
        current = flow_map.get(current)
        if not current:
            running_algorithm = False
            return False
        draw_func()

    running_algorithm = False
    return True

def ant_colony(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    pheromones = {node: 1.0 for row in grid for node in row if not node.is_blocked()}
    num_ants = 100
    best_path = None

    def construct_path():
        if check_stop_algorithm():
            return None
            
        path = [start]
        current = start
        visited = set([start])
        while current != end:
            if not current.neighbors:
                return None
            next_nodes = [n for n in current.neighbors if n not in visited and not n.is_blocked()]
            if not next_nodes:
                return None
            weights = [pheromones.get(n, 1.0) for n in next_nodes]
            current = random.choices(next_nodes, weights=weights)[0]
            path.append(current)
            visited.add(current)
        return path

    for _ in range(num_ants):
        if not running_algorithm:
            break
            
        path = construct_path()
        if path:
            for node in path:
                pheromones[node] += 1.0
            if not best_path or len(path) < len(best_path):
                best_path = path

    if best_path and running_algorithm:
        for node in best_path:
            if check_stop_algorithm():
                break
                
            if node != start and node != end:
                node.make_path()
            draw_func()
        running_algorithm = False
        return True
        
    running_algorithm = False
    return False

def jps(draw, grid, start, end):
    reset_grid(grid, start, end)
    open_set = PriorityQueue()
    came_from = {}
    g_score = {node: float('inf') for row in grid for node in row}
    g_score[start] = 0
    count = 0
    open_set.put((0, count, start))
    _, _, current = open_set.get()

    def jump(current, dx, dy):
        x, y = current.col, current.row
        while True:
            x += dx
            y += dy
            if not (0 <= x < COLS and 0 <= y < ROWS):
                return None
            node = grid[y][x]
            if node.is_blocked():
                return None
            if node == end:
                return node
            # Check for forced neighbors
            if dx != 0 and dy != 0:  # Diagonal
                if (not grid[y - dy][x].is_blocked() and grid[y - dy][x - dx].is_blocked()) or \
                   (not grid[y][x - dx].is_blocked() and grid[y - dy][x - dx].is_blocked()):
                    return node
            elif dx != 0:  # Horizontal
                if (not grid[y + 1][x].is_blocked() and grid[y + 1][x - dx].is_blocked()) or \
                   (not grid[y - 1][x].is_blocked() and grid[y - 1][x - dx].is_blocked()):
                    return node
            elif dy != 0:  # Vertical
                if (not grid[y][x + 1].is_blocked() and grid[y - dy][x + 1].is_blocked()) or \
                   (not grid[y][x - 1].is_blocked() and grid[y - dy][x - 1].is_blocked()):
                    return node

            if dx != 0 and dy != 0:
                if jump(node, dx, 0) or jump(node, 0, dy):
                    return node

    def successors(node):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (1, -1), (-1, 1), (1, 1)]
        result = []
        for dx, dy in dirs:
            next_node = jump(node, dx, dy)
            if next_node:
                result.append(next_node)
        return result

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        _, current = open_set.get()

        if current == end:
            while current in came_from:
                current = came_from[current]
                if current != start:
                    current.make_path()
                draw()
            return True

        for neighbor in successors(current):
            temp_g = g_score[current] + heuristic(current, neighbor)
            if temp_g < g_score[neighbor]:
                g_score[neighbor] = temp_g
                came_from[neighbor] = current
                count += 1
                open_set.put((temp_g + heuristic(neighbor, end), count, neighbor))
                neighbor.make_open()

        if current != start:
            current.make_closed()
        draw()

    return False

def hpa_star(draw_func, grid, start, end):
    global running_algorithm
    running_algorithm = True
    
    reset_grid(grid, start, end)
    block_size = 10
    clusters = {}

    for r in range(0, ROWS, block_size):
        for c in range(0, COLS, block_size):
            if check_stop_algorithm():
                running_algorithm = False
                return False
                
            clusters[(r, c)] = []
            for i in range(r, min(r + block_size, ROWS)):
                for j in range(c, min(c + block_size, COLS)):
                    node = grid[i][j]
                    if not node.is_blocked():
                        clusters[(r, c)].append(node)

    # Fallback to A* for now on abstract map
    result = astar(draw_func, grid, start, end)
    running_algorithm = False
    return result

def draw_legend(win, message=""):
    font_title = pygame.font.SysFont('Arial', 20, bold=True)  # Reduced font size
    font_normal = pygame.font.SysFont('Arial', 18)
    font_small = pygame.font.SysFont('Arial', 16)
    
    # Draw legend title with nice styling
    title_text = "PATHFINDING"  # Split into two lines
    subtitle_text = "VISUALIZER"
    title = font_title.render(title_text, True, DARK_GRAY)
    subtitle = font_title.render(subtitle_text, True, DARK_GRAY)
    
    # Calculate legend positions
    legend_x = GRID_WIDTH + 10
    y_offset = 20
    
    # Draw title and subtitle on separate lines
    win.blit(title, (legend_x, y_offset))
    y_offset += 25
    win.blit(subtitle, (legend_x, y_offset))
    y_offset += 35
    
    # Draw controls subtitle
    controls_subtitle = font_normal.render("CONTROLS", True, DARK_GRAY)
    win.blit(controls_subtitle, (legend_x, y_offset))
    y_offset += 30
    
    
    # Draw algorithm selection keys
    algorithm_labels = [
        "0 - Bidirectional A*",
        "1 - Dijkstra",
        "2 - A*",
        "3 - DFS",
        "4 - Greedy Best-First",
        "5 - Theta*",
        "6 - Bidirectional",
        "7 - Flow Field",
        "8 - Jump Point Search (JPS)",
        "9 - HPA*"
    ]
    
    # Highlight current algorithm if one is running
    for i, label in enumerate(algorithm_labels):
        color = BLUE
        if current_algorithm is not None and f"{current_algorithm}" in label:
            color = DARK_GRAY
            # Draw highlight background
            pygame.draw.rect(win, HIGHLIGHT_COLOR, 
                            (legend_x - 5, y_offset - 2, LEGEND_WIDTH - 10, 24))
        
        text = font_normal.render(label, True, color)
        win.blit(text, (legend_x, y_offset))
        y_offset += 24
    
    y_offset += 20
    
    # Draw other controls
    controls = [
        "ESC/C - Stop Algorithm",
        "SPACE - Pause/Resume",
        "Left Click - Place Start/End/Wall",
        "Right Click - Erase"
    ]
    
    win.blit(font_normal.render("OTHER CONTROLS:", True, DARK_GRAY), (legend_x, y_offset))
    y_offset += 30
    
    for control in controls:
        text = font_small.render(control, True, DARK_GRAY)
        win.blit(text, (legend_x, y_offset))
        y_offset += 22
    
    # Draw color legend
    y_offset += 20
    win.blit(font_normal.render("COLOR LEGEND:", True, DARK_GRAY), (legend_x, y_offset))
    y_offset += 30
    
    colors = [
        (GREEN, "Start Node"),
        (RED, "End Node"),
        (BLACK, "Wall"),
        (SKY, "Open Set"),
        (GREY, "Closed Set"),
        (PINK, "Path")
    ]
    
    for color, label in colors:
        pygame.draw.rect(win, color, (legend_x, y_offset, 20, 20))
        pygame.draw.rect(win, DARK_GRAY, (legend_x, y_offset, 20, 20), 1)
        text = font_small.render(label, True, DARK_GRAY)
        win.blit(text, (legend_x + 30, y_offset + 2))
        y_offset += 25
    
    # Draw algorithm status
    y_offset += 20
    if running_algorithm:
        status_text = "Status: Running"
        if algorithm_paused:
            status_text = "Status: Paused"
    else:
        if message:
            status_text = f"Status: {message}"
        else:
            status_text = "Status: Ready"
    
    win.blit(font_normal.render(status_text, True, DARK_GRAY), (legend_x, y_offset))

def get_clicked_pos(pos):
    x, y = pos
    if x >= GRID_WIDTH:  # Clicked in the legend area
        return None, None
    row = min(y // CELL_HEIGHT, ROWS - 1)
    col = min(x // CELL_WIDTH, COLS - 1)
    return row, col

def place_random_walls(grid):
    blocked = 0
    while blocked < BLOCKED_COUNT:
        r = random.randint(0, ROWS - 1)
        c = random.randint(0, COLS - 1)
        node = grid[r][c]
        if not node.is_blocked():
            node.make_barrier()
            blocked += 1

def update_all_neighbors(grid):
    for row in grid:
        for node in row:
            node.update_neighbors(grid)

def main():
    global running_algorithm, algorithm_paused, current_algorithm
    
    grid = make_grid()
    place_random_walls(grid)
    start = None
    end = None
    run = True
    status_message = "Welcome! Place start and end nodes"

    while run:
        draw(WIN, grid, status_message)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            # If algorithm is running, only process quit and pause events
            if running_algorithm:
                continue
                
            # Handle mouse clicks for placing start/end nodes and walls
            if pygame.mouse.get_pressed()[0]:  # Left mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                if row is None or col is None:  # Clicked in legend area
                    continue
                    
                node = grid[row][col]
                
                if not start and node != end:
                    start = node
                    start.make_start()
                    status_message = "Place end node"
                elif not end and node != start:
                    end = node
                    end.make_end()
                    status_message = "Draw walls or select algorithm (0-9)"
                    update_all_neighbors(grid)
                elif node != start and node != end:
                    node.make_barrier()
                    
            elif pygame.mouse.get_pressed()[2]:  # Right mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos)
                if row is None or col is None:  # Clicked in legend area
                    continue
                    
                node = grid[row][col]
                node.reset()
                
                if node == start:
                    start = None
                    status_message = "Place start node"
                elif node == end:
                    end = None
                    status_message = "Place end node" if start else "Place start node"
            
            # Handle keyboard for algorithm selection
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset grid
                    grid = make_grid()
                    place_random_walls(grid)
                    start = None
                    end = None
                    status_message = "Welcome! Place start and end nodes"
                
                # Algorithm selection keys
                if start and end:
                    update_all_neighbors(grid)
                    if event.key == pygame.K_0:
                        current_algorithm = 0
                        status_message = "Running Bidirectional A*..."
                        running_algorithm = True
                        bidirectional_a_star(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_1:
                        current_algorithm = 1
                        status_message = "Running Dijkstra's Algorithm..."
                        running_algorithm = True
                        dijkstra(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_2:
                        current_algorithm = 2
                        status_message = "Running A*..."
                        running_algorithm = True
                        astar(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_3:
                        current_algorithm = 3
                        status_message = "Running DFS..."
                        running_algorithm = True
                        dfs(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_4:
                        current_algorithm = 4
                        status_message = "Running Greedy Best-First Search..."
                        running_algorithm = True
                        greedy(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_5:
                        current_algorithm = 5
                        status_message = "Running Theta*..."
                        running_algorithm = True
                        theta_star(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_6:
                        current_algorithm = 6
                        status_message = "Running Bidirectional Search..."
                        running_algorithm = True
                        bidirectional_search(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_7:
                        current_algorithm = 7
                        status_message = "Running Flow Field..."
                        running_algorithm = True
                        flow_field(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_8:
                        current_algorithm = 8
                        status_message = "Running JPS Optimization..."
                        running_algorithm = True
                        jps(lambda: draw(WIN, grid, status_message), grid, start, end)
                    elif event.key == pygame.K_9:
                        current_algorithm = 9
                        status_message = "Running HPA*..."
                        running_algorithm = True
                        hpa_star(lambda: draw(WIN, grid, status_message), grid, start, end)
                else:
                    status_message = "Please place both start and end nodes first!"
                
                # Reset or clear algorithm results with 'c' key
                if event.key == pygame.K_c and not running_algorithm:
                    if start and end:
                        reset_grid(grid, start, end)
                        status_message = "Grid cleared. Select algorithm (0-9)"
                        
    pygame.quit()

if __name__ == "__main__":
    main()
