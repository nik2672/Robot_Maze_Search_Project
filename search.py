import sys
from collections import deque
import heapq
import time
import random
import matplotlib.pyplot as plt  # Importing matplotlib for visualization

class State:
    def __init__(self, posX, posY, movements=None, cost=0):
        self.posX = posX
        self.posY = posY
        self.movements = movements or []
        self.cost = cost

    def __hash__(self):
        return hash((self.posX, self.posY))

    def __eq__(self, other):
        return (self.posX, self.posY) == (other.posX, other.posY)

    def __lt__(self, other):
        return self.cost < other.cost

    def expand(self, maze):
        """Generates all possible movements from the current state."""
        outcomes = []
        moves = {
            (0, 1): 'RIGHT',
            (0, -1): 'LEFT',
            (1, 0): 'DOWN',
            (-1, 0): 'UP'
        }
        for dx, dy in moves:
            newX, newY = self.posX + dx, self.posY + dy
            if 0 <= newX < len(maze) and 0 <= newY < len(maze[0]) and maze[newX][newY] == 0:
                move = moves[(dx, dy)]
                new_state = State(newX, newY, self.movements + [move], self.cost + 1)
                outcomes.append(new_state)
        return outcomes

    def heuristic(self, target):
        """Calculates Manhattan distance to the target."""
        return abs(self.posX - target.posX) + abs(self.posY - target.posY)

def load_maze(filepath):
    """Loads the maze configuration from a file."""
    with open(filepath, 'r') as file:
        content = file.readlines()
    
    dimensions = content[0].strip().strip('[]').split(',')
    rows, cols = int(dimensions[0]), int(dimensions[1])
    maze = [[0] * cols for _ in range(rows)]

    start_coords = content[1].strip().strip('()').split(',')
    start_x, start_y = int(start_coords[0]), int(start_coords[1])
    start = State(start_x, start_y)

    targets = []
    target_coords = content[2].strip().split('|')
    for target in target_coords:
        x, y = map(int, target.strip().strip('()').replace(' ', '').split(','))
        targets.append(State(x, y))

    for line in content[3:]:
        block = line.strip().strip('()').replace(' ', '').split(',')
        block_x, block_y, block_w, block_h = map(int, block)
        for i in range(block_x, block_x + block_w):
            for j in range(block_y, block_y + block_h):
                if 0 <= i < rows and 0 <= j < cols:
                    maze[i][j] = 1  # Mark obstacles

    return maze, start, targets

def dfs(maze, start, targets):
    """Depth-First Search algorithm."""
    stack, visited, expansions = [(start, [start])], set(), 0
    while stack:
        current, path = stack.pop()
        expansions += 1
        if (current.posX, current.posY) in [(t.posX, t.posY) for t in targets]:
            return True, (current.posX, current.posY), current.movements, expansions
        if current not in visited:
            visited.add(current)
            for successor in current.expand(maze):
                stack.append((successor, path + [successor]))
    return False, None, [], expansions

def bfs(maze, start, targets):
    """Breadth-First Search algorithm."""
    queue, visited, expansions = deque([(start, [start])]), set(), 0
    while queue:
        current, path = queue.popleft()
        expansions += 1
        if (current.posX, current.posY) in [(t.posX, t.posY) for t in targets]:
            return True, (current.posX, current.posY), current.movements, expansions
        if current not in visited:
            visited.add(current)
            for successor in current.expand(maze):
                queue.append((successor, path + [successor]))
    return False, None, [], expansions

def gbfs(maze, start, targets):
    """Greedy Best-First Search algorithm."""
    priority_queue, visited, expansions = [(0, start, [start])], set(), 0
    while priority_queue:
        _, current, path = heapq.heappop(priority_queue)
        expansions += 1
        if (current.posX, current.posY) in [(t.posX, t.posY) for t in targets]:
            return True, (current.posX, current.posY), current.movements, expansions
        if current not in visited:
            visited.add(current)
            for successor in current.expand(maze):
                if successor not in visited:
                    heuristic = successor.heuristic(targets[0])
                    heapq.heappush(priority_queue, (heuristic, successor, path + [successor]))
    return False, None, [], expansions

def a_star(maze, start, targets):
    """A* Search algorithm."""
    priority_queue, visited, expansions = [(0, 0, start, [start])], set(), 0
    while priority_queue:
        f_cost, _, current, path = heapq.heappop(priority_queue)
        expansions += 1
        if (current.posX, current.posY) in [(t.posX, t.posY) for t in targets]:
            return True, (current.posX, current.posY), current.movements, expansions
        if current not in visited:
            visited.add(current)
            for successor in current.expand(maze):
                if successor not in visited:
                    g_cost = current.cost + 1
                    h_cost = successor.heuristic(targets[0])
                    f_cost = g_cost + h_cost
                    heapq.heappush(priority_queue, (f_cost, h_cost, successor, path + [successor]))
    return False, None, [], expansions

def random_dfs(maze, start, targets):
    """Randomized Depth-First Search algorithm."""
    stack, visited, expansions = [(start, [start])], set(), 0
    while stack:
        current, path = stack.pop()
        expansions += 1
        if (current.posX, current.posY) in [(t.posX, t.posY) for t in targets]:
            return True, (current.posX, current.posY), current.movements, expansions
        if current not in visited:
            visited.add(current)
            successors = current.expand(maze)
            random.shuffle(successors)
            for successor in successors:
                if successor not in visited:
                    stack.append((successor, path + [successor]))
    return False, None, [], expansions

def visualize_maze(maze, path, start, targets):
    """Function to visualize the maze and the path taken."""
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Draw the grid and obstacles
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 1:  # Obstacle
                ax.add_patch(plt.Rectangle((j, len(maze) - i - 1), 1, 1, color='black'))
            else:
                ax.add_patch(plt.Rectangle((j, len(maze) - i - 1), 1, 1, edgecolor='gray', facecolor='white'))

    # Draw the start position
    ax.add_patch(plt.Circle((start.posY + 0.5, len(maze) - start.posX - 0.5), 0.4, color='blue', label='Start'))

    # Draw the goal positions more prominently
    for idx, target in enumerate(targets):
        ax.add_patch(plt.Rectangle((target.posY, len(maze) - target.posX - 1), 1, 1, color='green', edgecolor='black', label='Goal' if idx == 0 else ""))

    # Draw the path taken by the search algorithm
    current_pos = (start.posX, start.posY)
    for move in path:
        if move == 'UP':
            current_pos = (current_pos[0] - 1, current_pos[1])
        elif move == 'DOWN':
            current_pos = (current_pos[0] + 1, current_pos[1])
        elif move == 'LEFT':
            current_pos = (current_pos[0], current_pos[1] - 1)
        elif move == 'RIGHT':
            current_pos = (current_pos[0], current_pos[1] + 1)
        ax.add_patch(plt.Circle((current_pos[1] + 0.5, len(maze) - current_pos[0] - 0.5), 0.3, color='red'))

    # Add legend to the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')

    # Enhance the plot's appearance
    plt.xlim(0, len(maze[0]))
    plt.ylim(0, len(maze))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(visible=True, color='gray', linestyle='--', linewidth=0.5)
    plt.axis('off')

    # Add title and additional information
    plt.title("Maze Visualization with Path Taken", fontsize=14)
    plt.text(0, len(maze) + 1, f"Search Method: {sys.argv[2].upper()}", fontsize=12)
    plt.text(0, len(maze), f"Steps Taken: {len(path)}", fontsize=12)

    plt.show()


def main():
    if len(sys.argv) != 4:
        print("how to use: python search.py <filename> <search_method> <num_trials>")
        sys.exit(1)

    maze_file = sys.argv[1]
    search_method = sys.argv[2].lower()
    num_trials = int(sys.argv[3])

    maze, start, targets = load_maze(maze_file)
    total_time, total_expansions = 0, 0

    for i in range(num_trials):
        start_time = time.perf_counter()
        goal_found, goal_pos, path_taken, expansions = None, None, None, None

        # Select the search method based on user input
        if search_method == 'dfs':
            goal_found, goal_pos, path_taken, expansions = dfs(maze, start, targets)
        elif search_method == 'bfs':
            goal_found, goal_pos, path_taken, expansions = bfs(maze, start, targets)
        elif search_method == 'gbfs':
            goal_found, goal_pos, path_taken, expansions = gbfs(maze, start, targets)
        elif search_method == 'astar':
            goal_found, goal_pos, path_taken, expansions = a_star(maze, start, targets)
        elif search_method == 'random_dfs':
            goal_found, goal_pos, path_taken, expansions = random_dfs(maze, start, targets)
        else:
            print(f"Search method '{search_method}' is not recognized.")
            sys.exit(1)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        total_time += run_time
        total_expansions += expansions

        print(f"Trial {i + 1}: {maze_file} using {search_method.upper()}")
        if goal_found:
            print(f"Goal reached at Node ({goal_pos[0]}, {goal_pos[1]}) with {expansions} expansions")
            print("Path to goal:", path_taken)
            # Visualize the maze and the path
            visualize_maze(maze, path_taken, start, targets)
        else:
            print("Goal not reached")
        print(f"Run time for trial {i + 1}: {run_time:.15f} seconds\n")

    # Calculate and print average time and expansions if multiple trials are conducted
    if num_trials > 1:
        average_time = total_time / num_trials
        average_expansions = total_expansions / num_trials
        print(f"Average time after {num_trials} trials: {average_time:.15f} seconds")
        print(f"Average expansions after {num_trials} trials: {average_expansions:.0f}")

if __name__ == "__main__":
    main()
