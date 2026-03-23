from agent import NaiveAgent
from environment import MazeEnvironment
from maze_loader import MazeLoader
from hazard_loader import load_hazards

import numpy as np
import matplotlib.pyplot as plt
from collections import deque


# ===============================
# VISUALIZATION (PIXEL-BASED)
# ===============================
def show_maze_with_hazards(grid, path=None, hazards=None):
    # base image
    img = np.stack([grid * 255]*3, axis=-1).astype(np.uint8)

    # draw path (red)
    if path:
        for (x, y) in path:
            img[x, y] = [255, 0, 0]

    # draw hazards
    if hazards:
        for (x, y), h_type in hazards.items():

            if h_type == "pit":
                img[x, y] = [0, 0, 255]      # 🔵 blue

            elif h_type == "teleport":
                img[x, y] = [255, 255, 0]    # 🟡 yellow

            elif h_type == "confusion":
                img[x, y] = [255, 0, 255]    # 🟣 purple

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.title("Maze + Hazards + Path")
    plt.axis('off')
    plt.show()


# ===============================
# 1. LOAD MAZE
# ===============================
def test_png_loading():
    print("=== Testing PNG Loader ===")

    loader = MazeLoader("mazes/maze1.png")
    grid = loader.load()

    print("Grid shape:", grid.shape)
    loader.print_sample(grid)

    return grid


# ===============================
# 2. SOLVE MAZE USING BFS
# ===============================
def solve_maze(grid):
    print("\n=== Solving Maze with BFS (Pixel-based) ===")

    h, w = grid.shape

    # find start (top row open)
    start = None
    for i in range(w):
        if grid[0][i] == 0:
            start = (0, i)
            break

    # find goal (bottom row open)
    goal = None
    for i in range(w):
        if grid[h-1][i] == 0:
            goal = (h-1, i)
            break

    print("Start:", start)
    print("Goal:", goal)

    if start is None or goal is None:
        print("❌ Could not find start or goal")
        return None

    # BFS
    q = deque([start])
    visited = set([start])
    parent = {}

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    while q:
        cur = q.popleft()

        if cur == goal:
            break

        for dx, dy in directions:
            nx, ny = cur[0] + dx, cur[1] + dy

            if 0 <= nx < h and 0 <= ny < w:
                if grid[nx][ny] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parent[(nx, ny)] = cur
                    q.append((nx, ny))

    # reconstruct path
    path = []
    cur = goal

    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            print("❌ Path reconstruction failed")
            return None

    path.append(start)
    path.reverse()

    print(f"✅ Path found! Length = {len(path)}")

    hazards = load_hazards("mazes/maze2.png")
    show_maze_with_hazards(grid, path, hazards)

    return path


# ===============================
# 3. LOAD + DEMONSTRATE HAZARDS
# ===============================
def demo_hazards():
    print("\n=== Hazard Demonstration ===")

    hazards = load_hazards("mazes/maze2.png")

    print(f"Total hazards detected: {len(hazards)}")

    sample = list(hazards.items())[:5]

    for pos, h_type in sample:
        print(f"\nAgent steps on {pos}")

        if h_type == "pit":
            print("💀 Agent dies and respawns at start")

        elif h_type == "teleport":
            print("🌀 Agent teleports to another location")

        elif h_type == "confusion":
            print("🤯 Controls inverted")

    return hazards


# ===============================
# 4. RUN ENVIRONMENT
# ===============================
def run_episode():
    print("\n=== Running Environment (Naive Agent) ===")

    env = MazeEnvironment('training')
    agent = NaiveAgent()

    env.reset()
    agent.reset_episode()

    last_result = None
    turn_count = 0

    while turn_count < 200:
        actions = agent.plan_turn(last_result)
        result = env.step(actions)

        print(f"Turn {turn_count}: "
              f"Pos={result.current_position}, "
              f"Walls={result.wall_hits}, "
              f"Dead={result.is_dead}")

        if result.is_goal_reached:
            print(f"\n✅ Goal reached in {turn_count} turns!")
            break

        last_result = result
        turn_count += 1

    stats = env.get_episode_stats()
    print("\nFinal Stats:", stats)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    grid = test_png_loading()
    solve_maze(grid)
    demo_hazards()
    run_episode()