from agent import NaiveAgent
from environment import MazeEnvironment
from maze_loader import MazeLoader

from planner import astar
from visualizer import show_maze
from hazard_loader import load_hazards


# ===============================
# 1. LOAD MAZE (NO HAZARDS)
# ===============================
def test_png_loading():
    print("=== Testing PNG Loader ===")

    loader = MazeLoader("mazes/maze1.png")
    grid = loader.load()

    print("Grid shape:", grid.shape)
    loader.print_sample(grid)

    return grid


# ===============================
# 2. SOLVE MAZE USING A*
# ===============================
def solve_maze(grid):
    print("\n=== Solving Maze with A* ===")

    # You may need to adjust start/goal depending on image
    start = (0, 0)
    goal = (len(grid) - 1, len(grid[0]) - 1)

    path = astar(grid, start, goal)

    if path:
        print(f"✅ Path found! Length = {len(path)}")
        show_maze(grid, path)
    else:
        print("❌ No path found")

    return path


# ===============================
# 3. LOAD + DEMONSTRATE HAZARDS
# ===============================
def demo_hazards():
    print("\n=== Hazard Demonstration ===")

    hazards = load_hazards("mazes/maze2.png")

    print(f"Total hazards detected: {len(hazards)}")

    # Show some examples
    sample = list(hazards.items())[:10]
    for pos, h_type in sample:
        print(f"At {pos}: {h_type}")

    # Demonstrate behavior
    print("\n--- Simulating hazard effects ---")

    for pos, h_type in sample[:3]:
        print(f"\nAgent steps on {pos}")

        if h_type == "pit":
            print("💀 Agent dies and respawns at start")

        elif h_type == "teleport":
            print("🌀 Agent teleports to another location")

        elif h_type == "confusion":
            print("🤯 Controls inverted (UP↔DOWN, LEFT↔RIGHT)")

    return hazards


# ===============================
# 4. RUN ENVIRONMENT (NAIVE AGENT)
# ===============================
def run_episode():
    print("\n=== Running Environment (Naive Agent) ===")

    env = MazeEnvironment('training')
    agent = NaiveAgent()

    env.reset()
    agent.reset_episode()

    last_result = None
    turn_count = 0

    while turn_count < 200:  # keep shorter for demo
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
    # Step 1: Load maze
    grid = test_png_loading()

    # Step 2: Solve maze (A*)
    solve_maze(grid)

    # Step 3: Demonstrate hazards
    demo_hazards()

    # Step 4: Run naive agent in environment
    run_episode()
