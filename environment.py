from typing import List, Tuple
from enum import Enum
import random


# ===============================
# Action Enum
# ===============================

class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


# ===============================
# Turn Result
# ===============================

class TurnResult:
    def __init__(self):
        self.wall_hits: int = 0
        self.current_position: Tuple[int, int] = (0, 0)
        self.is_dead: bool = False
        self.is_confused: bool = False
        self.is_goal_reached: bool = False
        self.teleported: bool = False
        self.actions_executed: int = 0


# ===============================
# Maze Environment
# ===============================

class MazeEnvironment:

    GRID_SIZE = 64

    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    PIT = 4
    TELEPORT = 5
    CONFUSION = 6

    def __init__(self, maze_id: str):
        self.maze_id = maze_id
        self.grid = [[self.EMPTY for _ in range(self.GRID_SIZE)]
                     for _ in range(self.GRID_SIZE)]

        self.start_pos = (0, 0)
        self.goal_pos = (63, 63)

        self.current_pos = None
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_explored = set()

        self.teleport_map = {}

        self._generate_simple_maze()

    # ===============================
    # Simple Maze Generator (Demo)
    # ===============================
    def _generate_simple_maze(self):

        # Set start and goal
        self.grid[0][0] = self.START
        self.grid[63][63] = self.GOAL

        self.start_pos = (0, 0)
        self.goal_pos = (63, 63)

        # Random walls
        for _ in range(400):
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            self.grid[y][x] = self.WALL

        # Random pits
        for _ in range(100):
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            self.grid[y][x] = self.PIT

        # One teleport
        self.grid[10][10] = self.TELEPORT
        self.teleport_map[(10, 10)] = (50, 50)

        # One confusion cell
        self.grid[20][20] = self.CONFUSION

    # ===============================
    # Reset Episode
    # ===============================
    def reset(self) -> Tuple[int, int]:
        self.current_pos = self.start_pos
        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.cells_explored = set()
        return self.current_pos

    # ===============================
    # Step Function
    # ===============================
    def step(self, actions: List[Action]) -> TurnResult:

        if len(actions) == 0 or len(actions) > 5:
            raise ValueError("Must submit 1-5 actions")

        result = TurnResult()
        confused = False

        for action in actions:

            result.actions_executed += 1

            dx, dy = 0, 0

            if action == Action.MOVE_UP:
                dy = -1
            elif action == Action.MOVE_DOWN:
                dy = 1
            elif action == Action.MOVE_LEFT:
                dx = -1
            elif action == Action.MOVE_RIGHT:
                dx = 1
            elif action == Action.WAIT:
                pass

            # Apply confusion inversion
            if confused:
                dx, dy = -dx, -dy

            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy

            # Check bounds
            if not (0 <= new_x < 64 and 0 <= new_y < 64):
                result.wall_hits += 1
                continue

            # Check wall
            if self.grid[new_y][new_x] == self.WALL:
                result.wall_hits += 1
                continue

            # Move agent
            self.current_pos = (new_x, new_y)
            self.cells_explored.add(self.current_pos)

            cell_type = self.grid[new_y][new_x]

            # Goal
            if cell_type == self.GOAL:
                result.is_goal_reached = True
                break

            # Pit
            if cell_type == self.PIT:
                result.is_dead = True
                self.deaths += 1
                break

            # Teleport
            if cell_type == self.TELEPORT:
                result.teleported = True
                self.current_pos = self.teleport_map[self.current_pos]

            # Confusion
            if cell_type == self.CONFUSION:
                confused = True
                result.is_confused = True
                self.confused_count += 1

        result.current_position = self.current_pos
        self.turns_taken += 1

        return result

    # ===============================
    # Stats
    # ===============================
    def get_episode_stats(self) -> dict:
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused_count,
            "cells_explored": len(self.cells_explored),
            "goal_reached": self.current_pos == self.goal_pos
        }
