from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from collections import deque, defaultdict
from PIL import Image


Position = Tuple[int, int]  # (x, y)
GRID_SIZE = 64


class Action(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    WAIT = 4


@dataclass
class TurnResult:
    wall_hits: int = 0
    current_position: Position = (0, 0)
    is_dead: bool = False
    is_confused: bool = False
    is_goal_reached: bool = False
    teleported: bool = False
    actions_executed: int = 0
    pushed: bool = False  # set when an arrow tile displaced the agent this turn


@dataclass
class FireGroup:
    pivot: Position
    initial_cells: List[Position]
    offsets: List[Position] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        px, py = self.pivot
        self.offsets = [(x - px, y - py) for x, y in self.initial_cells]

    def rotate_clockwise(self) -> None:
        # Screen coordinates: x increases right, y increases down.
        # 90 degrees clockwise: (dx, dy) -> (-dy, dx)
        self.offsets = [(-dy, dx) for dx, dy in self.offsets]

    def cells(self) -> Set[Position]:
        px, py = self.pivot
        result = set()

        for dx, dy in self.offsets:
            x, y = px + dx, py + dy

            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                result.add((x, y))

        return result


class MazeEnvironment:
    """
    Environment for any 64x64 maze image.

    Detects walls, start/goal, and hazards from the source image. Hazards:

        * Fire groups - rotate 90 degrees clockwise every N actions around a
          pivot. Lethal on contact.
        * Confusion cells - invert controls for the rest of this turn plus the
          next full turn.
        * Teleport pairs - one-way, deterministic. Colors: green, purple,
          yellow, red. Source is the higher-solidity (filled) cell; destination
          is the lower-solidity (sparkle) cell.
        * Arrow tiles - impassable. Attempting to step onto one displaces the
          agent one cell past the arrow in its direction. Chains resolve with a
          seen-set guard; a push that would hit a wall, leave the grid, or
          bounce the agent onto its own cell is treated as a wall hit.
    """

    DELTAS: Dict[Action, Position] = {
        Action.MOVE_UP: (0, -1),
        Action.MOVE_DOWN: (0, 1),
        Action.MOVE_LEFT: (-1, 0),
        Action.MOVE_RIGHT: (1, 0),
        Action.WAIT: (0, 0),
    }

    INVERTED: Dict[Action, Action] = {
        Action.MOVE_UP: Action.MOVE_DOWN,
        Action.MOVE_DOWN: Action.MOVE_UP,
        Action.MOVE_LEFT: Action.MOVE_RIGHT,
        Action.MOVE_RIGHT: Action.MOVE_LEFT,
        Action.WAIT: Action.WAIT,
    }

    def __init__(
        self,
        maze_path: str | Path = "data/maze_alpha.png",
        fire_rotation_interval: int = 5,
        death_on_fire_rotation: bool = True,
        arrow_tiles_override: Optional[Dict[Position, Action]] = None,
    ) -> None:
        self.maze_path = Path(maze_path)
        self.image = Image.open(self.maze_path).convert("RGB")
        self.width, self.height = self.image.size

        self.cell_size = (min(self.width, self.height) - 2) // GRID_SIZE
        self.offset_x = (self.width - self.cell_size * GRID_SIZE) // 2
        self.offset_y = (self.height - self.cell_size * GRID_SIZE) // 2

        self.fire_rotation_interval = fire_rotation_interval
        self.death_on_fire_rotation = death_on_fire_rotation

        self.blocked_edges: Set[Tuple[Position, Position]] = set()
        self.fire_groups: List[FireGroup] = []
        self.fire_cells: Set[Position] = set()
        self.confusion_cells: Set[Position] = set()
        self.teleport_map: Dict[Position, Position] = {}
        self.arrow_tiles: Dict[Position, Action] = {}

        self._parse_wall_edges()

        self.start = self._find_top_start()
        self.goal = self._find_bottom_goal()

        self._detect_hazards_from_image()

        # Explicit overrides win over image-based detection (useful for testing
        # arrow behaviour before the art is finalized).
        if arrow_tiles_override is not None:
            self.arrow_tiles = dict(arrow_tiles_override)

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Position:
        for group in self.fire_groups:
            group.reset()

        self._refresh_fire_cells()

        self.current_position: Position = self.start
        self.pending_respawn = False
        self.confused_next_turn = False

        self.turns_taken = 0
        self.deaths = 0
        self.confused_count = 0
        self.push_count = 0
        self.goal_reached = False
        self.total_actions_executed = 0
        self.path_length = 0

        self.visited_cells: Set[Position] = {self.start}
        self.position_history: List[Position] = [self.start]
        self.event_history: List[dict] = []

        return self.start

    def step(self, actions: Sequence[Action]) -> TurnResult:
        if not 1 <= len(actions) <= 5:
            raise ValueError("actions must contain 1 to 5 actions")

        if self.pending_respawn:
            self.current_position = self.start
            self.pending_respawn = False
            self.position_history.append(self.current_position)

        result = TurnResult(current_position=self.current_position)

        confused_active = self.confused_next_turn
        confuse_next_turn = False

        for requested_action in actions:
            action = self.INVERTED[requested_action] if confused_active else requested_action

            triggered_confusion = self._execute_single_action(action, result)

            if triggered_confusion:
                confused_active = True
                confuse_next_turn = True

            self.total_actions_executed += 1
            result.actions_executed += 1

            self._rotate_fire_if_needed(result)

            result.current_position = self.current_position

            self.event_history.append(
                {
                    "turn": self.turns_taken,
                    "requested_action": requested_action.name,
                    "executed_action": action.name,
                    "position": self.current_position,
                    "dead": result.is_dead,
                    "confused": result.is_confused,
                    "teleported": result.teleported,
                    "pushed": result.pushed,
                    "goal": result.is_goal_reached,
                    "fire_phase": self.fire_phase,
                }
            )

            if result.is_dead or result.is_goal_reached:
                break

        self.confused_next_turn = confuse_next_turn
        self.turns_taken += 1

        return result

    def get_episode_stats(self) -> dict:
        total_cells_visited = max(1, len(self.position_history))
        unique_cells = len(self.visited_cells)
        return {
            "turns_taken": self.turns_taken,
            "deaths": self.deaths,
            "confused": self.confused_count,
            "pushes": self.push_count,
            "cells_explored": unique_cells,
            "goal_reached": self.goal_reached,
            "path_length": self.path_length,
            "total_actions_executed": self.total_actions_executed,
            "total_cells_visited": total_cells_visited,
            "exploration_efficiency": unique_cells / total_cells_visited,
            "map_completeness": unique_cells / float(GRID_SIZE * GRID_SIZE),
        }

    @staticmethod
    def calculate_metrics(episode_stats: Sequence[dict]) -> dict:
        total_episodes = len(episode_stats)

        if total_episodes == 0:
            raise ValueError("episode_stats cannot be empty")

        successful = [s for s in episode_stats if s.get("goal_reached", False)]

        total_turns = sum(s.get("turns_taken", 0) for s in episode_stats)
        total_deaths = sum(s.get("deaths", 0) for s in episode_stats)
        total_cells_visited = sum(s.get("total_cells_visited", max(1, s.get("path_length", 0))) for s in episode_stats)
        total_unique_cells = sum(s.get("cells_explored", 0) for s in episode_stats)

        success_rate = (len(successful) / total_episodes) * 100.0

        avg_path_length = (
            sum(s.get("path_length", 0) for s in successful) / len(successful)
            if successful
            else float("inf")
        )

        avg_turns = (
            sum(s.get("turns_taken", 0) for s in successful) / len(successful)
            if successful
            else float("inf")
        )

        death_rate = total_deaths / total_turns if total_turns > 0 else 0.0
        exploration_efficiency = (
            total_unique_cells / total_cells_visited if total_cells_visited > 0 else 0.0
        )
        map_completeness = (
            sum(s.get("map_completeness", 0.0) for s in episode_stats) / total_episodes
        )

        return {
            "success_rate": success_rate,
            "average_path_length": avg_path_length,
            "average_turns_to_solution": avg_turns,
            "death_rate": death_rate,
            "exploration_efficiency": exploration_efficiency,
            "map_completeness": map_completeness,
            "average_confusions": sum(s.get("confused", 0) for s in episode_stats) / total_episodes,
            "average_arrow_pushes": sum(s.get("pushes", 0) for s in episode_stats) / total_episodes,
        }

    def get_hazard_summary(self) -> dict:
        return {
            "start": self.start,
            "goal": self.goal,
            "fire_groups": len(self.fire_groups),
            "fire_cells_now": len(self.fire_cells),
            "confusion_cells": len(self.confusion_cells),
            # teleport_map stores both directions, so two entries = one pair.
            "teleport_pairs": len(self.teleport_map) // 2,
            "teleport_entries": len(self.teleport_map),
            "arrow_tiles": len(self.arrow_tiles),
        }

    @property
    def fire_phase(self) -> int:
        return (self.total_actions_executed // self.fire_rotation_interval) % 4

    # ------------------------------------------------------------------
    # Movement / hazard behavior
    # ------------------------------------------------------------------

    def _execute_single_action(self, action: Action, result: TurnResult) -> bool:
        if action == Action.WAIT:
            return False

        dx, dy = self.DELTAS[action]
        x, y = self.current_position
        target = (x + dx, y + dy)

        if not self._inside_grid(target) or self._is_blocked(self.current_position, target):
            result.wall_hits += 1
            return False

        # Arrow tiles are impassable. Attempting to enter one triggers a
        # directional push; chained arrows resolve with a cycle guard.
        if target in self.arrow_tiles:
            return self._resolve_arrow_push(target, result)

        return self._enter_cell(target, result)

    def _resolve_arrow_push(self, first_arrow: Position, result: TurnResult) -> bool:
        """
        Chain-resolve an arrow push. Returns whether confusion was triggered.

        The push begins at `first_arrow` and follows each successive arrow's
        direction. If the push lands on another arrow, it chains. Termination
        conditions (all treated as a failed move, wall_hits +=1):
            * Push would leave the grid.
            * A wall blocks the edge from the current arrow to its neighbor.
            * The chain revisits an arrow tile (cycle).
            * The push would land the agent back on its own cell.
        """
        seen: Set[Position] = set()
        current_arrow = first_arrow

        while True:
            if current_arrow in seen:
                result.wall_hits += 1
                return False
            seen.add(current_arrow)

            push_action = self.arrow_tiles[current_arrow]
            dx, dy = self.DELTAS[push_action]
            ax, ay = current_arrow
            dest = (ax + dx, ay + dy)

            if not self._inside_grid(dest) or self._is_blocked(current_arrow, dest):
                result.wall_hits += 1
                return False

            if dest == self.current_position:
                result.wall_hits += 1
                return False

            if dest in self.arrow_tiles:
                current_arrow = dest
                continue

            result.pushed = True
            self.push_count += 1
            return self._enter_cell(dest, result)

    def _enter_cell(self, pos: Position, result: TurnResult) -> bool:
        """
        Move the agent onto `pos` and resolve cell-entry side effects.
        Returns True if stepping on the cell triggered confusion.
        """
        self.current_position = pos
        self.visited_cells.add(pos)
        self.position_history.append(pos)
        self.path_length += 1

        if pos in self.fire_cells:
            self._mark_dead(result)
            return False

        triggered_confusion = False

        if pos in self.confusion_cells:
            result.is_confused = True
            self.confused_count += 1
            triggered_confusion = True

        self._handle_teleport_chain(result)

        if self.current_position == self.goal:
            result.is_goal_reached = True
            self.goal_reached = True

        return triggered_confusion

    def _handle_teleport_chain(self, result: TurnResult) -> None:
        """
        Resolve teleport behavior.

        Course mazes use paired, two-way teleport pads. Stepping onto either pad
        moves the agent to its paired pad once. The destination is also a
        teleport pad, but it must NOT immediately teleport the agent back in the
        same action. The agent can trigger the reverse direction later only by
        stepping off the pad and stepping back onto it.
        """
        source = self.current_position

        if source not in self.teleport_map:
            return

        destination = self.teleport_map[source]

        if destination == source:
            return

        self.current_position = destination
        self.visited_cells.add(destination)
        self.position_history.append(destination)
        result.teleported = True

        if destination in self.fire_cells:
            self._mark_dead(result)

    def _rotate_fire_if_needed(self, result: TurnResult) -> None:
        if self.total_actions_executed % self.fire_rotation_interval != 0:
            return

        for group in self.fire_groups:
            group.rotate_clockwise()

        self._refresh_fire_cells()

        if (
            self.death_on_fire_rotation
            and not result.is_dead
            and not result.is_goal_reached
            and self.current_position in self.fire_cells
        ):
            self._mark_dead(result)

    def _mark_dead(self, result: TurnResult) -> None:
        result.is_dead = True
        result.current_position = self.current_position
        self.deaths += 1
        self.pending_respawn = True

    # ------------------------------------------------------------------
    # Wall detection from image
    # ------------------------------------------------------------------

    def _parse_wall_edges(self) -> None:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                current = (x, y)

                if x < GRID_SIZE - 1 and self._vertical_wall_exists(x + 1, y):
                    self._add_blocked_edge(current, (x + 1, y))

                if y < GRID_SIZE - 1 and self._horizontal_wall_exists(y + 1, x):
                    self._add_blocked_edge(current, (x, y + 1))

    def _find_top_start(self) -> Position:
        openings = [
            x for x in range(GRID_SIZE)
            if not self._horizontal_wall_exists(0, x)
        ]

        if len(openings) != 1:
            raise ValueError(f"Expected exactly 1 top opening, found {openings}")

        return (openings[0], 0)

    def _find_bottom_goal(self) -> Position:
        openings = [
            x for x in range(GRID_SIZE)
            if not self._horizontal_wall_exists(GRID_SIZE, x)
        ]

        if len(openings) != 1:
            raise ValueError(f"Expected exactly 1 bottom opening, found {openings}")

        return (openings[0], GRID_SIZE - 1)

    def _horizontal_wall_exists(self, y_boundary: int, cell_x: int) -> bool:
        py = self.offset_y + y_boundary * self.cell_size

        x0 = self.offset_x + cell_x * self.cell_size + 3
        x1 = self.offset_x + (cell_x + 1) * self.cell_size - 3

        box = (
            x0,
            max(0, py - 1),
            x1,
            min(self.height, py + 2),
        )

        return self._black_ratio(box) > 0.35

    def _vertical_wall_exists(self, x_boundary: int, cell_y: int) -> bool:
        px = self.offset_x + x_boundary * self.cell_size

        y0 = self.offset_y + cell_y * self.cell_size + 3
        y1 = self.offset_y + (cell_y + 1) * self.cell_size - 3

        box = (
            max(0, px - 1),
            y0,
            min(self.width, px + 2),
            y1,
        )

        return self._black_ratio(box) > 0.35

    def _black_ratio(self, box: Tuple[int, int, int, int]) -> float:
        crop = self.image.crop(box)
        pixels = list(crop.getdata())

        if not pixels:
            return 1.0

        black_pixels = sum(
            1 for r, g, b in pixels
            if r < 80 and g < 80 and b < 80
        )

        return black_pixels / len(pixels)

    def _add_blocked_edge(self, a: Position, b: Position) -> None:
        self.blocked_edges.add((a, b))
        self.blocked_edges.add((b, a))

    def _is_blocked(self, a: Position, b: Position) -> bool:
        return (a, b) in self.blocked_edges

    @staticmethod
    def _inside_grid(pos: Position) -> bool:
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    # ------------------------------------------------------------------
    # Automatic hazard detection
    # ------------------------------------------------------------------

    def _detect_hazards_from_image(self) -> None:
        teleport_candidates = defaultdict(list)
        detected_fire_cells = set()

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                metrics = self._cell_color_metrics((x, y))

                if metrics is None:
                    continue

                label = self._classify_hazard_cell(metrics)

                if label == "fire":
                    detected_fire_cells.add((x, y))

                elif label == "confusion":
                    self.confusion_cells.add((x, y))

                elif label == "arrow":
                    direction = self._arrow_direction_from_cell((x, y))
                    if direction is None:
                        direction = self._arrow_direction_from_metrics(metrics)
                    if direction is not None:
                        self.arrow_tiles[(x, y)] = direction

                elif label.startswith("teleport_"):
                    teleport_candidates[label].append(
                        {
                            "pos": (x, y),
                            "solidity": metrics["solidity"],
                            "colored_count": metrics["colored_count"],
                        }
                    )

        self.fire_groups = self._build_fire_groups(detected_fire_cells)
        self._refresh_fire_cells()

        self.teleport_map = self._build_teleport_map(teleport_candidates)

    def _cell_color_metrics(self, pos: Position) -> dict | None:
        x, y = pos

        margin = max(2, self.cell_size // 8)

        left = self.offset_x + x * self.cell_size + margin
        top = self.offset_y + y * self.cell_size + margin
        right = self.offset_x + (x + 1) * self.cell_size - margin
        bottom = self.offset_y + (y + 1) * self.cell_size - margin

        crop = self.image.crop((left, top, right, bottom))
        pixels = list(crop.getdata())

        colored_pixels = []
        colored_coords = []

        crop_width, crop_height = crop.size

        for index, (r, g, b) in enumerate(pixels):
            mx = max(r, g, b)
            mn = min(r, g, b)
            saturation = 0 if mx == 0 else (mx - mn) / mx

            is_white = r > 240 and g > 240 and b > 240
            is_black = r < 80 and g < 80 and b < 80

            if mx > 50 and saturation > 0.20 and not is_white and not is_black:
                colored_pixels.append((r, g, b))

                px = index % crop_width
                py = index // crop_width
                colored_coords.append((px, py))

        if len(colored_pixels) < 20:
            return None

        avg_r = sum(p[0] for p in colored_pixels) / len(colored_pixels)
        avg_g = sum(p[1] for p in colored_pixels) / len(colored_pixels)
        avg_b = sum(p[2] for p in colored_pixels) / len(colored_pixels)

        red_count = sum(
            1 for r, g, b in colored_pixels
            if r > 150 and g < 140 and b < 140
        )

        pure_red_count = sum(
            1 for r, g, b in colored_pixels
            if r > 180 and g < 95 and b < 95
        )

        orange_count = sum(
            1 for r, g, b in colored_pixels
            if r > 180 and 70 < g < 220 and b < 140
        )

        yellow_count = sum(
            1 for r, g, b in colored_pixels
            if r > 180 and g > 130 and b < 150
        )

        green_count = sum(
            1 for r, g, b in colored_pixels
            if g > 130 and r < 170 and b < 190
        )

        purple_count = sum(
            1 for r, g, b in colored_pixels
            if b > 130 and r > 80 and g < 160
        )

        blue_count = sum(
            1 for r, g, b in colored_pixels
            if b > 140 and r < 130 and g < 170
        )

        xs = [p[0] for p in colored_coords]
        ys = [p[1] for p in colored_coords]

        bbox_area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
        solidity = len(colored_coords) / bbox_area if bbox_area else 0.0

        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        offset_x = avg_x - crop_width / 2
        offset_y = avg_y - crop_height / 2

        return {
            "colored_count": len(colored_pixels),
            "avg_r": avg_r,
            "avg_g": avg_g,
            "avg_b": avg_b,
            "red_count": red_count,
            "pure_red_count": pure_red_count,
            "orange_count": orange_count,
            "yellow_count": yellow_count,
            "green_count": green_count,
            "purple_count": purple_count,
            "blue_count": blue_count,
            "solidity": solidity,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "crop_size": (crop_width, crop_height),
        }

    def _classify_hazard_cell(self, metrics: dict) -> str:
        green = metrics["green_count"]
        purple = metrics["purple_count"]
        yellow = metrics["yellow_count"]
        blue = metrics["blue_count"]
        pure_red = metrics["pure_red_count"]
        count = metrics["colored_count"]

        avg_r = metrics["avg_r"]
        avg_g = metrics["avg_g"]

        # Arrows: blue-dominant shapes. Checked first so they don't fall through
        # to the confusion branch when the blob is large.
        if blue >= 40 and blue > green + purple + yellow:
            return "arrow"

        if green >= 25 and green >= purple and green >= yellow:
            return "teleport_green"

        if purple >= 25 and purple > green:
            return "teleport_purple"

        # Yellow teleport symbols are larger and brighter than normal fire.
        if count >= 100 and yellow >= 70 and avg_r > 220 and avg_g > 140:
            return "teleport_yellow"

        # Red teleport: distinct pure-red mass (not the orange-red of fire).
        # Threshold tuned to catch both the filled circle (high solidity) and
        # the sparkle (lower solidity). May need retuning against a real image.
        if pure_red >= 30 and pure_red > yellow and avg_g < 120:
            return "teleport_red"

        # Confusion symbols are larger orange/yellow face-like symbols.
        # Fire emojis are smaller clusters.
        if count >= 120 and avg_r < 220:
            return "confusion"

        return "fire"


    def _arrow_direction_from_cell(self, pos: Position) -> Optional[Action]:
        """
        Infer direction for blue arrow tiles by looking at the white arrow
        symbol inside the blue square.

        The blue background itself is nearly centered, so we crop away the outer
        2-pixel cell border and analyze only the central white arrow component.
        """
        x, y = pos

        left = self.offset_x + x * self.cell_size
        top = self.offset_y + y * self.cell_size
        right = self.offset_x + (x + 1) * self.cell_size
        bottom = self.offset_y + (y + 1) * self.cell_size

        crop = self.image.crop((left, top, right, bottom)).convert("RGB")
        width, height = crop.size

        # Ignore the outer border because it may contain white path pixels or
        # wall highlights that are not part of the arrow symbol.
        margin = max(2, self.cell_size // 8)
        white_coords = []

        for py in range(margin, max(margin, height - margin)):
            for px in range(margin, max(margin, width - margin)):
                r, g, b = crop.getpixel((px, py))

                # White arrow foreground.
                if r > 200 and g > 200 and b > 200:
                    white_coords.append((px, py))

        if len(white_coords) < 6:
            return None

        min_x = min(px for px, _ in white_coords)
        max_x = max(px for px, _ in white_coords)
        min_y = min(py for _, py in white_coords)
        max_y = max(py for _, py in white_coords)

        symbol_width = max_x - min_x + 1
        symbol_height = max_y - min_y + 1

        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        top_count = sum(1 for _, py in white_coords if py <= mid_y)
        bottom_count = len(white_coords) - top_count
        left_count = sum(1 for px, _ in white_coords if px <= mid_x)
        right_count = len(white_coords) - left_count

        # A vertical arrow is taller than wide. The arrow head side has more
        # white pixels than the tail side.
        if symbol_height > symbol_width:
            return Action.MOVE_UP if top_count >= bottom_count else Action.MOVE_DOWN

        # A horizontal arrow is wider than tall.
        return Action.MOVE_LEFT if left_count >= right_count else Action.MOVE_RIGHT

    def _arrow_direction_from_metrics(self, metrics: dict) -> Optional[Action]:
        """
        Infer the arrow's pointing direction from the centroid of its colored
        pixels. For filled arrow shapes the centroid sits biased toward the
        tip, so the dominant offset axis picks the direction.
        """
        ox = metrics["offset_x"]
        oy = metrics["offset_y"]

        # Centered blobs can't be directionally classified.
        if abs(ox) < 1.0 and abs(oy) < 1.0:
            return None

        if abs(oy) >= abs(ox):
            return Action.MOVE_UP if oy < 0 else Action.MOVE_DOWN
        return Action.MOVE_LEFT if ox < 0 else Action.MOVE_RIGHT

    def _build_teleport_map(self, teleport_candidates: dict) -> Dict[Position, Position]:
        teleport_map: Dict[Position, Position] = {}

        for label, items in teleport_candidates.items():
            if len(items) != 2:
                continue

            # Each color represents one paired teleport. Teleports are TWO-WAY:
            # stepping on either endpoint sends the agent to the other endpoint.
            # Do not choose a one-way source/destination by solidity.
            items = sorted(items, key=lambda item: (item["pos"][1], item["pos"][0]))
            a = items[0]["pos"]
            b = items[1]["pos"]

            if a != b:
                teleport_map[a] = b
                teleport_map[b] = a

        return teleport_map

    def _build_fire_groups(self, fire_cells: Set[Position]) -> List[FireGroup]:
        components = self._connected_components(fire_cells)
        groups = []

        for component in components:
            pivot = self._infer_fire_pivot(component)

            groups.append(
                FireGroup(
                    pivot=pivot,
                    initial_cells=sorted(component),
                )
            )

        return groups

    def _connected_components(self, cells: Set[Position]) -> List[Set[Position]]:
        remaining = set(cells)
        components = []

        while remaining:
            start = remaining.pop()
            component = {start}
            queue = deque([start])

            while queue:
                current = queue.popleft()

                for neighbor in self._neighbors8(current):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        return components

    def _infer_fire_pivot(self, cells: Set[Position]) -> Position:
        """
        Infers the pivot as the most likely V-corner cell.

        For each possible pivot, remove it and check whether it splits the fire
        group into two balanced branches. The best balanced split is used.
        """

        if len(cells) <= 2:
            return sorted(cells)[0]

        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]

        center_x = (min(xs) + max(xs)) / 2
        center_y = (min(ys) + max(ys)) / 2

        best_cell = None
        best_score = None

        for candidate in cells:
            remaining = set(cells)
            remaining.remove(candidate)

            components = self._components_inside_set(remaining)

            if len(components) != 2:
                continue

            size_a = len(components[0])
            size_b = len(components[1])

            imbalance = abs(size_a - size_b)

            cx, cy = candidate
            extreme_distance = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5

            score = (imbalance, -extreme_distance)

            if best_score is None or score < best_score:
                best_score = score
                best_cell = candidate

        if best_cell is not None:
            return best_cell

        # Fallback: choose the most extreme cell from the component center.
        return max(
            cells,
            key=lambda p: ((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2),
        )

    def _components_inside_set(self, cells: Set[Position]) -> List[Set[Position]]:
        remaining = set(cells)
        components = []

        while remaining:
            start = remaining.pop()
            component = {start}
            queue = deque([start])

            while queue:
                current = queue.popleft()

                for neighbor in self._neighbors8(current):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        return components

    @staticmethod
    def _neighbors8(pos: Position) -> List[Position]:
        x, y = pos
        result = []

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    result.append((nx, ny))

        return result

    def _refresh_fire_cells(self) -> None:
        self.fire_cells = set()

        for group in self.fire_groups:
            self.fire_cells.update(group.cells())