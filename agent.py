from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple
import random
import heapq

from environment import Action, TurnResult


Position = Tuple[int, int]
State = Tuple[int, int, int, int]
GRID_SIZE = 64


class HazardAwareRLAgent:
    """
    One agent class with two consistent modes.

    HONEST mode:
        The agent does not receive the image map.  It learns only from TurnResult:
        wall hits, deaths, confusion, teleports, pushes, and current position.
        The important upgrade is phase-aware exploration. A fire/death cell is not
        treated as permanently blocked after one death; it is remembered by fire
        phase, then retried safely in other phases.

    HYBRID mode:
        The same planner can optionally load the image-derived environment model
        and run a time-expanded BFS over (position, time mod fire cycle).  This is
        the debug/oracle/model-based mode.
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

    MOVEMENT_ACTIONS: Sequence[Action] = (
        Action.MOVE_DOWN,
        Action.MOVE_RIGHT,
        Action.MOVE_LEFT,
        Action.MOVE_UP,
    )

    def __init__(
        self,
        start_pos: Optional[Position] = None,
        goal_pos: Optional[Position] = None,
        learning_rate: float = 0.35,
        discount_factor: float = 0.92,
        epsilon: float = 0.08,
        epsilon_decay: float = 0.997,
        min_epsilon: float = 0.02,
        fire_rotation_interval: int = 5,
        seed: Optional[int] = 7,
    ) -> None:
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.fire_rotation_interval = fire_rotation_interval
        self.fire_cycle = self.fire_rotation_interval * 4

        self.random = random.Random(seed)
        self.q: Dict[Tuple[State, Action], float] = defaultdict(float)

        # Learned spatial memory.
        self.safe_cells: Set[Position] = set()
        self.open_edges: Set[Tuple[Position, Position]] = set()
        self.wall_edges: Set[Tuple[Position, Position]] = set()

        # Tried unknown edges are phase-aware.  This is the main honest-mode fix.
        self.attempted_edges: Set[Tuple[Position, Position]] = set()
        self.attempted_edges_by_phase: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)

        # Learned dynamic hazards.
        self.death_cells_by_phase: Dict[int, Set[Position]] = defaultdict(set)
        self.death_edges_by_phase: Dict[int, Set[Tuple[Position, Position]]] = defaultdict(set)
        self.all_death_cells: Set[Position] = set()

        self.confusion_cells: Set[Position] = set()
        self.teleports: Dict[Position, Position] = {}

        # Learned non-standard hazard: arrows/pushes.
        self.arrow_cells: Set[Position] = set()
        self.arrow_directions: Dict[Position, Action] = {}
        self.special_transitions: Dict[Tuple[Position, Action], Position] = {}

        # Optional image-derived model for hybrid mode only.
        self.visible_fire_by_phase: Dict[int, Set[Position]] = defaultdict(set)
        self.visible_confusion_cells: Set[Position] = set()
        self.visible_teleports: Dict[Position, Position] = {}
        self.image_blocked_edges: Set[Tuple[Position, Position]] = set()
        self.image_arrow_tiles: Dict[Position, Action] = {}
        self.use_image_guidance = False
        self.image_guided_actions: Deque[Action] = deque()

        # Visit memory.
        self.global_visit_count: Dict[Position, int] = defaultdict(int)
        self.episode_visit_count: Dict[Position, int] = defaultdict(int)

        # Episode state.
        self.current_pos: Optional[Position] = start_pos
        self.confused_next_turn = False
        self.total_actions_executed = 0
        self.episode_path: List[Position] = []
        self.recent_positions: Deque[Position] = deque(maxlen=60)
        self.planned_actions: Deque[Action] = deque()
        self.stuck_counter = 0

        # Previous action tracking.
        self.last_state: Optional[State] = None
        self.last_position: Optional[Position] = None
        self.last_intended_action: Optional[Action] = None
        self.last_requested_action: Optional[Action] = None
        self.last_expected_position: Optional[Position] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear_learning_memory(self) -> None:
        self.q.clear()
        self.safe_cells.clear()
        self.open_edges.clear()
        self.wall_edges.clear()
        self.attempted_edges.clear()
        self.attempted_edges_by_phase.clear()

        self.death_cells_by_phase.clear()
        self.death_edges_by_phase.clear()
        self.all_death_cells.clear()

        self.confusion_cells.clear()
        self.teleports.clear()
        self.arrow_cells.clear()
        self.arrow_directions.clear()
        self.special_transitions.clear()

        self.global_visit_count.clear()
        self.planned_actions.clear()
        self.image_guided_actions.clear()


    def prepare_for_new_maze(self, start_pos: Optional[Position] = None, goal_pos: Optional[Position] = None) -> None:
        """
        Clear maze-specific spatial memory before evaluating a different maze.

        This keeps the learned Q-table/policy parameters but removes walls,
        hazards, teleports, arrows, planned paths, and visit history from the
        previous maze so maze-beta/gamma are not polluted by maze-alpha data.
        """
        if start_pos is not None:
            self.start_pos = start_pos
        if goal_pos is not None:
            self.goal_pos = goal_pos

        self.safe_cells.clear()
        self.open_edges.clear()
        self.wall_edges.clear()
        self.attempted_edges.clear()
        self.attempted_edges_by_phase.clear()

        self.death_cells_by_phase.clear()
        self.death_edges_by_phase.clear()
        self.all_death_cells.clear()

        self.confusion_cells.clear()
        self.teleports.clear()
        self.arrow_cells.clear()
        self.arrow_directions.clear()
        self.special_transitions.clear()

        self.visible_fire_by_phase.clear()
        self.visible_confusion_cells = set()
        self.visible_teleports = {}
        self.image_blocked_edges = set()
        self.image_arrow_tiles = {}
        self.image_guided_actions.clear()
        self.planned_actions.clear()

        self.global_visit_count.clear()
        self.episode_visit_count.clear()
        self.episode_path.clear()
        self.recent_positions.clear()
        self.stuck_counter = 0

        self.current_pos = self.start_pos
        self.confused_next_turn = False
        self.total_actions_executed = 0
        self.last_state = None
        self.last_position = None
        self.last_intended_action = None
        self.last_requested_action = None
        self.last_expected_position = None

        if self.current_pos is not None:
            self.safe_cells.add(self.current_pos)

    def load_detected_hazards_from_environment(
        self,
        env,
        *,
        enable_image_guidance: bool = False,
        reveal_to_agent_memory: bool = False,
    ) -> None:
        """
        Load the optional image-derived model.

        Honest mode should not call this with enable_image_guidance=True.
        Hybrid mode calls it so the same time-expanded planner can solve from
        the parsed maze image.
        """
        self.use_image_guidance = enable_image_guidance
        self.visible_fire_by_phase.clear()
        self.visible_confusion_cells = set()
        self.visible_teleports = {}
        self.image_blocked_edges = set()
        self.image_arrow_tiles = {}
        self.image_guided_actions.clear()

        if not enable_image_guidance and not reveal_to_agent_memory:
            return

        self.visible_confusion_cells = set(getattr(env, "confusion_cells", set()))
        self.visible_teleports = dict(getattr(env, "teleport_map", {}))

        if enable_image_guidance:
            self.image_blocked_edges = set(getattr(env, "blocked_edges", set()))
            self.image_arrow_tiles = dict(getattr(env, "arrow_tiles", {}))
            # The planner needs the teleport map internally.
            self.teleports.update(self.visible_teleports)

        if reveal_to_agent_memory:
            self.confusion_cells.update(self.visible_confusion_cells)
            self.teleports.update(self.visible_teleports)

        for group in getattr(env, "fire_groups", []):
            px, py = group.pivot
            offsets = [(x - px, y - py) for x, y in group.initial_cells]
            for phase in range(4):
                cells = set()
                for dx, dy in offsets:
                    x = px + dx
                    y = py + dy
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        cells.add((x, y))
                self.visible_fire_by_phase[phase].update(cells)
                offsets = [(-dy, dx) for dx, dy in offsets]

    def reset_episode(
        self,
        start_pos: Optional[Position] = None,
        goal_pos: Optional[Position] = None,
        keep_learning_memory: bool = True,
    ) -> None:
        if start_pos is not None:
            self.start_pos = start_pos
        if goal_pos is not None:
            self.goal_pos = goal_pos
        if not keep_learning_memory:
            self.clear_learning_memory()

        self.current_pos = self.start_pos
        self.confused_next_turn = False
        self.total_actions_executed = 0

        self.last_state = None
        self.last_position = None
        self.last_intended_action = None
        self.last_requested_action = None
        self.last_expected_position = None

        self.episode_visit_count = defaultdict(int)
        self.episode_path = []
        self.recent_positions = deque(maxlen=60)
        self.planned_actions = deque()
        self.image_guided_actions = deque()
        self.stuck_counter = 0

        if self.current_pos is not None:
            self.safe_cells.add(self.current_pos)
            self._record_position(self.current_pos)

    def plan_turn(self, last_result: Optional[TurnResult]) -> List[Action]:
        if last_result is not None:
            self._learn_from_result(last_result)

        intended_action = self._choose_intended_action()
        requested_action = self._convert_intended_to_requested(intended_action)

        self.last_state = self._state_key(self.current_pos)
        self.last_position = self.current_pos
        self.last_intended_action = intended_action
        self.last_requested_action = requested_action
        self.last_expected_position = self._expected_position(self.current_pos, intended_action)

        return [requested_action]

    def end_episode(self, goal_reached: bool) -> None:
        if goal_reached:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        else:
            # Keep exploration alive.
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.999)

    def get_known_map(self) -> dict:
        return {
            "start": self.start_pos,
            "goal": self.goal_pos,
            "current_position": self.current_pos,
            "safe_cells": set(self.safe_cells),
            "open_edges": set(self.open_edges),
            "wall_edges": set(self.wall_edges),
            "attempted_edges": set(self.attempted_edges),
            "death_cells_by_phase": {p: set(c) for p, c in self.death_cells_by_phase.items()},
            "all_death_cells": set(self.all_death_cells),
            "confusion_cells": set(self.confusion_cells),
            "teleports": dict(self.teleports),
            "visit_count": dict(self.global_visit_count),
            "episode_path": list(self.episode_path),
            "epsilon": self.epsilon,
        }

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _learn_from_result(self, result: TurnResult) -> None:
        if self.last_position is None or self.last_intended_action is None:
            self.current_pos = result.current_position
            self.safe_cells.add(result.current_position)
            self._record_position(result.current_position)
            return

        old_pos = self.last_position
        expected_pos = self.last_expected_position
        previous_state = self.last_state
        previous_action = self.last_intended_action

        old_safe_count = len(self.safe_cells)

        actions_before = self.total_actions_executed
        phase_before = self._phase_from_tmod(actions_before % self.fire_cycle)

        self.total_actions_executed += result.actions_executed
        actions_after = self.total_actions_executed
        phase_after = self._phase_from_tmod(actions_after % self.fire_cycle)

        if previous_action != Action.WAIT and expected_pos is not None:
            self._mark_attempted_edge(old_pos, expected_pos, phase_before)

        if result.wall_hits > 0 and expected_pos is not None:
            # A real wall hit leaves the agent at old_pos.  If the returned
            # position is different, the agent/environment were out of sync
            # (usually after death/respawn or a special tile), so resync instead
            # of poisoning the map with a false wall.
            if result.current_position == old_pos:
                self._mark_wall(old_pos, expected_pos)
                self.current_pos = old_pos
                self.safe_cells.add(old_pos)
            else:
                self.current_pos = result.current_position
                self.safe_cells.add(self.current_pos)
                self._record_position(self.current_pos)
            self.planned_actions.clear()
            self.image_guided_actions.clear()

        elif result.is_dead:
            death_pos = result.current_position
            self.all_death_cells.add(death_pos)

            # Death can happen when entering the cell, or after the action if fire
            # rotates onto the agent.  Store both when unsure; planning remains
            # phase-specific instead of blocking the cell forever.
            if previous_action == Action.WAIT or death_pos == old_pos:
                self.death_cells_by_phase[phase_after].add(death_pos)
            else:
                self.death_cells_by_phase[phase_before].add(death_pos)
                if self._rotation_happens_after_tmod(actions_before % self.fire_cycle):
                    self.death_cells_by_phase[phase_after].add(death_pos)
                if expected_pos is not None:
                    self.death_edges_by_phase[phase_before].add((old_pos, expected_pos))
                    self.death_edges_by_phase[phase_before].add((expected_pos, old_pos))

            self.current_pos = self.start_pos
            if self.current_pos is not None:
                self.safe_cells.add(self.current_pos)
                self._record_position(self.current_pos)

            self.planned_actions.clear()
            self.image_guided_actions.clear()

        else:
            new_pos = result.current_position
            pushed = bool(getattr(result, "pushed", False))
            unexpected_displacement = (
                previous_action != Action.WAIT
                and expected_pos is not None
                and new_pos != expected_pos
                and not result.teleported
            )

            if previous_action == Action.WAIT:
                self.current_pos = new_pos
                self.safe_cells.add(new_pos)
                self._record_position(new_pos)

            elif unexpected_displacement and new_pos == self.start_pos and old_pos != self.start_pos:
                # Defensive resync: this pattern usually means the environment
                # respawned after a death that the planner was not aligned with.
                # Do not label the expected cell as an arrow or teleport.
                if expected_pos is not None:
                    self.death_cells_by_phase[phase_before].add(expected_pos)
                    self.all_death_cells.add(expected_pos)
                    self.death_edges_by_phase[phase_before].add((old_pos, expected_pos))
                    self.death_edges_by_phase[phase_before].add((expected_pos, old_pos))
                self.current_pos = self.start_pos
                if self.current_pos is not None:
                    self.safe_cells.add(self.current_pos)
                    self._record_position(self.current_pos)
                self.planned_actions.clear()
                self.image_guided_actions.clear()

            elif pushed:
                # Arrow tile: the expected cell is the arrow, not a normal standable cell.
                assert expected_pos is not None
                self.arrow_cells.add(expected_pos)
                inferred_direction = self._infer_arrow_direction(expected_pos, new_pos)
                if inferred_direction is not None:
                    self.arrow_directions[expected_pos] = inferred_direction

                self.special_transitions[(old_pos, previous_action)] = new_pos
                self.safe_cells.add(old_pos)
                self.safe_cells.add(new_pos)
                self.current_pos = new_pos
                self._record_position(new_pos)
                self.planned_actions.clear()
                self.image_guided_actions.clear()

            elif result.teleported and expected_pos is not None and new_pos != expected_pos:
                # Teleport source is expected_pos; destination is new_pos.
                # Teleports are paired/two-way in these mazes, so learn both
                # directions. The environment applies only one teleport per
                # cell entry, so this will not bounce the agent back instantly.
                self.safe_cells.add(expected_pos)
                self.safe_cells.add(new_pos)
                self._mark_open_edge(old_pos, expected_pos)
                self.teleports[expected_pos] = new_pos
                self.teleports[new_pos] = expected_pos
                self.special_transitions[(old_pos, previous_action)] = new_pos
                self.current_pos = new_pos
                self._record_position(new_pos)
                self.planned_actions.clear()
                self.image_guided_actions.clear()

            else:
                if expected_pos is not None:
                    self.safe_cells.add(expected_pos)
                    self._mark_open_edge(old_pos, expected_pos)
                self.safe_cells.add(new_pos)
                self.current_pos = new_pos
                self._record_position(new_pos)

            if result.is_confused:
                confusion_pos = self.current_pos
                if result.teleported and expected_pos is not None:
                    confusion_pos = expected_pos
                elif not pushed and not unexpected_displacement and expected_pos is not None:
                    confusion_pos = expected_pos
                if confusion_pos is not None:
                    self.confusion_cells.add(confusion_pos)
                self.planned_actions.clear()
                self.image_guided_actions.clear()

        self.confused_next_turn = result.is_confused

        new_safe_cell_discovered = len(self.safe_cells) > old_safe_count
        if new_safe_cell_discovered:
            self.stuck_counter = 0
        else:
            self.stuck_counter += 1

        reward = self._calculate_reward(result, new_safe_cell_discovered)
        if previous_state is not None and previous_action is not None:
            self._q_update(previous_state, previous_action, reward, self._state_key(self.current_pos))

    def _calculate_reward(self, result: TurnResult, new_safe_cell_discovered: bool) -> float:
        reward = -1.0
        if result.is_goal_reached:
            reward += 2500.0
        if result.is_dead:
            reward -= 650.0
        if result.wall_hits > 0:
            reward -= 40.0 * result.wall_hits
        if result.is_confused:
            reward -= 5.0
        if result.teleported:
            reward += 8.0
        if getattr(result, "pushed", False):
            reward += 4.0
        if new_safe_cell_discovered:
            reward += 80.0
        if self.current_pos is not None:
            reward -= self.episode_visit_count[self.current_pos] * 2.0
            if self.goal_pos is not None and self.last_position is not None:
                if self._manhattan(self.current_pos, self.goal_pos) < self._manhattan(self.last_position, self.goal_pos):
                    reward += 8.0
                else:
                    reward -= 1.0
        if self._is_looping():
            reward -= 40.0
        return reward

    def _q_update(self, state: State, action: Action, reward: float, next_state: State) -> None:
        old_value = self.q[(state, action)]
        best_next = max(self.q[(next_state, a)] for a in self.MOVEMENT_ACTIONS)
        self.q[(state, action)] = old_value + self.alpha * (reward + self.gamma * best_next - old_value)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _choose_intended_action(self) -> Action:
        if self.current_pos is None:
            return Action.MOVE_DOWN
        if self.goal_pos is not None and self.current_pos == self.goal_pos:
            return Action.WAIT

        if self.use_image_guidance:
            image_action = self._next_image_guided_action()
            if image_action is not None:
                return image_action

        # Existing plan, validated against current phase.  This must happen
        # before expensive replanning, otherwise honest A* runs every turn.
        if self.planned_actions:
            action = self.planned_actions.popleft()
            if self._action_is_still_valid(self.current_pos, action):
                return action
            self.planned_actions.clear()

        # Honest mode upgrade: goal-directed optimistic exploration.
        # Unknown edges are allowed unless already proven wall/deadly.  This
        # keeps honest mode moving toward the exit without using the image map.
        optimistic_action = self._optimistic_greedy_action_to_goal()
        if optimistic_action is not None:
            return optimistic_action

        # If we have actually reached/discovered the goal cell, use the known graph.
        if self.goal_pos is not None and self.goal_pos in self.safe_cells:
            path = self._time_expanded_path_to_known_cell(self.goal_pos)
            if path:
                self.planned_actions = deque(path[1:])
                return path[0]

        # Try an untested local edge in the current fire phase.
        local_untried = self._untried_actions_from_state(
            self.current_pos,
            self.total_actions_executed % self.fire_cycle,
        )
        if local_untried:
            return max(local_untried, key=self._exploration_action_score)

        # Move to a known frontier using a fast learned-graph BFS.
        frontier_path = self._fast_path_to_frontier()
        if frontier_path:
            self.planned_actions = deque(frontier_path[1:])
            return frontier_path[0]

        # If a learned fire/death edge may become testable after waiting, wait.
        if self._wait_is_safe_for_tmod(self.current_pos, self.total_actions_executed % self.fire_cycle):
            if self._future_phase_has_untried_local_edge():
                return Action.WAIT

        # Last fallback: least-visited known transition.
        known_actions = self._known_open_actions_from(self.current_pos)
        if known_actions:
            return min(known_actions, key=self._loop_escape_score)

        # Controlled exploration if no graph move is available.
        candidates = self._candidate_actions_from(self.current_pos)
        if candidates:
            if self.random.random() < self.epsilon:
                return self.random.choice(candidates)
            return max(candidates, key=self._q_action_score)

        if self._wait_is_safe_for_tmod(self.current_pos, self.total_actions_executed % self.fire_cycle):
            return Action.WAIT
        return Action.MOVE_DOWN

    def _action_is_still_valid(self, position: Position, action: Action) -> bool:
        if action == Action.WAIT:
            return self._wait_is_safe_for_tmod(position, self.total_actions_executed % self.fire_cycle)
        # In honest mode, planned paths may intentionally include unknown cells.
        # They remain valid until feedback proves a wall/death.
        return self._transition_after_action(
            position,
            action,
            self.total_actions_executed % self.fire_cycle,
            known_only=self.use_image_guidance,
        ) is not None

    # ------------------------------------------------------------------
    # Hybrid image-guided planner
    # ------------------------------------------------------------------

    def _next_image_guided_action(self) -> Optional[Action]:
        if self.image_guided_actions:
            return self.image_guided_actions.popleft()

        path = self._time_expanded_path_to_goal_image()
        if not path:
            return None

        self.image_guided_actions = deque(path[1:])
        return path[0]

    def _time_expanded_path_to_goal_image(self, max_expansions: int = 200000) -> List[Action]:
        if self.current_pos is None or self.goal_pos is None:
            return []

        start_state = (self.current_pos, self.total_actions_executed % self.fire_cycle)
        queue = deque([start_state])
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Optional[Action]]] = {
            start_state: (None, None)
        }

        expansions = 0
        while queue and expansions < max_expansions:
            pos, tmod = queue.popleft()
            expansions += 1

            if pos == self.goal_pos:
                return self._reconstruct_time_path((pos, tmod), parent)

            for action in (*self.MOVEMENT_ACTIONS, Action.WAIT):
                nxt = self._image_transition(pos, action, tmod)
                if nxt is None:
                    continue
                next_pos, next_tmod = nxt
                state = (next_pos, next_tmod)
                if state in parent:
                    continue
                parent[state] = ((pos, tmod), action)
                queue.append(state)

        return []

    def _image_transition(self, pos: Position, action: Action, tmod: int) -> Optional[Tuple[Position, int]]:
        phase = self._phase_from_tmod(tmod)
        next_tmod = (tmod + 1) % self.fire_cycle
        next_phase = self._phase_from_tmod(next_tmod)

        if pos in self.visible_fire_by_phase[phase]:
            return None

        if action == Action.WAIT:
            arrival = pos
        else:
            target = self._expected_position(pos, action)
            if target is None or not self._inside_grid(target):
                return None
            if not self._image_move_is_valid(pos, target):
                return None
            if target in self.image_arrow_tiles:
                arrival = self._resolve_image_arrow_push(pos, target)
                if arrival is None:
                    return None
            else:
                arrival = self._arrival_after_known_teleport(target)

        if arrival in self.visible_fire_by_phase[phase]:
            return None
        if self._rotation_happens_after_tmod(tmod) and arrival in self.visible_fire_by_phase[next_phase]:
            return None

        return arrival, next_tmod

    def _resolve_image_arrow_push(self, origin: Position, first_arrow: Position) -> Optional[Position]:
        seen: Set[Position] = set()
        current_arrow = first_arrow

        while True:
            if current_arrow in seen:
                return None
            seen.add(current_arrow)

            push_action = self.image_arrow_tiles.get(current_arrow)
            if push_action is None:
                return None

            dest = self._expected_position(current_arrow, push_action)
            if dest is None or not self._inside_grid(dest):
                return None
            if not self._image_move_is_valid(current_arrow, dest):
                return None
            if dest == origin:
                return None
            if dest in self.image_arrow_tiles:
                current_arrow = dest
                continue
            return self._arrival_after_known_teleport(dest)

    def _image_move_is_valid(self, a: Position, b: Position) -> bool:
        return (a, b) not in self.image_blocked_edges and (b, a) not in self.image_blocked_edges

    def _optimistic_static_path_to_goal(self, max_expansions: int = 10000) -> List[Action]:
        """
        Fast honest A* over positions only.

        This is intentionally optimistic: unknown edges are considered possible
        until a TurnResult proves a wall or hazard. It does not use the image map.
        """
        if self.current_pos is None or self.goal_pos is None:
            return []

        start = self.current_pos
        goal = self.goal_pos

        parent: Dict[Position, Tuple[Optional[Position], Optional[Action]]] = {start: (None, None)}
        g_cost: Dict[Position, float] = {start: 0.0}
        heap: List[Tuple[float, float, Position]] = []
        heapq.heappush(heap, (self._manhattan(start, goal), 0.0, start))

        expansions = 0
        while heap and expansions < max_expansions:
            _, current_g, pos = heapq.heappop(heap)
            if current_g != g_cost.get(pos):
                continue

            expansions += 1
            if pos == goal:
                actions: List[Action] = []
                cur = pos
                while parent[cur][0] is not None:
                    prev, action = parent[cur]
                    if action is not None:
                        actions.append(action)
                    cur = prev  # type: ignore[assignment]
                actions.reverse()
                return actions

            actions_order = list(self.MOVEMENT_ACTIONS)
            actions_order.sort(key=lambda a: self._manhattan(self._expected_position(pos, a) or pos, goal))

            for action in actions_order:
                target = self._expected_position(pos, action)
                if target is None or not self._inside_grid(target):
                    continue
                if self._is_known_wall(pos, target):
                    continue

                # If every phase has killed this target/edge, treat it as blocked.
                if all(target in self.death_cells_by_phase[p] for p in range(4)):
                    continue
                if all((pos, target) in self.death_edges_by_phase[p] for p in range(4)):
                    continue

                if (pos, action) in self.special_transitions:
                    arrival = self.special_transitions[(pos, action)]
                elif target in self.arrow_directions:
                    arrival = self._resolve_learned_arrow_push(pos, target)
                    if arrival is None:
                        continue
                else:
                    arrival = self._arrival_after_known_teleport(target)

                if not self._inside_grid(arrival):
                    continue
                if all(arrival in self.death_cells_by_phase[p] for p in range(4)):
                    continue

                step = 1.0
                if arrival not in self.safe_cells:
                    step += 0.7
                if target not in self.safe_cells:
                    step += 0.4
                if any(arrival in self.death_cells_by_phase[p] for p in range(4)):
                    step += 1.5
                step += min(1.5, self.global_visit_count[arrival] * 0.005)

                new_g = current_g + step
                if new_g >= g_cost.get(arrival, float("inf")):
                    continue

                g_cost[arrival] = new_g
                parent[arrival] = (pos, action)
                heapq.heappush(heap, (new_g + self._manhattan(arrival, goal), new_g, arrival))

        return []

    def _optimistic_greedy_action_to_goal(self) -> Optional[Action]:
        if self.current_pos is None or self.goal_pos is None:
            return None

        tmod = self.total_actions_executed % self.fire_cycle
        candidates: List[Tuple[float, Action]] = []

        for action in self.MOVEMENT_ACTIONS:
            nxt = self._transition_after_action(self.current_pos, action, tmod, known_only=False)
            if nxt is None:
                continue
            arrival, _ = nxt

            target = self._expected_position(self.current_pos, action)
            if target is None:
                continue

            score = -float(self._manhattan(arrival, self.goal_pos))

            # Prefer discovering new cells/edges, but still head to the goal.
            if arrival not in self.safe_cells:
                score += 18.0
            if target not in self.safe_cells:
                score += 8.0
            if (self.current_pos, target) in self.open_edges or (self.current_pos, action) in self.special_transitions:
                score += 3.0

            # Avoid local loops.
            score -= self.episode_visit_count[arrival] * 1.8
            score -= self.global_visit_count[arrival] * 0.04

            # If the current area is looping, strongly prefer a cell not in the
            # recent loop even if it is not the most direct Manhattan move.
            if arrival in self.recent_positions:
                score -= 6.0
            if self._is_looping() and arrival not in self.recent_positions:
                score += 20.0

            candidates.append((score, action))

        if not candidates:
            # Sometimes the only honest safe move is to wait for another fire phase.
            if self._wait_is_safe_for_tmod(self.current_pos, tmod):
                return Action.WAIT
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_action = candidates[0]

        # If the best move is extremely revisited and there is a local frontier,
        # let frontier logic handle it instead.
        best_next = self._transition_after_action(self.current_pos, best_action, tmod, known_only=False)
        if best_next is not None:
            best_arrival, _ = best_next
            if self.episode_visit_count[best_arrival] > 12:
                return None

        return best_action

    def _optimistic_path_to_goal(self, max_expansions: int = 90000) -> List[Action]:
        """
        Honest A* planner over the 64x64 grid.

        Unknown edges are treated as possible. Known walls and known dangerous
        phase/cell combinations are avoided. This lets the honest agent move
        purposefully toward the goal instead of exhaustively mapping an entire
        component before trying the next corridor.
        """
        if self.current_pos is None or self.goal_pos is None:
            return []

        start_state = (self.current_pos, self.total_actions_executed % self.fire_cycle)
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Optional[Action]]] = {
            start_state: (None, None)
        }
        g_cost: Dict[Tuple[Position, int], float] = {start_state: 0.0}

        heap: List[Tuple[float, float, Tuple[Position, int]]] = []
        heapq.heappush(heap, (self._manhattan(self.current_pos, self.goal_pos), 0.0, start_state))

        expansions = 0
        while heap and expansions < max_expansions:
            _, current_g, state = heapq.heappop(heap)
            pos, tmod = state

            if current_g != g_cost.get(state):
                continue

            expansions += 1

            if pos == self.goal_pos:
                return self._reconstruct_time_path(state, parent)

            # Goal-biased action order.
            actions = list(self.MOVEMENT_ACTIONS)
            actions.sort(
                key=lambda a: self._manhattan(
                    self._expected_position(pos, a) or pos,
                    self.goal_pos,
                )
            )

            for action in (*actions, Action.WAIT):
                nxt = self._optimistic_transition_after_action(pos, action, tmod)
                if nxt is None:
                    continue

                next_pos, next_tmod = nxt
                next_state = (next_pos, next_tmod)

                step_cost = 1.0
                if action == Action.WAIT:
                    step_cost = 1.2
                elif next_pos not in self.safe_cells:
                    step_cost = 1.8
                elif self.global_visit_count[next_pos] > 0:
                    step_cost = 1.0 + min(2.0, self.global_visit_count[next_pos] * 0.01)

                new_g = current_g + step_cost

                if new_g >= g_cost.get(next_state, float("inf")):
                    continue

                g_cost[next_state] = new_g
                parent[next_state] = (state, action)
                h = self._manhattan(next_pos, self.goal_pos)
                heapq.heappush(heap, (new_g + h, new_g, next_state))

        return []

    def _optimistic_transition_after_action(
        self,
        pos: Position,
        action: Action,
        tmod: int,
    ) -> Optional[Tuple[Position, int]]:
        phase = self._phase_from_tmod(tmod)
        next_tmod = (tmod + 1) % self.fire_cycle
        next_phase = self._phase_from_tmod(next_tmod)

        if pos in self.death_cells_by_phase[phase]:
            return None

        if action == Action.WAIT:
            arrival = pos
            target = pos
        else:
            target = self._expected_position(pos, action)
            if target is None or not self._inside_grid(target):
                return None
            if self._is_known_wall(pos, target):
                return None
            if (pos, target) in self.death_edges_by_phase[phase]:
                return None

            if (pos, action) in self.special_transitions:
                arrival = self.special_transitions[(pos, action)]
            elif target in self.arrow_directions:
                arrival = self._resolve_learned_arrow_push(pos, target)
                if arrival is None:
                    return None
            else:
                arrival = self._arrival_after_known_teleport(target)

        if target in self.death_cells_by_phase[phase]:
            return None
        if arrival in self.death_cells_by_phase[phase]:
            return None

        if self._rotation_happens_after_tmod(tmod):
            if arrival in self.death_cells_by_phase[next_phase]:
                return None

        return arrival, next_tmod

    # ------------------------------------------------------------------
    # Honest time-expanded planners
    # ------------------------------------------------------------------

    def _fast_path_to_frontier(self) -> List[Action]:
        if self.current_pos is None:
            return []

        start_tmod = self.total_actions_executed % self.fire_cycle

        # Try current phase first, then safe waits into later phases. This keeps
        # the beta fire gate honest: the agent can learn that a route is blocked
        # now but reachable after waiting for the right phase.
        wait_prefix: List[Action] = []
        for wait_steps in range(self.fire_cycle):
            future_tmod = (start_tmod + wait_steps) % self.fire_cycle

            if wait_steps > 0:
                prev_tmod = (start_tmod + wait_steps - 1) % self.fire_cycle
                if not self._wait_is_safe_for_tmod(self.current_pos, prev_tmod):
                    break
                wait_prefix.append(Action.WAIT)

            path = self._fast_path_to_frontier_at_tmod(future_tmod)
            if path:
                return wait_prefix + path

        return []

    def _fast_path_to_frontier_at_tmod(self, tmod: int) -> List[Action]:
        assert self.current_pos is not None

        queue = deque([self.current_pos])
        parent: Dict[Position, Tuple[Optional[Position], Optional[Action]]] = {
            self.current_pos: (None, None)
        }

        while queue:
            pos = queue.popleft()

            if self._untried_actions_from_state(pos, tmod):
                actions: List[Action] = []
                cur = pos
                while parent[cur][0] is not None:
                    prev, action = parent[cur]
                    if action is not None:
                        actions.append(action)
                    cur = prev  # type: ignore[assignment]
                actions.reverse()
                return actions

            for action in self.MOVEMENT_ACTIONS:
                nxt = self._transition_after_action(pos, action, tmod, known_only=True)
                if nxt is None:
                    continue
                arrival, _ = nxt
                if arrival not in self.safe_cells:
                    continue
                if arrival in parent:
                    continue
                parent[arrival] = (pos, action)
                queue.append(arrival)

        return []

    def _time_expanded_path_to_known_cell(self, target: Position, max_expansions: int = 120000) -> List[Action]:
        if self.current_pos is None:
            return []
        if target not in self.safe_cells:
            return []

        start_state = (self.current_pos, self.total_actions_executed % self.fire_cycle)
        queue = deque([start_state])
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Optional[Action]]] = {
            start_state: (None, None)
        }

        expansions = 0
        while queue and expansions < max_expansions:
            pos, tmod = queue.popleft()
            expansions += 1

            if pos == target:
                return self._reconstruct_time_path((pos, tmod), parent)

            for action in (*self.MOVEMENT_ACTIONS, Action.WAIT):
                nxt = self._transition_after_action(pos, action, tmod, known_only=True)
                if nxt is None:
                    continue
                next_pos, next_tmod = nxt
                if next_pos not in self.safe_cells:
                    continue
                state = (next_pos, next_tmod)
                if state in parent:
                    continue
                parent[state] = ((pos, tmod), action)
                queue.append(state)

        return []

    def _time_expanded_path_to_best_frontier(self, max_expansions: int = 3000) -> List[Action]:
        """
        Find the nearest reachable frontier in (position, time) space.

        The older version scored every frontier in the known graph each turn.
        That was correct but slow once the honest map grew large.  For honest
        exploration, the nearest frontier is usually better: it keeps moving,
        keeps discovering edges, and still handles fire timing because WAIT is
        part of the BFS state space.
        """
        if self.current_pos is None:
            return []

        start_state = (self.current_pos, self.total_actions_executed % self.fire_cycle)
        queue = deque([start_state])
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Optional[Action]]] = {
            start_state: (None, None)
        }

        expansions = 0
        while queue and expansions < max_expansions:
            pos, tmod = queue.popleft()
            expansions += 1

            if self._untried_actions_from_state(pos, tmod):
                return self._reconstruct_time_path((pos, tmod), parent)

            # Fixed order is much faster than sorting every expanded state.
            for action in (*self.MOVEMENT_ACTIONS, Action.WAIT):
                nxt = self._transition_after_action(pos, action, tmod, known_only=True)
                if nxt is None:
                    continue
                next_pos, next_tmod = nxt
                if next_pos not in self.safe_cells:
                    continue
                state = (next_pos, next_tmod)
                if state in parent:
                    continue
                parent[state] = ((pos, tmod), action)
                queue.append(state)

        return []

    def _transition_after_action(
        self,
        pos: Position,
        action: Action,
        tmod: int,
        *,
        known_only: bool,
    ) -> Optional[Tuple[Position, int]]:
        phase = self._phase_from_tmod(tmod)
        next_tmod = (tmod + 1) % self.fire_cycle
        next_phase = self._phase_from_tmod(next_tmod)

        if pos in self.death_cells_by_phase[phase]:
            return None

        if action == Action.WAIT:
            arrival = pos
            target = pos
        else:
            target = self._expected_position(pos, action)
            if target is None or not self._inside_grid(target):
                return None
            if self._is_known_wall(pos, target):
                return None
            if (pos, target) in self.death_edges_by_phase[phase]:
                return None

            if (pos, action) in self.special_transitions:
                arrival = self.special_transitions[(pos, action)]
            elif target in self.arrow_directions:
                arrival = self._resolve_learned_arrow_push(pos, target)
                if arrival is None:
                    return None
            elif target in self.teleports and (pos, target) in self.open_edges:
                arrival = self._arrival_after_known_teleport(target)
            elif (pos, target) in self.open_edges:
                arrival = target
            elif not known_only:
                arrival = target
            else:
                return None

        # Entering a known fire/death cell at the current phase is unsafe.
        if target in self.death_cells_by_phase[phase]:
            return None
        if arrival in self.death_cells_by_phase[phase]:
            return None

        # If this action triggers a rotation, the arrival cell must also survive
        # the next phase.
        if self._rotation_happens_after_tmod(tmod):
            if arrival in self.death_cells_by_phase[next_phase]:
                return None

        return arrival, next_tmod

    def _resolve_learned_arrow_push(self, origin: Position, arrow: Position) -> Optional[Position]:
        action = self.arrow_directions.get(arrow)
        if action is None:
            return None

        dest = self._expected_position(arrow, action)
        if dest is None or not self._inside_grid(dest):
            return None
        if self._is_known_wall(arrow, dest):
            return None
        if dest == origin:
            return None
        if dest in self.arrow_directions:
            return self._resolve_learned_arrow_push(origin, dest)
        return self._arrival_after_known_teleport(dest)

    def _untried_actions_from_state(self, position: Position, tmod: int) -> List[Action]:
        phase = self._phase_from_tmod(tmod)
        actions: List[Action] = []

        for action in self.MOVEMENT_ACTIONS:
            next_pos = self._expected_position(position, action)
            if next_pos is None or not self._inside_grid(next_pos):
                continue
            if self._is_known_wall(position, next_pos):
                continue

            # Already known transition, not a frontier edge.
            if (position, action) in self.special_transitions:
                continue
            if (position, next_pos) in self.open_edges:
                continue

            # If the phase has already been tested and failed, wait for a
            # different phase instead of repeating the same death.
            if self._phase_edge_was_attempted(position, next_pos, phase):
                continue

            # Avoid a phase that is already known to be lethal.
            if (position, next_pos) in self.death_edges_by_phase[phase]:
                continue
            if next_pos in self.death_cells_by_phase[phase]:
                continue

            # If this move triggers rotation, do not step into a cell known
            # lethal immediately after rotation.
            if self._rotation_happens_after_tmod(tmod):
                next_phase = self._phase_from_tmod((tmod + 1) % self.fire_cycle)
                if next_pos in self.death_cells_by_phase[next_phase]:
                    continue

            actions.append(action)

        return actions

    # ------------------------------------------------------------------
    # Path helpers and scoring
    # ------------------------------------------------------------------

    def _reconstruct_time_path(
        self,
        target: Tuple[Position, int],
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Optional[Action]]],
    ) -> List[Action]:
        actions: List[Action] = []
        current = target
        while parent[current][0] is not None:
            previous, action = parent[current]
            if action is not None:
                actions.append(action)
            current = previous  # type: ignore[assignment]
        actions.reverse()
        return actions

    def _frontier_score(self, pos: Position, dist_from_current: int) -> float:
        score = float(dist_from_current)

        if self.goal_pos is not None:
            score += self._manhattan(pos, self.goal_pos) * 0.18
        else:
            score -= pos[1] * 0.2

        score += self.episode_visit_count[pos] * 1.5
        score += self.global_visit_count[pos] * 0.08

        # If stuck, prefer escaping farther from the loop.
        if self.stuck_counter > 80:
            score -= dist_from_current * 0.4

        return score

    def _known_open_actions_from(self, position: Position) -> List[Action]:
        actions: List[Action] = []
        tmod = self.total_actions_executed % self.fire_cycle
        for action in self.MOVEMENT_ACTIONS:
            if self._transition_after_action(position, action, tmod, known_only=True) is not None:
                actions.append(action)
        return actions

    def _candidate_actions_from(self, position: Position) -> List[Action]:
        actions: List[Action] = []
        tmod = self.total_actions_executed % self.fire_cycle
        for action in self.MOVEMENT_ACTIONS:
            if self._transition_after_action(position, action, tmod, known_only=False) is not None:
                actions.append(action)
        return actions

    def _exploration_action_score(self, action: Action) -> float:
        assert self.current_pos is not None
        next_pos = self._expected_position(self.current_pos, action)
        if next_pos is None:
            return -1e9

        score = 0.0
        if next_pos not in self.safe_cells:
            score += 180.0
        if self.goal_pos is not None:
            old_dist = self._manhattan(self.current_pos, self.goal_pos)
            new_dist = self._manhattan(next_pos, self.goal_pos)
            if new_dist < old_dist:
                score += 65.0
            else:
                score -= 8.0

        phase = self.fire_phase
        # Prefer phases not previously associated with death for this cell.
        tried_phases = sum(1 for p in range(4) if next_pos in self.death_cells_by_phase[p])
        score -= tried_phases * 15.0
        score -= self.global_visit_count[next_pos] * 0.2

        # Small deterministic tie-breaker by action priority.
        score += {
            Action.MOVE_DOWN: 3.0,
            Action.MOVE_RIGHT: 2.0,
            Action.MOVE_LEFT: 1.0,
            Action.MOVE_UP: 0.0,
            Action.WAIT: -5.0,
        }[action]
        return score

    def _q_action_score(self, action: Action) -> float:
        assert self.current_pos is not None
        state = self._state_key(self.current_pos)
        next_pos = self._expected_position(self.current_pos, action)
        if next_pos is None:
            return -1e9

        score = self.q[(state, action)]
        if next_pos not in self.safe_cells:
            score += 60.0
        if self.goal_pos is not None:
            if self._manhattan(next_pos, self.goal_pos) < self._manhattan(self.current_pos, self.goal_pos):
                score += 20.0
        score -= self.global_visit_count[next_pos] * 0.1
        return score

    def _loop_escape_score(self, action: Action) -> Tuple[int, int, int]:
        assert self.current_pos is not None
        nxt = self._transition_after_action(
            self.current_pos,
            action,
            self.total_actions_executed % self.fire_cycle,
            known_only=True,
        )
        if nxt is None:
            return (999999, 999999, 999999)
        arrival, _ = nxt
        goal_dist = self._manhattan(arrival, self.goal_pos) if self.goal_pos is not None else 0
        return (self.episode_visit_count[arrival], self.global_visit_count[arrival], goal_dist)

    def _future_phase_has_untried_local_edge(self) -> bool:
        if self.current_pos is None:
            return False
        tmod = self.total_actions_executed % self.fire_cycle
        for wait_steps in range(1, self.fire_cycle + 1):
            future_tmod = (tmod + wait_steps) % self.fire_cycle
            if self._untried_actions_from_state(self.current_pos, future_tmod):
                return True
        return False

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _record_position(self, pos: Position) -> None:
        self.global_visit_count[pos] += 1
        self.episode_visit_count[pos] += 1
        self.episode_path.append(pos)
        self.recent_positions.append(pos)

    def _mark_attempted_edge(self, a: Position, b: Position, phase: Optional[int] = None) -> None:
        self.attempted_edges.add((a, b))
        self.attempted_edges.add((b, a))
        if phase is None:
            phase = self.fire_phase
        self.attempted_edges_by_phase[phase].add((a, b))
        self.attempted_edges_by_phase[phase].add((b, a))

    def _mark_open_edge(self, a: Position, b: Position) -> None:
        self.open_edges.add((a, b))
        self.open_edges.add((b, a))
        # Once an edge is open, it is spatially tried in every phase.
        for phase in range(4):
            self.attempted_edges_by_phase[phase].add((a, b))
            self.attempted_edges_by_phase[phase].add((b, a))
        self.attempted_edges.add((a, b))
        self.attempted_edges.add((b, a))

    def _mark_wall(self, a: Position, b: Position) -> None:
        self.wall_edges.add((a, b))
        self.wall_edges.add((b, a))
        for phase in range(4):
            self.attempted_edges_by_phase[phase].add((a, b))
            self.attempted_edges_by_phase[phase].add((b, a))
        self.attempted_edges.add((a, b))
        self.attempted_edges.add((b, a))

    def _is_known_wall(self, a: Position, b: Position) -> bool:
        return (a, b) in self.wall_edges or (b, a) in self.wall_edges

    def _phase_edge_was_attempted(self, a: Position, b: Position, phase: int) -> bool:
        return (a, b) in self.attempted_edges_by_phase[phase] or (b, a) in self.attempted_edges_by_phase[phase]

    def _arrival_after_known_teleport(self, pos: Position) -> Position:
        """Return the one-hop arrival for a paired/two-way teleport pad."""
        return self.teleports.get(pos, pos)

    def _infer_arrow_direction(self, arrow_pos: Position, destination: Position) -> Optional[Action]:
        dx = destination[0] - arrow_pos[0]
        dy = destination[1] - arrow_pos[1]
        for action, delta in self.DELTAS.items():
            if action == Action.WAIT:
                continue
            if delta == (dx, dy):
                return action
        return None

    def _is_looping(self) -> bool:
        if len(self.recent_positions) < 40:
            return False
        return len(set(self.recent_positions)) <= 8

    # ------------------------------------------------------------------
    # Timing / danger utilities
    # ------------------------------------------------------------------

    @property
    def fire_phase(self) -> int:
        return self._phase_from_tmod(self.total_actions_executed % self.fire_cycle)

    def _phase_from_tmod(self, tmod: int) -> int:
        return (tmod // self.fire_rotation_interval) % 4

    def _rotation_happens_after_tmod(self, tmod: int) -> bool:
        return (tmod + 1) % self.fire_rotation_interval == 0

    def _wait_is_safe_for_tmod(self, pos: Optional[Position], tmod: int) -> bool:
        if pos is None:
            return False
        phase = self._phase_from_tmod(tmod)
        if pos in self.death_cells_by_phase[phase]:
            return False
        if self.use_image_guidance and pos in self.visible_fire_by_phase[phase]:
            return False

        next_tmod = (tmod + 1) % self.fire_cycle
        next_phase = self._phase_from_tmod(next_tmod)
        if self._rotation_happens_after_tmod(tmod):
            if pos in self.death_cells_by_phase[next_phase]:
                return False
            if self.use_image_guidance and pos in self.visible_fire_by_phase[next_phase]:
                return False
        return True

    # ------------------------------------------------------------------
    # Confusion / state / geometry utilities
    # ------------------------------------------------------------------

    def _convert_intended_to_requested(self, intended_action: Action) -> Action:
        if self.confused_next_turn:
            return self.INVERTED[intended_action]
        return intended_action

    def _state_key(self, position: Optional[Position]) -> State:
        if position is None:
            return (-1, -1, self.fire_phase, int(self.confused_next_turn))
        x, y = position
        return (x, y, self.fire_phase, int(self.confused_next_turn))

    def _expected_position(self, position: Optional[Position], action: Action) -> Optional[Position]:
        if position is None:
            return None
        if action == Action.WAIT:
            return position
        dx, dy = self.DELTAS[action]
        return (position[0] + dx, position[1] + dy)

    @staticmethod
    def _inside_grid(pos: Position) -> bool:
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
