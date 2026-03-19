import random
from environment import Action

class NaiveAgent:
    def __init__(self):
        self.last_move = None
        self.current_pos = None
        self.visited = set()

    def reset_episode(self):
        self.last_move = None
        self.current_pos = None
        self.visited.clear()

    def plan_turn(self, last_result):
        # Update state
        if last_result:
            self.current_pos = last_result.current_position
            self.visited.add(self.current_pos)

        moves = [
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT
        ]

        # Avoid going back immediately
        if self.last_move:
            opposite = {
                Action.MOVE_UP: Action.MOVE_DOWN,
                Action.MOVE_DOWN: Action.MOVE_UP,
                Action.MOVE_LEFT: Action.MOVE_RIGHT,
                Action.MOVE_RIGHT: Action.MOVE_LEFT
            }
            if opposite[self.last_move] in moves:
                moves.remove(opposite[self.last_move])

        move = random.choice(moves)
        self.last_move = move

        return [move]
