import numpy as np
import random
from typing import List, Optional, Iterator
from tqdm import tqdm


class Player:
    def play(self, state) -> int:
        raise NotImplementedError()

    def outcome(self, value: int):
        pass


def rollout(player: Player, iters=10_000) -> np.ndarray:
    outcomes = np.zeros(iters)
    for i in tqdm(range(iters)):
        outcomes[i] = _rollout_one(player)
    return outcomes


def board_value(board: List[str]) -> Optional[int]:
    if all(cell != " " for row in board for cell in row):
        return 0

    def lines(board: List[str]) -> Iterator[str]:
        r = range(3)
        for i in r:
            yield board[i]
        for j in r:
            yield "".join(board[i][j] for i in r)
        yield "".join(board[i][i] for i in r)
        yield "".join(board[i][2 - i] for i in r)

    for line in lines(board):
        if line == "XXX":
            return 1
        elif line == "OOO":
            return -1

    return None


def set_square(row, col, state, char):
    state[row] = state[row][:col] + char + state[row][col + 1 :]
    return state


def _rollout_one(player: Player) -> int:
    state = ["   ", "   ", "   "]
    is_player_turn = True

    while (value := board_value(state)) is None:
        available_moves = [
            (i, j) for i in range(3) for j in range(3) if state[i][j] == " "
        ]

        if is_player_turn:
            move = player.play("\n".join(state))
            row, col = move // 3, move % 3
            if (row, col) not in available_moves:
                value = -1
                break
            state = set_square(row, col, state, "X")
        else:
            if available_moves:
                row, col = random.choice(available_moves)
                state = set_square(row, col, state, "O")

        is_player_turn = not is_player_turn

    player.outcome(value)
    return value
