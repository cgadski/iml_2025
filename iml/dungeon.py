import numpy as np
from typing import Optional, Iterable


class Dungeon:
    """
    A simple gridworld/dungeon environment.

    `transitions[action, now, then]` gives the probability of transitioning into state `then` given that we are in state `now` and perform action `action`.

    We begin in state `start` and want to reach state `end` as quickly as possible. Walls (tiles marked with `#`) are impassable, and holes (tiles marked with `O`) teleport us back to the start when we move into them.
    """

    def __init__(self, lines: Iterable[str], p_slippery=0.5):
        self.tile_mat = self._read_dungeon(lines)  # h w -> char
        self.h, self.w = self.tile_mat.shape
        self.n_states = self.h * self.w
        self.tiles = self.tile_mat.flatten()
        self.p_slippery = p_slippery

        self.start = self.tile_indices("s")[0]
        self.end = self.tile_indices("e")[0]
        self.transitions = self._build_transitions()

    def tile_indices(self, char: str) -> np.ndarray:
        return np.where(self.tiles == char)[0]

    def _read_dungeon(self, lines: Iterable[str]) -> np.ndarray:
        max_len = max(len(line) for line in lines)
        return np.array([list(line.ljust(max_len)) for line in lines])

    def _evaluate_action(self, action: np.ndarray, pos: np.ndarray) -> np.ndarray:
        _, _ = pos.shape  # b 2
        start_pos = np.array([x[0] for x in np.where(self.tile_mat == "s")])

        directions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        pos_ = pos + directions[action]  # Move
        pos_ = np.maximum(pos_, 0)  # Clip to bounds
        pos_ = np.minimum(pos_, np.array([self.h - 1, self.w - 1]))
        tile_ = self.tile_mat[pos_[:, 0], pos_[:, 1], None]
        pos_ = np.where(tile_ == "#", pos, pos_)  # Don't move into walls
        pos_ = np.where(tile_ == "O", start_pos, pos_)  # Holes teleport to start
        return pos_

    def _build_transitions(self) -> np.ndarray:
        mesh = np.mgrid[0:4, 0 : self.h, 0 : self.w]
        action, y, x = [arr.flatten() for arr in mesh]

        y_, x_ = self._evaluate_action(action, np.stack([y, x], axis=-1)).T
        transitions = np.eye(self.n_states)[
            np.ravel_multi_index((y_, x_), (self.h, self.w))
        ]
        transitions = transitions.reshape((4, self.h * self.w, self.h * self.w))
        action_transfer = (self.p_slippery / 4) * np.ones((4, 4)) + (
            1 - self.p_slippery
        ) * np.eye(4)
        return np.tensordot(action_transfer, transitions, axes=(1, 0))

    def show(self, vect: Optional[np.ndarray] = None, size=0.6, values=True):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots()
        plt.tight_layout()
        ax.set_xlim(-0.5, self.w - 0.5)
        ax.set_ylim(self.h - 0.5, -0.5)
        fig.set_size_inches(self.w * size, self.h * size)
        ax.set_aspect(1)
        ax.set_axis_off()

        cmap = plt.get_cmap("Blues")

        y, x = [x.flatten() for x in np.mgrid[0 : self.h, 0 : self.w]]
        chars = self.tile_mat[y, x].flatten()
        for i in range(self.n_states):
            text_opts = lambda style: {
                "color": "black",
                "ha": "center",
                "va": "center",
                "fontweight": style,
            }
            is_obstacle = chars[i] in ["#", "O"]

            text_pos = lambda offset: (x[i], y[i] + offset)
            if is_obstacle:
                ax.text(*text_pos(0), chars[i], **text_opts("bold"))
            else:
                ax.text(*text_pos(0.2), chars[i], **text_opts("regular"))

            rect_opts = {"facecolor": "white", "edgecolor": "grey"}
            if chars[i] in ["#", "O"]:
                ax.add_patch(Rectangle((x[i] - 0.5, y[i] - 0.5), 1, 1, **rect_opts))
                continue

            rect_opts = {"facecolor": "white", "edgecolor": "grey"}
            if vect is not None and vect.dtype.type != np.str_:
                norm = Normalize(vect.min(), vect.max())
                rect_opts["facecolor"] = cmap(norm(vect[i]) * 0.8)

            ax.add_patch(Rectangle((x[i] - 0.5, y[i] - 0.5), 1, 1, **rect_opts))

            if values and vect is not None:
                if vect.dtype.type == np.str_:
                    text = vect[i]
                else:
                    text = f"{vect[i]:.1f}"
                ax.text(*text_pos(-0.2), text, **text_opts("regular"))

    def show_policy(self, actions):
        chars = np.array(["↓", "↑", "→", "←"])
        self.show(chars[actions])


def make_ring_dungeon():
    return Dungeon(
        [
            "s   ",
            " ## ",
            "   e",
        ]
    )


def make_big_dungeon():
    return Dungeon(
        [
            "s       #",
            "        #",
            "#####   #",
            "        #",
            "        #",
            "  OOOOOO#",
            "        #",
            "       e#",
        ]
    )
