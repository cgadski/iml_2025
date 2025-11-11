import numpy as np
import pandas as pd
from random import choice
from tqdm import tqdm
from matplotlib import pyplot as plt


class BanditProblem:
    def __init__(self, n_arms=100):
        self.n_arms = 100
        self.good_arm = choice(range(100))

    def take_action(self, arm) -> float:
        if arm == self.good_arm:
            return np.random.random() <= 2 / 3
        else:
            return np.random.random() <= 0.5
