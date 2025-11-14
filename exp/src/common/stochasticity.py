import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class StatefulRng:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        if self.state is None:
            set_seed(self.seed)
        else:
            random.setstate(self.state[0])
            np.random.set_state(self.state[1])
            torch.set_rng_state(self.state[2])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(self.state[3])

    def __exit__(self, exc_type, exc_value, traceback):
        self.state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        )
