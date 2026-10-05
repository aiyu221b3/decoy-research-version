"""legal next-state maxima and every-visit returns are different estimators."""

from collections import defaultdict

import numpy as np

from .actions import INV_ACTIONS
from .encoding import tabular_state


class TabularLearner:
    def __init__(self, config):
        self.config = config
        self.q = defaultdict(lambda: np.zeros(len(INV_ACTIONS)))
        self.counts = defaultdict(lambda: np.zeros(len(INV_ACTIONS), dtype=np.int64))
        self.rng = np.random.default_rng(config.seed)

    def act(self, obs, epsilon=0.1):
        state = tabular_state(obs, self.config.exact_resources)
        legal = np.flatnonzero(obs.action_mask)
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        values = self.q[state][legal]
        return int(self.rng.choice(legal[values == values.max()]))

    def update(self, transitions):
        if self.config.algorithm == "monte_carlo":
            value = 0.0
            for state, action, reward, next_state, next_mask, done in reversed(transitions):
                value = reward+self.config.gamma*value
                self.counts[state][action] += 1
                self.q[state][action] += (value-self.q[state][action])/self.counts[state][action]
        else:
            for state, action, reward, next_state, next_mask, done in transitions:
                future = 0.0 if done else self.q[next_state][np.asarray(next_mask, dtype=bool)].max()
                target = reward+self.config.gamma*future
                self.q[state][action] += 0.1*(target-self.q[state][action])

    def state_dict(self):
        return {"q": {",".join(map(str, key)): value.tolist() for key, value in self.q.items()},
                "counts": {",".join(map(str, key)): value.tolist() for key, value in self.counts.items()},
                "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.q.update({tuple(map(int, key.split(","))): np.asarray(value) for key, value in state["q"].items()})
        self.counts.update({tuple(map(int, key.split(","))): np.asarray(value, dtype=np.int64) for key, value in state["counts"].items()})
        self.rng.bit_generator.state = state["rng"]
