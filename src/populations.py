"""a trader knows its own role, not the hidden roles of its neighbors."""

import numpy as np

from .actions import TRADER_ACTIONS


class ReactivePopulation:
    def __init__(self, family="noisy", noise=0.2):
        self.family, self.noise = family, noise

    def reset(self):
        pass

    def probabilities(self, view):
        role = view.own_role
        market = np.array(((0.75, 0.15, 0.10), (0.20, 0.65, 0.15), (0.30, 0.15, 0.55))[role])
        speech = np.array(((0.12, 0.35, 0.20, 0.15, 0.04, 0.04, 0.04, 0.03, 0.03),
                           (0.08, 0.18, 0.12, 0.05, 0.17, 0.16, 0.16, 0.04, 0.04),
                           (0.10, 0.09, 0.11, 0.07, 0.10, 0.12, 0.12, 0.15, 0.14))[role])
        pressure = view.public.traders[view.identity].pressure
        if self.family == "imitative" and role == 1:
            speech = np.array((0.10, 0.36, 0.24, 0.10, 0.10, 0.03, 0.03, 0.02, 0.02))
        if self.family == "strategic":
            market = market + np.array((0.25 * pressure, -0.15 * pressure, -0.10 * pressure))
            speech[1] += 0.2 * pressure
            speech[3] += 0.2 * view.public.credibility
        if self.family == "conventional":
            market = np.array((0.88, 0.06, 0.06)) if role == 0 else market
        q = self.noise if self.family != "conventional" else 0.0
        market = np.maximum(market, 0.01)
        market = (1-q) * market / market.sum() + q / 3
        speech = (1-q) * speech / speech.sum() + q / 9
        return np.outer(market, speech).ravel()

    def act(self, view, rng, phase="market"):
        probabilities = self.probabilities(view)
        if phase == "response":
            return int(rng.choice(9, p=probabilities.reshape(3, 9).sum(axis=0)))
        return int(rng.choice(len(TRADER_ACTIONS), p=probabilities))
