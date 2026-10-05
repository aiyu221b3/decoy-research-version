"""simple completion policies expose incentive and resource confounds."""

import itertools

import numpy as np

from .actions import Action, encode

PERMUTATIONS = np.asarray(list(itertools.permutations(range(3))))


def evidence_probabilities(obs):
    scores = np.zeros((3, 3))
    for i, evidence in enumerate(obs.traders):
        residual = evidence.market-obs.macro-obs.news
        scores[i] -= 0.5*((residual-np.array((0.0, 0.35, 0.25)))/0.55)**2
        if evidence.record_count:
            scores[i] -= 0.5*((evidence.record_score-np.array((0.0, 0.65, 0.30)))/0.35)**2
    logp = np.asarray([sum(scores[i, role] for i, role in enumerate(p)) for p in PERMUTATIONS])
    p = np.exp(logp-logp.max())
    return p/p.sum()


class ControlPolicy:
    def __init__(self, name="evidence", seed=0, objective="success"):
        if name not in ("random", "immediate", "reserve", "evidence"):
            raise ValueError("unknown control")
        self.name, self.objective = name, objective
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.target = None

    def probabilities(self, obs):
        return evidence_probabilities(obs)

    def act(self, obs):
        legal = np.flatnonzero(obs.action_mask)
        if self.name == "random":
            return int(self.rng.choice(legal))
        if self.name == "immediate":
            if self.target is None:
                self.target = int(self.rng.integers(3))
            kind = "vote" if obs.accusation == self.target else "accuse"
            candidate = encode(Action(kind, self.target))
            return candidate if obs.action_mask[candidate] else int(legal[0])
        posterior = evidence_probabilities(obs)
        marginals = np.array([[posterior[PERMUTATIONS[:, i] == role].sum() for role in range(3)] for i in range(3)])
        values = marginals[:, 1] if self.objective == "success" else marginals @ np.array((-10, 10, -3))
        self.target = int(np.argmax(values))
        vote = encode(Action("vote", self.target))
        if obs.action_mask[vote]:
            return vote
        reserve_now = obs.attention <= 3 or obs.tick+2 >= obs.horizon
        if self.name == "reserve" and not reserve_now:
            return encode(Action("observe", (obs.tick-1) % 3))
        if obs.accusation is not None and obs.attention <= 2:
            return encode(Action("vote", obs.accusation))
        if self.name == "reserve" or reserve_now or marginals[self.target, 1] > 0.78:
            accuse = encode(Action("accuse", self.target))
            if obs.attention >= 2 and obs.tick+1 < obs.horizon and obs.action_mask[accuse]:
                if self.objective != "original" or values[self.target] > -1:
                    return accuse
            return encode(Action("wait", None))
        for target in np.argsort(-marginals[:, 1]):
            audit = encode(Action("audit", int(target)))
            if obs.action_mask[audit] and not obs.traders[target].record_count:
                return audit
        question = encode(Action("statement", self.target, "question"))
        if obs.action_mask[question]:
            return question
        return encode(Action("observe", self.target))
