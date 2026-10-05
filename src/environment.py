"""structured communication changes costs, evidence and opponent pressure."""

import itertools
from collections import Counter

import numpy as np

from .actions import INV_ACTIONS, TRADER_ACTIONS, decode, trader_target
from .budget import WorkBudget
from .config import Scenario
from .language import render
from .populations import ReactivePopulation
from .types import Event, Evidence, Observation, TraderView, Transition

PERMUTATIONS = tuple(itertools.permutations(range(3)))
INV_UTILITY = {"fraudster_caught": 10.0, "innocent_caught": -10.0,
               "trickster_caught": -3.0, "timeout": -1.0}
TRADER_UTILITY = np.array(((5, -10, 1, 0), (-10, 10, 2, 1), (-3, 5, -10, 1)), dtype=float)
OUTCOMES = ("fraudster_caught", "innocent_caught", "trickster_caught", "timeout")


class InvestigationEnv:
    def __init__(self, scenario=None, budget=None, trader_controller=None):
        self.scenario = scenario or Scenario()
        self.budget = budget or WorkBudget(None)
        self.population = ReactivePopulation(self.scenario.family, self.scenario.noise)
        self.trader_controller = trader_controller
        self.rng = np.random.default_rng()
        self._started = False

    @property
    def max_episode_work(self):
        n = self.scenario.attention + self.scenario.focus
        background = 2 + self.scenario.witnesses if self.scenario.background else 0
        return 2*n + (n+1)*(4+background) + self.scenario.audits

    def reset(self, *, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if not self.budget.can_reserve(self.max_episode_work):
            from .budget import BudgetExhausted
            raise BudgetExhausted("not enough shared work for a complete bounded episode")
        s = self.scenario
        self._roles = self.rng.permutation(3)
        self._latent = np.zeros(3)
        self._market = np.zeros(3)
        self._volume = np.ones(3)
        self._market_history = [[] for _ in range(3)]
        self._inspection = [(False, 0.0, 0.0) for _ in range(3)]
        self._record_scores = [[] for _ in range(3)]
        self._pending = {}
        self._pressure = np.zeros(3)
        self._speech = [Counter() for _ in range(3)]
        self._last_market = ["normal"] * 3
        self._events = []
        self._macro, self._liquidity, self._news = 0.0, 1.0, 0.0
        self.attention, self.focus, self.audits = s.attention, s.focus, s.audits
        self.communication, self.credibility = s.communication, 0.6
        self.accusation, self.tick, self.actions = None, 0, 0
        self.done, self.outcome, self.reason = False, None, None
        self._unreported_costs = np.zeros(3)
        self.trader_returns = np.zeros(3)
        self.population.reset()
        self._started = True
        self._advance_market()
        return self.observation()

    @staticmethod
    def trader_cost(action_id, phase="market"):
        action = TRADER_ACTIONS[action_id]
        market = {"normal": 0.0, "conceal": -0.02, "signal": -0.01}[action.market] if phase == "market" else 0.0
        return market - (0.005 if action.speech == "delay" else 0.0)

    def _choose(self, identity, public, phase):
        view = TraderView(public, identity, int(self._roles[identity]), float(self._latent[identity]))
        self.budget.charge("trader_decisions")
        controller = self.trader_controller
        action_id = self.population.act(view, self.rng, phase) if controller is None else controller(view, self.rng, phase)
        if type(action_id) is not int or not 0 <= action_id < len(TRADER_ACTIONS):
            raise ValueError("trader controller returned an invalid action")
        if phase == "response" and action_id >= 9:
            raise ValueError("response phase allows only speech, not a market action")
        cost = self.trader_cost(action_id, phase)
        self._unreported_costs[identity] += cost
        self.trader_returns[identity] += cost
        return TRADER_ACTIONS[action_id]

    def _event(self, speaker, kind, target=None, value=0.0):
        event = Event(len(self._events), self.tick, speaker, kind, target, float(value), render(kind, target, value))
        self._events.append(event)
        if speaker < 3:
            self._speech[speaker][kind] += 1
            if target == speaker and target in self._pending:
                if kind == "provide":
                    self._pending[target] = max(self.tick+1, self._pending[target]-1)
                elif kind == "delay":
                    self._pending[target] += 1
        return event

    def _reply(self, target):
        action = self._choose(target, self.observation(), "response")
        self._event(target, action.speech, trader_target(target, action.relative_target))

    def _advance_market(self, focused=None):
        self.budget.charge("market_updates")
        public = self.observation()
        choices = [self._choose(i, public, "market") for i in range(3)]
        self.tick += 1
        if self.scenario.background:
            self.budget.charge("background_updates", 2)
            self._macro = 0.85*self._macro + self.rng.normal(0, 0.16)
            self._liquidity = float(np.clip(0.8*self._liquidity + 0.2 + self.rng.normal(0, 0.12), 0.35, 1.8))
            self._news = float(self.rng.normal(0, 0.2))
        self._pressure *= 0.85
        for i, action in enumerate(choices):
            self._last_market[i] = action.market
            self._latent[i] = 0.85*self._latent[i] + self.rng.normal(0, 0.12)
            drift = (0.0, 0.55, 0.20)[self._roles[i]]
            scale = {"normal": 1.0, "conceal": 0.4, "signal": 1.5}[action.market]
            sigma = 0.35 / np.sqrt(self._liquidity)
            if focused == i:
                sigma *= 0.45
            self._market[i] = self._macro + self._news + scale*(self._latent[i]+drift) + self.rng.normal(0, sigma)
            self._volume[i] = float(max(0.1, self._liquidity + abs(self._latent[i]) + self.rng.normal(0, 0.15)))
            self._market_history[i].append(float(self._market[i]))
            self._event(i, action.speech, trader_target(i, action.relative_target))
        for target, due in list(self._pending.items()):
            if self.tick >= due:
                self.budget.charge("background_updates")
                drift = (0.0, 0.8, 0.3)[self._roles[target]]
                conceal = self._last_market[target] == "conceal"
                score = drift*(0.65 if conceal else 1.0) + self.rng.normal(0, 0.3/self._liquidity)
                self._record_scores[target].append(float(score))
                self._event(4, "record", target, score)
                del self._pending[target]
        if self.scenario.background:
            for witness in range(self.scenario.witnesses):
                self.budget.charge("background_updates")
                if self.rng.random() < 0.25:
                    target = int(self.rng.integers(3))
                    score = self._market[target] - self._macro + self.rng.normal(0, 0.45)
                    self._event(4+witness, "witness", target, score)

    def _mask(self):
        if not self._started or self.done:
            return (False,) * len(INV_ACTIONS)
        legal = []
        for action in INV_ACTIONS:
            allowed = self.attention > 0
            if action.kind == "focus":
                allowed = self.focus > 0
            elif action.kind == "vote":
                allowed = allowed and self.accusation == action.target
            elif action.kind == "audit":
                allowed = allowed and self.scenario.records and self.audits > 0 and action.target not in self._pending
            elif action.kind == "statement":
                allowed = allowed and self.scenario.statements and self.communication > 0
                if action.intent == "publish":
                    allowed = allowed and (self._inspection[action.target][0] or bool(self._record_scores[action.target]))
            legal.append(bool(allowed))
        return tuple(legal)

    def observation(self):
        if not self._started:
            raise RuntimeError("reset before requesting an observation")
        evidence = []
        for i in range(3):
            count = sum(self._speech[i].values())
            inspected, mean, trend = self._inspection[i]
            evidence.append(Evidence(float(self._market[i]), float(self._volume[i]), inspected, mean, trend,
                float(np.mean(self._record_scores[i])) if self._record_scores[i] else 0.0,
                len(self._record_scores[i]), i in self._pending, float(self._pressure[i]), count,
                self._speech[i]["defend"]/max(count, 1), self._speech[i]["accuse"]/max(count, 1),
                self._speech[i]["support"]/max(count, 1)))
        return Observation(tuple(evidence), tuple(self._events[-8:]), self.attention, self.scenario.attention,
            self.tick, self.scenario.horizon, self.focus, self.audits, self.communication,
            self.credibility, self.accusation, self._macro, self._liquidity, self._news, self._mask())

    def step(self, action_id):
        if not self._started or self.done:
            raise RuntimeError("reset is required before stepping a live episode")
        action = decode(action_id)
        if not self._mask()[int(action_id)]:
            raise ValueError("action is not legal")
        self.budget.charge("investigator_actions")
        self.actions += 1
        focused = None
        target = action.target
        if action.kind == "focus":
            self.focus -= 1
            focused = target
        else:
            self.attention -= 1
        if action.kind == "observe":
            history = self._market_history[target][-5:]
            self._inspection[target] = True, float(np.mean(history)), float(history[-1]-history[0])
        elif action.kind == "accuse":
            self.accusation = target
            self._pressure[target] = min(1.0, self._pressure[target]+0.35)
            self._event(3, "accuse", target)
            self._reply(target)
        elif action.kind == "vote":
            self._event(3, "vote", target)
            self.outcome = ("innocent_caught", "fraudster_caught", "trickster_caught")[self._roles[target]]
            self.reason, self.done = "vote", True
        elif action.kind == "audit":
            self.audits -= 1
            self._pending[target] = self.tick + self.scenario.record_delay
            self._event(3, "audit", target)
        elif action.kind == "statement":
            self.communication -= 1
            self._event(3, action.intent, target)
            if action.intent == "question":
                self.credibility = min(1.0, self.credibility+0.02)
                self._reply(target)
            elif action.intent == "warn":
                self._pressure[target] = min(1.0, self._pressure[target]+0.55)
                self.credibility = max(0.0, self.credibility-0.05)
            else:
                self._pressure[target] = min(1.0, self._pressure[target]+0.3)
                self.credibility = min(1.0, self.credibility+0.05)
        if not self.done:
            if self.attention == 0:
                self.done, self.reason, self.outcome = True, "attention", "timeout"
            else:
                self._advance_market(focused)
                if self.tick >= self.scenario.horizon:
                    self.done, self.reason, self.outcome = True, "horizon", "timeout"
        original = INV_UTILITY[self.outcome] if self.done else 0.0
        reward = original if self.scenario.objective == "original" else float(self.outcome == "fraudster_caught")
        trader_rewards = self._unreported_costs.copy()
        self._unreported_costs.fill(0)
        if self.done:
            terminal = TRADER_UTILITY[self._roles, OUTCOMES.index(self.outcome)]
            trader_rewards += terminal
            self.trader_returns += terminal
        return Transition(self.observation(), reward, original, self.done, False,
                          self.outcome, self.reason, tuple(float(x) for x in trader_rewards))

    def training_labels(self):
        if not self.done:
            raise RuntimeError("labels are only available after episode collection")
        return {"roles": tuple(int(x) for x in self._roles),
                "permutation": PERMUTATIONS.index(tuple(self._roles)),
                "trader_terminal": tuple(float(x) for x in TRADER_UTILITY[self._roles, OUTCOMES.index(self.outcome)])}

    def transcript(self):
        return tuple(self._events)
