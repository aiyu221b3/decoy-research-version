"""actor samples contain features and actions; labels arrive after collection."""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.func import functional_call

from .actions import Action, INV_ACTIONS, encode
from .encoding import encode_public, encode_trader
from .environment import InvestigationEnv
from .models import distribution


def episode_seed(seed, phase, episode):
    return int(np.random.SeedSequence((seed, phase, episode)).generate_state(1)[0])


@dataclass
class Trajectory:
    features: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    values: list = field(default_factory=list)
    role_probabilities: list = field(default_factory=list)
    success_predictions: list = field(default_factory=list)
    role_label: int = 0
    successful: bool = False
    guarded: bool = False

    def tensors(self):
        return (torch.from_numpy(np.stack(self.features)), torch.tensor(self.masks, dtype=torch.bool),
                torch.tensor(self.actions, dtype=torch.long))


def collect(scenario, budget, model, seed, exact=True, greedy=False, trader_model=None, parameters=None,
            completion_guard=False):
    trajectory = Trajectory(guarded=completion_guard)
    traders = [Trajectory() for _ in range(3)]
    trader_hidden, trader_previous = [None]*3, [-1]*3

    def trader_controller(view, rng, phase):
        i = view.identity
        feature = encode_trader(view, trader_previous[i], exact, phase)
        mask = np.ones(27, dtype=bool)
        if phase == "response":
            mask[9:] = False
        with torch.no_grad():
            output = trader_model(torch.from_numpy(feature).unsqueeze(0), trader_hidden[i])
            logits, value, role, success, trader_hidden[i] = output
            dist = distribution(logits[0], torch.from_numpy(mask))
            action = dist.probs.argmax() if greedy else dist.sample()
        record = traders[i]
        record.features.append(feature)
        record.masks.append(mask.tolist())
        record.actions.append(int(action))
        record.logp.append(float(dist.log_prob(action)))
        record.values.append(float(value[0]))
        record.rewards.append(InvestigationEnv.trader_cost(int(action), phase))
        trader_previous[i] = int(action)
        return int(action)

    env = InvestigationEnv(scenario, budget, trader_controller if trader_model is not None else None)
    obs = env.reset(seed=seed)
    hidden, previous = None, -1
    public_trace = []
    while True:
        public_trace.append(obs.as_dict())
        feature = encode_public(obs, previous, exact)
        with torch.no_grad():
            arguments = (torch.from_numpy(feature).unsqueeze(0), hidden)
            output = model(*arguments) if parameters is None else functional_call(model, parameters, arguments)
            logits, value, role, success, hidden = output
            dist = distribution(logits[0], torch.tensor(obs.action_mask, dtype=torch.bool))
            action = dist.probs.argmax() if greedy else dist.sample()
            if completion_guard and (obs.attention <= 3 or obs.tick+2 >= obs.horizon):
                if obs.accusation is not None:
                    action = torch.tensor(encode(Action("vote", obs.accusation)))
                elif obs.attention >= 2 and obs.tick+1 < obs.horizon:
                    from .environment import PERMUTATIONS
                    probability = torch.softmax(role[0], -1).numpy()
                    marginals = np.asarray([[sum(p for p, z in zip(probability, PERMUTATIONS) if z[i] == r)
                                             for r in range(3)] for i in range(3)])
                    scores = marginals[:, 1] if scenario.objective == "success" else marginals @ np.array((-10, 10, -3))
                    if scenario.objective == "success" or scores.max() > -1:
                        action = torch.tensor(encode(Action("accuse", int(scores.argmax()))))
        trajectory.features.append(feature)
        trajectory.masks.append(obs.action_mask)
        trajectory.actions.append(int(action))
        trajectory.logp.append(float(dist.log_prob(action)))
        trajectory.values.append(float(value[0]))
        trajectory.role_probabilities.append(torch.softmax(role[0], -1).tolist())
        trajectory.success_predictions.append(float(success[0]))
        transition = env.step(int(action))
        trajectory.rewards.append(transition.reward)
        if transition.terminated:
            break
        obs, previous = transition.observation, int(action)
    labels = env.training_labels()
    trajectory.role_label = labels["permutation"]
    trajectory.successful = transition.outcome == "fraudster_caught"
    for i, record in enumerate(traders):
        if record.rewards:
            record.rewards[-1] += labels["trader_terminal"][i]
    details = {"outcome": transition.outcome, "reason": transition.reason,
               "original_utility": transition.reference_reward, "actions": env.actions,
               "market_updates": env.tick, "trader_returns": env.trader_returns.tolist(),
               "action_ids": trajectory.actions, "role_label": labels["permutation"],
               "public_trace": public_trace,
               "events": [event.__dict__ for event in env.transcript()]}
    return trajectory, traders, details
