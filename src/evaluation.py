"""fixed-policy uncertainty is different from variation between training seeds."""

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .actions import INV_ACTIONS
from .environment import InvestigationEnv
from .io import write_json
from .policies import ControlPolicy
from .rollouts import collect, episode_seed
from .tabular import TabularLearner
from .training import load_run


def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half), float(center+half)]


def summarize(rows):
    counts = Counter(row["outcome"] for row in rows)
    n = len(rows)
    if not n:
        return {"episodes": 0, "outcomes": {}}
    resolved = n-counts["timeout"]
    actions = Counter(str(action) for row in rows for action in row["action_ids"])
    probabilities = np.array(list(actions.values()), dtype=float)
    probabilities /= probabilities.sum()
    result = {"episodes": n, "outcomes": dict(counts), "success": counts["fraudster_caught"]/n,
        "innocent_error": counts["innocent_caught"]/n, "trickster_error": counts["trickster_caught"]/n,
        "timeout": counts["timeout"]/n, "vote_coverage": resolved/n,
        "conditional_success": counts["fraudster_caught"]/resolved if resolved else None,
        "original_utility": float(np.mean([row["original_utility"] for row in rows])),
        "mean_actions": float(np.mean([len(row["action_ids"]) for row in rows])),
        "success_wilson_fixed_policy": wilson(counts["fraudster_caught"], n),
        "action_counts": dict(actions), "action_entropy": float(-sum(probabilities*np.log(probabilities)))}
    predictions = [row for row in rows if "role_probabilities" in row]
    if predictions:
        p = np.array([r["role_probabilities"] for r in predictions])
        y = np.eye(6)[[r["role_label"] for r in predictions]]
        result["role_brier"] = float(np.mean(np.sum((p-y)**2, axis=1)))
        result["role_accuracy"] = float(np.mean(p.argmax(axis=1) == y.argmax(axis=1)))
        result["success_brier"] = float(np.mean([(r["success_prediction"]-float(r["outcome"] == "fraudster_caught"))**2 for r in predictions]))
    return result


def evaluate(scenario, budget, episodes, seed, model=None, config=None, control=None,
             trader_model=None, greedy=True, completion_guard=False):
    rows = []
    if model is not None:
        torch.manual_seed(seed)
    for index in range(episodes):
        if not budget.can_reserve(InvestigationEnv(scenario).max_episode_work):
            break
        start = episode_seed(seed, 1, index)
        if model is not None and not isinstance(model, TabularLearner):
            trajectory, _, details = collect(scenario, budget, model, start,
                config.exact_resources, greedy=greedy, trader_model=trader_model,
                completion_guard=completion_guard)
            details["role_probabilities"] = trajectory.role_probabilities[-1]
            details["success_prediction"] = trajectory.success_predictions[-1]
        else:
            env = InvestigationEnv(scenario, budget)
            obs = env.reset(seed=start)
            if control is not None:
                control.reset()
            action_ids, public_trace = [], []
            while True:
                public_trace.append(obs.as_dict())
                action = control.act(obs) if control is not None else model.act(obs, epsilon=0)
                action_ids.append(action)
                result = env.step(action)
                if result.terminated:
                    break
                obs = result.observation
            details = {"outcome": result.outcome, "reason": result.reason,
                "original_utility": result.reference_reward, "action_ids": action_ids,
                "role_label": env.training_labels()["permutation"],
                "public_trace": public_trace,
                "events": [event.__dict__ for event in env.transcript()]}
        details["episode_seed"] = start
        if index >= 2:
            details.pop("public_trace", None)
        rows.append(details)
    budget.save()
    return {"status": "completed" if len(rows) == episodes else "budget_limited",
        "scenario": scenario.as_dict(), "requested_episodes": episodes, "greedy": greedy,
        "population": "frozen_learned" if trader_model is not None else scenario.family,
        "completion_guard": completion_guard,
        "metrics": summarize(rows), "episodes": rows}


def evaluate_checkpoint(path, budget, episodes=32, families=None, frozen_traders=None, greedy=True):
    scenario, config, model, trader, state = load_run(path)
    families = families or ("conventional", "noisy", "imitative", "strategic")
    results = []
    for family in families:
        result = evaluate(replace(scenario, family=family), budget, episodes, config.seed+100000,
            model, config, greedy=greedy)
        result["label"] = family
        results.append(result)
        if config.role_weight > 0 and not isinstance(model, TabularLearner):
            guarded = evaluate(replace(scenario, family=family), budget, episodes, config.seed+100000,
                model, config, greedy=greedy, completion_guard=True)
            guarded["label"] = family+"_guard"
            results.append(guarded)
    if trader is not None or frozen_traders is not None:
        if frozen_traders is not None:
            _, trader_config, _, trader, _ = load_run(frozen_traders)
            if trader is None:
                raise ValueError("requested checkpoint has no learned traders")
            if trader_config.exact_resources != config.exact_resources:
                raise ValueError("frozen trader encoder must match the evaluation feature contract")
        result = evaluate(scenario, budget, episodes, config.seed+100000, model, config,
                          trader_model=trader, greedy=greedy)
        result["label"] = "frozen_learned"
        results.append(result)
    return {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "evaluation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_seed": config.seed, "training_episodes": state["completed"],
            "algorithm": config.algorithm, "memory": config.memory, "role_weight": config.role_weight,
            "success_weight": config.success_weight, "results": results}


def crossplay(checkpoints, budget, episodes, output):
    entries = [load_run(path) for path in checkpoints]
    cells = []
    for i, (scenario, config, model, _, _) in enumerate(entries):
        for j, (_, other_config, _, trader, _) in enumerate(entries):
            if trader is None or other_config.exact_resources != config.exact_resources:
                raise ValueError("crossplay needs joint checkpoints with matching feature contracts")
            result = evaluate(scenario, budget, episodes, 9000000+i*100+j,
                              model, config, trader_model=trader)
            cells.append({"investigator": i, "traders": j, "result": result})
    report = {"checkpoints": [str(p) for p in checkpoints], "cells": cells,
              "dependence": "cells share both policy axes; not independent training seeds"}
    write_json(output, report)
    return report
