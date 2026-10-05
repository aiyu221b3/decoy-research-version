"""checkpoint boundaries preserve optimizer and sampling state across resumes."""

import csv
import json
import time
from pathlib import Path

import torch

from .config import Scenario, TrainConfig, from_dict
from .encoding import encode_public, encode_trader, tabular_state
from .environment import InvestigationEnv
from .io import configure_cpu, fingerprint, runtime, save_checkpoint, source_hashes, write_json
from .learning import ppo_update, td_update
from .models import ActorCritic
from .rollouts import collect, episode_seed
from .tabular import TabularLearner
from .types import TraderView


def build_models(scenario, config, budget):
    env = InvestigationEnv(scenario, budget)
    obs = env.reset(seed=episode_seed(config.seed, 99, 0))
    features = len(encode_public(obs))
    model = ActorCritic(features, hidden=config.hidden, memory=config.memory)
    trader = None
    if config.joint:
        dimension = len(encode_trader(TraderView(obs, 0, 0, 0.0)))
        trader = ActorCritic(dimension, actions=27, hidden=config.hidden, memory=config.memory)
    return model, trader


def load_run(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "decoy_checkpoint_v2" or state["source_sha256"] != source_hashes():
        raise ValueError("checkpoint requires its recorded scientific source revision")
    scenario = from_dict(Scenario, state["scenario"])
    config = from_dict(TrainConfig, state["config"])
    configure_cpu(config.seed)
    if config.algorithm in ("q_learning", "monte_carlo"):
        learner = TabularLearner(config)
        learner.load_state_dict(state["tabular"])
        return scenario, config, learner, None, state
    model = ActorCritic(state["features"], hidden=config.hidden, memory=config.memory)
    model.load_state_dict(state["model"])
    model.eval()
    trader = None
    if state["trader_model"] is not None:
        trader = ActorCritic(state["trader_features"], actions=27, hidden=config.hidden, memory=config.memory)
        trader.load_state_dict(state["trader_model"])
        trader.eval()
    return scenario, config, model, trader, state


def train(scenario, config, output, budget, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure_cpu(config.seed)
    started, work_start = time.monotonic(), budget.used
    identity_config = {k: v for k, v in config.as_dict().items() if k not in ("episodes", "eval_episodes", "checkpoint_every")}
    identity = fingerprint({"scenario": scenario.as_dict(), "learner": identity_config})
    hashes = source_hashes()
    checkpoint = output / "checkpoint.pt"
    tabular = config.algorithm in ("q_learning", "monte_carlo")
    model, trader = (None, None) if tabular else build_models(scenario, config, budget)
    learner = TabularLearner(config) if tabular else None
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) if config.algorithm == "ppo" else None
    trader_optimizer = torch.optim.Adam(trader.parameters(), lr=config.learning_rate) if trader is not None else None
    gate = torch.tensor(0.0, requires_grad=config.audit_gate) if config.algorithm == "abps_td" else None
    completed, outer_completed, inner_actions, inner_market, updates = 0, 0, 0, 0, 0
    history, cumulative_successes, previous_work = [], 0, 0
    if checkpoint.exists():
        if not resume:
            raise FileExistsError("run exists; use resume or a new output directory")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["identity"] != identity or state["source_sha256"] != hashes:
            raise ValueError("resume requires the same mechanics, learner and source revision")
        completed, outer_completed, updates = state["completed"], state["outer_completed"], state["updates"]
        inner_actions, inner_market = state["inner_actions"], state["inner_market"]
        history, cumulative_successes = state["history"], state["successes"]
        previous_work = state["total_counted_work"]
        if completed > config.episodes:
            raise ValueError("resume target is below saved episode count")
        if tabular:
            learner.load_state_dict(state["tabular"])
        else:
            model.load_state_dict(state["model"])
            if optimizer is not None:
                optimizer.load_state_dict(state["optimizer"])
            if trader is not None:
                trader.load_state_dict(state["trader_model"])
                trader_optimizer.load_state_dict(state["trader_optimizer"])
            if gate is not None:
                with torch.no_grad():
                    gate.copy_(state["gate"])
        torch.set_rng_state(state["torch_rng"])

    def persist():
        state = {"format": "decoy_checkpoint_v2", "identity": identity, "scenario": scenario.as_dict(),
            "config": config.as_dict(), "source_sha256": hashes, "completed": completed,
            "outer_completed": outer_completed, "updates": updates, "inner_actions": inner_actions,
            "inner_market": inner_market, "history": history, "successes": cumulative_successes,
            "total_counted_work": previous_work+budget.used-work_start,
            "torch_rng": torch.get_rng_state(), "model": model.state_dict() if model else None,
            "features": model.features if model else None,
            "trader_model": trader.state_dict() if trader else None,
            "trader_features": trader.features if trader else None,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "trader_optimizer": trader_optimizer.state_dict() if trader_optimizer else None,
            "gate": gate.detach() if gate is not None else None,
            "tabular": learner.state_dict() if learner else None}
        save_checkpoint(checkpoint, state)
        budget.save()

    max_work = InvestigationEnv(scenario).max_episode_work
    while completed < config.episodes:
        until_checkpoint = config.checkpoint_every-(completed % config.checkpoint_every)
        count = min(config.batch_episodes, config.episodes-completed, until_checkpoint)
        outer = config.outer_episodes if gate is not None and config.audit_gate else 0
        if not budget.can_reserve((count+outer)*max_work):
            break
        trajectories, trader_trajectories, batch_successes = [], [], 0
        for index in range(count):
            seed = episode_seed(config.seed, 0, completed+index)
            if tabular:
                env = InvestigationEnv(scenario, budget)
                obs, transitions = env.reset(seed=seed), []
                while True:
                    action = learner.act(obs, epsilon=config.epsilon)
                    result = env.step(action)
                    transitions.append((tabular_state(obs, config.exact_resources), action, result.reward,
                        tabular_state(result.observation, config.exact_resources), result.observation.action_mask,
                        result.terminated))
                    if result.terminated:
                        break
                    obs = result.observation
                learner.update(transitions)
                batch_successes += int(result.outcome == "fraudster_caught")
                inner_actions, inner_market = inner_actions+env.actions, inner_market+env.tick
            else:
                trajectory, others, details = collect(scenario, budget, model, seed,
                    config.exact_resources, trader_model=trader)
                trajectories.append(trajectory)
                trader_trajectories.extend(record for record in others if record.actions)
                batch_successes += int(trajectory.successful)
                inner_actions += details["actions"]
                inner_market += details["market_updates"]
        diagnostics = {}
        if config.algorithm == "ppo":
            diagnostics = ppo_update(model, optimizer, trajectories, config)
            if trader is not None:
                diagnostics["trader_loss"] = ppo_update(trader, trader_optimizer, trader_trajectories, config, trader=True)["loss"]
        elif not tabular:
            diagnostics = td_update(model, trajectories, config, gate, scenario, budget, outer_completed)
            outer_completed += outer
        completed += count
        updates += 1
        cumulative_successes += batch_successes
        history.append({"episode": completed, "inner_actions": inner_actions, "market_updates": inner_market,
            "outer_episodes": outer_completed, "batch_success": batch_successes/count,
            "cumulative_success": cumulative_successes/completed,
            "counted_work": previous_work+budget.used-work_start, **diagnostics})
        if completed % config.checkpoint_every == 0:
            persist()
            save_checkpoint(output / f"checkpoint_{completed:09d}.pt", torch.load(checkpoint, weights_only=True))
    persist()
    with (output / "learning_curve.csv").open("w", newline="") as handle:
        columns = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    report = {"status": "completed" if completed == config.episodes else "budget_limited",
        "scenario": scenario.as_dict(), "config": config.as_dict(), "identity": identity,
        "source_sha256": hashes, "runtime": runtime(), "completed_episodes": completed,
        "outer_episodes": outer_completed, "inner_actions": inner_actions, "market_updates": inner_market,
        "work_this_invocation": budget.used-work_start, "shared_budget": budget.as_dict(),
        "total_counted_work": previous_work+budget.used-work_start,
        "elapsed_seconds": time.monotonic()-started,
        "investigator_parameters": sum(p.numel() for p in model.parameters()) if model else 0,
        "shared_trader_parameters": sum(p.numel() for p in trader.parameters()) if trader else 0,
        "strategic_agents": 4, "background_learners": 0, "history": history}
    write_json(output / "training.json", report)
    return report
