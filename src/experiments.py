"""a manifest enumerates independent seeds before any result is inspected."""

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .budget import WorkBudget
from .config import Scenario, TrainConfig, from_dict
from .evaluation import evaluate_checkpoint
from .io import source_hashes, write_json
from .training import train

LOCAL_LIMIT = 95000


def plan(manifest, only=None):
    data = json.loads(Path(manifest).read_text())
    allowed = {"name", "description", "seeds", "scenario", "train", "runs", "evaluation"}
    if set(data)-allowed:
        raise ValueError("unknown manifest keys")
    if not data["runs"] or len(data["seeds"]) != len(set(data["seeds"])):
        raise ValueError("nonempty runs and unique independent seeds required")
    jobs, names = [], set()
    for run in data["runs"]:
        if set(run)-{"name", "scenario", "train"}:
            raise ValueError("unknown run keys")
        if run["name"] in names or not run["name"].replace("_", "").replace("-", "").isalnum():
            raise ValueError("run names must be unique safe identifiers")
        names.add(run["name"])
        for seed in data["seeds"]:
            scenario = from_dict(Scenario, {**data.get("scenario", {}), **run.get("scenario", {})})
            config = from_dict(TrainConfig, {**data.get("train", {}), **run.get("train", {}), "seed": seed})
            jobs.append({"name": f"{run['name']}/seed_{seed}", "scenario": scenario.as_dict(),
                         "train": config.as_dict(), "evaluation": data.get("evaluation", {})})
    if only:
        if set(only)-names:
            raise ValueError("requested configuration is not in the manifest")
        jobs = [job for job in jobs if job["name"].split("/")[0] in only]
    return {"manifest": str(manifest), "name": data["name"], "jobs": jobs,
            "training_episodes": sum(job["train"]["episodes"] for job in jobs),
            "independent_seeds_per_configuration": len(data["seeds"])}


def run_job(arguments):
    job, output, budget_path, limit, resume = arguments
    budget = WorkBudget(limit, Path(budget_path))
    scenario, config = from_dict(Scenario, job["scenario"]), from_dict(TrainConfig, job["train"])
    folder = Path(output)/job["name"]
    cached_train, cached_eval = folder/"training.json", folder/"evaluation.json"
    if resume and cached_train.exists() and cached_eval.exists():
        old, evaluated = json.loads(cached_train.read_text()), json.loads(cached_eval.read_text())
        wanted = job["evaluation"]
        families = wanted.get("families", ["conventional", "noisy", "imitative", "strategic"])
        labels = set(families)
        if config.role_weight > 0:
            labels.update(family+"_guard" for family in families)
        if config.joint:
            labels.add("frozen_learned")
        if (old["status"] == "completed" and old["config"] == config.as_dict()
            and old["scenario"] == scenario.as_dict() and old["source_sha256"] == source_hashes()
            and evaluated["evaluation_source_sha256"] == hashlib.sha256(Path(__file__).with_name("evaluation.py").read_bytes()).hexdigest()
            and {r["label"] for r in evaluated["results"]} == labels
            and all(r["status"] == "completed" and r["requested_episodes"] == wanted.get("episodes", config.eval_episodes)
                    and r["greedy"] == wanted.get("greedy", True) for r in evaluated["results"])):
            return {"name": job["name"], "status": "completed", "cached": True,
                    "completed_episodes": old["completed_episodes"], "evaluation_complete": True}
    report = train(scenario, config, folder, budget, resume)
    evaluation = job["evaluation"]
    results = evaluate_checkpoint(folder/"checkpoint.pt", budget,
        evaluation.get("episodes", config.eval_episodes), evaluation.get("families"),
        greedy=evaluation.get("greedy", True))
    write_json(folder/"evaluation.json", results)
    budget.save()
    return {"name": job["name"], "status": report["status"],
            "completed_episodes": report["completed_episodes"],
            "evaluation_complete": all(r["status"] == "completed" for r in results["results"])}


def run_manifest(manifest, output, budget_path, cloud=False, workers=1, resume=False, only=None):
    planned = plan(manifest, only)
    if not cloud and (workers != 1 or any(job["train"]["episodes"] > 1000 for job in planned["jobs"])):
        raise ValueError("local profile uses one worker and small jobs; large manifests use cloud")
    arguments = []
    for job in planned["jobs"]:
        ledger = Path(output)/job["name"]/"work.json" if cloud else Path(budget_path)
        arguments.append((job, str(output), str(ledger), None if cloud else LOCAL_LIMIT, resume))
    write_json(Path(output)/"plan.json", planned)
    if workers == 1:
        results = [run_job(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_job, arguments))
    report = {"profile": "cloud" if cloud else "local", "results": results}
    write_json(Path(output)/"manifest_results.json", report)
    return report
