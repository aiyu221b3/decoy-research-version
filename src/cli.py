"""small local checks and large remote jobs use distinct execution profiles."""

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .budget import WorkBudget
from .config import Scenario
from .evaluation import crossplay, evaluate, evaluate_checkpoint
from .experiments import LOCAL_LIMIT, plan, run_manifest
from .io import write_json
from .policies import ControlPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--only", nargs="+")
        if name == "run":
            p.add_argument("--profile", choices=("local", "cloud"), default="local")
            p.add_argument("--workers", type=int, default=1)
            p.add_argument("--resume", action="store_true")
            p.add_argument("--budget-file", type=Path, default=Path("runs/local_work.json"))
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--frozen-traders", type=Path)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--profile", choices=("local", "cloud"), default="local")
    p.add_argument("--budget-file", type=Path, default=Path("runs/local_work.json"))
    p = sub.add_parser("controls")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-file", type=Path, default=Path("runs/local_work.json"))
    p = sub.add_parser("crossplay")
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("local", "cloud"), default="local")
    p.add_argument("--budget-file", type=Path, default=Path("runs/local_work.json"))
    for name in ("analyze", "plot"):
        p = sub.add_parser(name)
        p.add_argument("--runs", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        report = plan(args.manifest, args.only)
        write_json(args.output, report)
        print(json.dumps({"jobs": len(report["jobs"]), "training_episodes": report["training_episodes"]}))
    elif args.command == "run":
        if args.workers < 1:
            parser.error("positive worker count required")
        report = run_manifest(args.manifest, args.output, args.budget_file,
                              args.profile == "cloud", args.workers, args.resume, args.only)
        print(json.dumps(report, indent=2))
    elif args.command in ("evaluate", "crossplay"):
        if args.episodes < 1:
            parser.error("positive evaluation episodes required")
        budget = WorkBudget(None if args.profile == "cloud" else LOCAL_LIMIT, args.budget_file)
        if args.command == "evaluate":
            report = evaluate_checkpoint(args.checkpoint, budget, args.episodes,
                                         frozen_traders=args.frozen_traders, greedy=not args.stochastic)
            write_json(args.output, report)
        else:
            crossplay(args.checkpoints, budget, args.episodes, args.output)
    elif args.command == "controls":
        if args.episodes < 1:
            parser.error("positive episodes required")
        if args.output.exists():
            cached = json.loads(args.output.read_text())
            if len(cached["results"]) == 32 and all(r["status"] == "completed" and r["requested_episodes"] == args.episodes for r in cached["results"]):
                print(json.dumps({"status": "cached_completed_controls", "cells": 32}))
                return
            raise FileExistsError("choose a new controls output for a different or incomplete experiment")
        budget = WorkBudget(LOCAL_LIMIT, args.budget_file)
        results = []
        for objective in ("success", "original"):
            for family in ("conventional", "noisy", "imitative", "strategic"):
                for name in ("random", "immediate", "reserve", "evidence"):
                    scenario = Scenario(family=family, objective=objective)
                    result = evaluate(scenario, budget, args.episodes, 104,
                        control=ControlPolicy(name, 104, objective))
                    result["control"] = name
                    results.append(result)
        write_json(args.output, {"results": results, "budget": budget.as_dict()})
        print(json.dumps({"cells": len(results), "episodes": sum(r["metrics"]["episodes"] for r in results),
                          "budget": budget.as_dict()}, indent=2))
    elif args.command == "analyze":
        report = analyze(args.runs, args.output)
        print(json.dumps({"rows": len(report["rows"]), "summaries": len(report["summaries"])}))
    else:
        from .plotting import make_figures
        report = make_figures(args.runs, args.output)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
