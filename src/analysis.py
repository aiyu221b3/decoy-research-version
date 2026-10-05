"""seed-level summaries avoid treating evaluation episodes as training replicates."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import write_json


def analyze(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, groups = [], defaultdict(list)
    for path in sorted(root.rglob("evaluation.json")):
        data = json.loads(path.read_text())
        config = path.parent.parent.name
        for result in data["results"]:
            row = {"configuration": config, "seed": data["training_seed"],
                   "population": result["label"], "complete": result["status"] == "completed",
                   **{key: value for key, value in result["metrics"].items()
                      if isinstance(value, (int, float)) or value is None}}
            rows.append(row)
            if row["complete"]:
                groups[(config, result["label"])].append(row)
    summaries = []
    rng = np.random.default_rng(7345498)
    for (config, population), records in groups.items():
        seeds = [r["seed"] for r in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seed records must be resolved before aggregation")
        values = np.array([r["success"] for r in records])
        interval = None
        if len(values) >= 2:
            bootstrap = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
            interval = np.quantile(bootstrap, (0.025, 0.975)).tolist()
        summaries.append({"configuration": config, "population": population,
            "training_seeds": seeds, "mean_success": float(values.mean()),
            "seed_bootstrap_95": interval,
            "evidence_level": "single-seed pilot" if len(values) == 1 else "independent-seed summary"})
    comparisons = []
    by_configuration = {(config, family): {r["seed"]: r for r in records} for (config, family), records in groups.items()}
    for (config, family), candidate in by_configuration.items():
        baseline = by_configuration.get(("ppo_gru", family))
        if baseline is None or config == "ppo_gru":
            continue
        seeds = sorted(set(candidate) & set(baseline))
        if not seeds:
            continue
        difference = np.array([candidate[seed]["success"]-baseline[seed]["success"] for seed in seeds])
        interval = None if len(seeds) < 2 else np.quantile(
            rng.choice(difference, size=(4000, len(seeds)), replace=True).mean(axis=1), (0.025, 0.975)).tolist()
        comparisons.append({"candidate": config, "baseline": "ppo_gru", "population": family,
            "paired_training_seeds": seeds, "mean_success_difference": float(difference.mean()),
            "paired_seed_bootstrap_95": interval,
            "evidence_level": "pilot" if len(seeds) < 10 else "ten-or-more-seed descriptive contrast"})
    report = {"rows": rows, "summaries": summaries, "paired_comparisons": comparisons,
        "interpretation": "single-seed pilots do not establish algorithm superiority; incomplete cells excluded"}
    write_json(output/"summary.json", report)
    with (output/"metrics.csv").open("w", newline="") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return report
