import ast
import base64
import hashlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont

from .actions import INV_ACTIONS
from .io import write_json

COLORS = {"fraudster_caught": "#14a38b", "innocent_caught": "#df6561", "trickster_caught": "#d9ad45", "timeout": "#9aa7bb"}


def make_figures(runs, output):
    runs, output = Path(runs), Path(output)
    root = Path(__file__).resolve().parents[2]
    font = root/"assets/fonts/InterVariable.ttf"
    if not font.is_file():
        font = Path(sys.prefix)/"share/decoy-research/fonts/InterVariable.ttf"
    if not font.is_file():
        raise FileNotFoundError("the bundled Inter font is required for figure reproducibility")
    font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
        "svg.fonttype": "none", "figure.facecolor": "#ffffff", "axes.facecolor": "#fafbfe"})
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"font_family": family, "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
                "figures": [], "data_sha256": {}}

    def save(fig, name, kind, explanation):
        fig.tight_layout()
        files = []
        for extension in ("svg", "pdf", "png"):
            path = output/f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            if extension == "svg":
                tree = ET.parse(path)
                namespace = "http://www.w3.org/2000/svg"
                text = "".join(tree.getroot().itertext())
                instance = TTFont(font)
                options = subset.Options()
                options.name_IDs += [13, 14]
                sub = subset.Subsetter(options=options)
                sub.populate(text=text)
                sub.subset(instance)
                buffer = io.BytesIO()
                instance.save(buffer)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                definitions = tree.getroot().find(f"{{{namespace}}}defs")
                style = ET.SubElement(definitions, f"{{{namespace}}}style", {"type": "text/css"})
                style.text = f"@font-face {{font-family: '{family}'; src: url(data:font/ttf;base64,{encoded}) format('truetype');}}"
                tree.write(path, encoding="utf-8", xml_declaration=True)
            files.append(str(path))
        manifest["figures"].append({"name": name, "type": kind, "explanation": explanation, "files": files})
        plt.close(fig)

    def box(ax, x, y, text, color="#e4ebfb", width=2.3, height=0.7):
        from matplotlib.patches import FancyBboxPatch
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.08", fc=color, ec="#7890b5")
        ax.add_patch(patch)
        ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=10)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set(xlim=(0, 11), ylim=(0, 6))
    ax.axis("off")
    box(ax, 4.3, 4.8, "Investigator\npublic features → FF / GRU32", width=2.5)
    for i in range(3):
        box(ax, 1+i*3.3, 2.8, f"Trader {i}\nown role + public evidence\nindependent memory", "#e1f4ec", width=2.6, height=1)
        ax.annotate("", (5.5, 4.7), (2.3+i*3.3, 3.9), arrowprops={"arrowstyle": "<->", "color": "#45617d"})
    box(ax, 0.5, 0.4, "Macro · liquidity · witnesses\nrule-based, no trained brain", "#fff1d9", width=3.5, height=0.9)
    box(ax, 4.5, 0.4, "Audit records · event templates\ndelayed / noisy evidence", "#fff1d9", width=3.5, height=0.9)
    ax.set_title("Exactly four strategic agents; background factors are processes", pad=18)
    save(fig, "01_four_agent_architecture", "schematic", "information permissions and four-agent scope")

    modules = sorted(path.stem for path in Path(__file__).parent.glob("*.py") if path.stem != "__init__")
    edges = set()
    for path in Path(__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level and node.module in modules:
                edges.add((path.stem, node.module))
    fig, ax = plt.subplots(figsize=(13, 7))
    left = [name for name in modules if name in ("cli", "experiments", "training", "evaluation", "analysis", "plotting", "rollouts", "learning")]
    right = [name for name in modules if name not in left]
    positions = {}
    for col, names in enumerate((left, right)):
        for i, name in enumerate(names):
            positions[name] = (col*6, i)
            ax.text(col*6, i, name, ha="center", va="center", bbox={"boxstyle": "round", "fc": "#e4ebfb", "ec": "#97a8c1"})
    for source, target in edges:
        if source in positions and target in positions:
            ax.annotate("", positions[target], positions[source], arrowprops={"arrowstyle": "->", "alpha": 0.15, "color": "#526784"}, zorder=0)
    ax.set(xlim=(-1.7, 7.7), ylim=(-1, max(len(left), len(right))))
    ax.axis("off")
    ax.set_title("Code dependency map (generated from Python imports)")
    save(fig, "02_module_dependencies", "code graph", "arrows point from an importing module to its dependency")
    (output/"module_dependencies.dot").write_text("digraph decoy {\n"+"\n".join(f'  "{a}" -> "{b}";' for a, b in sorted(edges))+"\n}\n")

    labels = [f"{i:02d}  {a.kind}{' / '+a.intent if a.intent else ''}  {a.target if a.target is not None else '—'}" for i, a in enumerate(INV_ACTIONS)]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(np.arange(25), [0 if a.kind == "focus" else 1 for a in INV_ACTIONS], color=["#c194e2" if a.kind == "statement" else "#69a5d3" for a in INV_ACTIONS])
    ax.set_yticks(np.arange(25), labels)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.2), xlabel="Attention units (focus uses a separate charge)", title="Stable action catalog: old IDs retained, new IDs appended")
    save(fig, "03_action_catalog", "mechanics diagram", "masks additionally enforce focus/audit/communication/evidence requirements")

    fig, ax = plt.subplots(figsize=(10, 4))
    grid = np.array([[int(b >= 2 and h >= 3)+int(b >= 3 and h >= 4) for h in range(2, 19)] for b in range(1, 11)])
    image = ax.imshow(grid, origin="lower", cmap="YlGnBu", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(17), range(2, 19))
    ax.set_yticks(range(10), range(1, 11))
    ax.set(xlabel="Market horizon including initial tick", ylabel="Attention", title="Resolution feasibility: 0 neither, 1 immediate, 2 after one probe")
    fig.colorbar(image, ax=ax, ticks=(0, 1, 2))
    save(fig, "04_completion_feasibility", "derived mechanics", "algebraic costs; not an empirical performance plot")

    training, evaluations = {}, {}
    for path in sorted(runs.rglob("training.json")):
        name = path.parent.parent.name+" / "+path.parent.name
        training[name] = json.loads(path.read_text())
        manifest["data_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(runs.rglob("evaluation.json")):
        name = path.parent.parent.name+" / "+path.parent.name
        evaluations[name] = json.loads(path.read_text())
        manifest["data_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    grouped_training, grouped_evaluation = {}, {}
    for name, value in training.items():
        grouped_training.setdefault(name.split(" / ")[0], []).append(value)
    for name, value in evaluations.items():
        grouped_evaluation.setdefault(name.split(" / ")[0], []).append(value)
    training = {}
    for name, records in grouped_training.items():
        seeds = [record["config"]["seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError("plot one run root; duplicate configuration/seed artifacts cannot be pooled")
        entry = dict(records[0])
        if len(records) > 1:
            histories = [record["history"] for record in records if record["history"]]
            start = max(history[0]["counted_work"] for history in histories)
            end = min(history[-1]["counted_work"] for history in histories)
            grid = np.linspace(start, end, 100)
            curves = np.array([np.interp(grid, [p["counted_work"] for p in history],
                [p["cumulative_success"] for p in history]) for history in histories])
            entry["history"] = [{"counted_work": float(x), "cumulative_success": float(y)}
                                for x, y in zip(grid, curves.mean(axis=0))]
        training[name] = entry
    evaluations = {}
    for name, records in grouped_evaluation.items():
        seeds = [record["training_seed"] for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate evaluation seed records")
        entry, combined = dict(records[0]), []
        labels = sorted({r["label"] for record in records for r in record["results"]})
        for label in labels:
            cells = [r for record in records for r in record["results"]
                     if r["label"] == label and r["status"] == "completed" and r["metrics"]["episodes"]]
            if not cells:
                continue
            result, metric = dict(cells[0]), dict(cells[0]["metrics"])
            for key in ("success", "timeout", "innocent_error", "trickster_error"):
                metric[key] = float(np.mean([r["metrics"][key] for r in cells]))
            metric["episodes"] = sum(r["metrics"]["episodes"] for r in cells)
            metric["outcomes"] = dict(sum((Counter(r["metrics"]["outcomes"]) for r in cells), Counter()))
            metric["action_counts"] = dict(sum((Counter(r["metrics"]["action_counts"]) for r in cells), Counter()))
            metric["outcome_rates"] = {outcome: float(np.mean([r["metrics"]["outcomes"].get(outcome, 0)/r["metrics"]["episodes"] for r in cells])) for outcome in COLORS}
            result["metrics"], result["episodes"] = metric, [e for r in cells for e in r["episodes"]]
            combined.append(result)
        entry["results"] = combined
        evaluations[name] = entry
    independent = max((len(records) for records in grouped_training.values()), default=1)
    manifest["independent_training_seeds"] = independent
    level = "single-seed pilot" if independent == 1 else "independent-seed means"
    if training:
        fig, ax = plt.subplots(figsize=(11, 5))
        for name, data in training.items():
            ax.plot([p["counted_work"] for p in data["history"]], [p["cumulative_success"] for p in data["history"]], label=name.split(" / ")[0], marker=".")
        ax.set(xlabel="Counted training work including opponents/background/outer", ylabel="Cumulative training success", ylim=(0, 1), title=f"Learning curves: {level}; not held-out performance")
        ax.legend(fontsize=8, ncol=3)
        save(fig, "05_pilot_learning_curves", "learning diagnostic", "training outcomes; seed means use only a shared reached work grid")
        fig, ax = plt.subplots(figsize=(11, 4))
        names = list(training)
        ax.bar(range(len(names)), [training[n]["investigator_parameters"] for n in names], color="#709cd2")
        ax.set_xticks(range(len(names)), [n.split(" / ")[0] for n in names], rotation=25, ha="right")
        ax.set(ylabel="Investigator trainable parameters", title="Small models; FF56 and GRU32 are approximately capacity-matched")
        save(fig, "06_model_capacity", "implementation measurement", "tabular learners have zero neural parameters, not zero stored Q-values")

    if evaluations:
        names = list(evaluations)
        families = sorted({result["label"] for value in evaluations.values() for result in value["results"]})
        success, timeout = np.full((len(names), len(families)), np.nan), np.full((len(names), len(families)), np.nan)
        for i, name in enumerate(names):
            for result in evaluations[name]["results"]:
                j = families.index(result["label"])
                if result["metrics"]["episodes"]:
                    success[i, j], timeout[i, j] = result["metrics"]["success"], result["metrics"]["timeout"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 7))
        for ax, data, title in zip(axes, (success, timeout), ("Correct completion", "Timeout")):
            image = ax.imshow(data, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(families)), families, rotation=20)
            ax.set_yticks(range(len(names)), [n.split(" / ")[0] for n in names])
            ax.set_title(title)
            for i in range(len(names)):
                for j in range(len(families)):
                    if np.isfinite(data[i, j]):
                        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", color="white" if data[i,j] > .55 else "#18354f")
        fig.colorbar(image, ax=axes[-1], fraction=.046)
        fig.suptitle(f"Greedy evaluation: {level}; intervals and contrasts are in the analysis tables")
        save(fig, "07_policy_transfer", "pilot result", "families are reactive held-out recipes; frozen_learned is a separate population")
        fig, ax = plt.subplots(figsize=(12, 5))
        bottom = np.zeros(len(names))
        for outcome, color in COLORS.items():
            values = []
            for name in names:
                result = next(r for r in evaluations[name]["results"] if r["label"] == "noisy")
                n = result["metrics"]["episodes"]
                values.append(result["metrics"]["outcome_rates"].get(outcome, 0) if n else 0)
            ax.bar(range(len(names)), values, bottom=bottom, label=outcome.replace("_", " "), color=color)
            bottom += values
        ax.set_xticks(range(len(names)), [n.split(" / ")[0] for n in names], rotation=25, ha="right")
        ax.set(ylabel="Episode fraction", ylim=(0, 1), title="All outcomes reported together — noisy-family pilot")
        ax.legend(ncol=2, fontsize=8)
        save(fig, "08_pilot_outcomes", "pilot result", "conditional accuracy alone hides timeout and false catches")
        groups = ("observe", "focus", "audit", "statement", "accuse", "vote", "wait")
        usage = np.zeros((len(names), len(groups)))
        for i, name in enumerate(names):
            for result in evaluations[name]["results"]:
                for key, count in result["metrics"].get("action_counts", {}).items():
                    usage[i, groups.index(INV_ACTIONS[int(key)].kind)] += count
        usage /= np.maximum(usage.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(11, 6))
        image = ax.imshow(usage, vmin=0, vmax=1, cmap="PuBu", aspect="auto")
        ax.set_xticks(range(len(groups)), groups)
        ax.set_yticks(range(len(names)), [n.split(" / ")[0] for n in names])
        ax.set_title("Action distribution: a diverse action set does not guarantee diverse learned use")
        fig.colorbar(image, ax=ax)
        save(fig, "09_action_distribution", "pilot diagnostic", "counts pooled across evaluation families for visualization, not independent-seed inference")
        supervised = next((d for d in evaluations.values() if d["role_weight"] > 0), None)
        if supervised:
            episodes = [e for r in supervised["results"] if not r["label"].endswith("_guard") for e in r["episodes"]]
            matrix = np.zeros((6, 6))
            for e in episodes:
                matrix[e["role_label"], int(np.argmax(e["role_probabilities"]))] += 1
            fig, ax = plt.subplots(figsize=(7, 5))
            image = ax.imshow(matrix, cmap="Blues")
            ax.set(xlabel="Predicted permutation", ylabel="True permutation", title="Auxiliary role head — pilot confusion, including timeouts")
            fig.colorbar(image, ax=ax)
            save(fig, "10_role_permutation_confusion", "pilot diagnostic", "six assignments; supervision is training-only and must be matched")
            predictions = np.array([e["success_prediction"] for e in episodes])
            outcomes = np.array([e["outcome"] == "fraudster_caught" for e in episodes])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot((0, 1), (0, 1), "--", color="#acb5c4")
            for low in np.arange(0, 1, .2):
                mask = (predictions >= low) & (predictions < low+.2+1e-8)
                if mask.any():
                    ax.scatter(predictions[mask].mean(), outcomes[mask].mean(), s=40+mask.sum()*3, color="#6797d0")
                    ax.annotate(f"n={mask.sum()}", (predictions[mask].mean(), outcomes[mask].mean()), xytext=(5, 5), textcoords="offset points")
            ax.set(xlim=(0, 1), ylim=(-.03, 1.03), xlabel="Predicted continuation success", ylabel="Observed success", title="Coarse pilot calibration; sparse bins do not establish reliability")
            save(fig, "11_success_calibration", "pilot diagnostic", "stochastic-training continuation head under greedy policy/family shift; not a risk guarantee")
        example = next(iter(evaluations.values()))["results"][0]["episodes"][0]
        trace = example.get("public_trace", [])
        if trace:
            fig, axes = plt.subplots(2, 1, figsize=(11, 7))
            ticks = [obs["tick"] for obs in trace]
            for i in range(3):
                axes[0].plot(ticks, [o["traders"][i]["market"] for o in trace], marker="o", label=f"Trader {i}")
            axes[0].plot(ticks, [o["macro"]+o["news"] for o in trace], "--", label="Public common factor")
            axes[0].legend(ncol=4, fontsize=8)
            axes[0].set(ylabel="Public market signal", title="Example public trajectory: common shocks and individual activity")
            axes[1].step(ticks, [o["attention"] for o in trace], label="Attention", where="post")
            axes[1].step(ticks, [o["communication"] for o in trace], label="Communication tokens", where="post")
            axes[1].set(xlabel="Market update counter", ylabel="Remaining resources")
            axes[1].legend()
            save(fig, "12_public_episode_trace", "example trajectory", f"source episode seed {example['episode_seed']}; no latent-role curve is drawn")
            transcript = "# Public episode transcript\n\n"+f"Seed: {example['episode_seed']}. Outcome: {example['outcome']}.\n\n"
            transcript += "\n".join(f"- tick {event['tick']}, speaker {event['speaker']}: {event['text']}" for event in example["events"])
            (output/"example_transcript.md").write_text(transcript+"\n")

    controls = runs/"controls.json"
    if controls.is_file():
        data = json.loads(controls.read_text())
        manifest["data_sha256"][str(controls)] = hashlib.sha256(controls.read_bytes()).hexdigest()
        labels, completions, utility = [], [], []
        for objective in ("success", "original"):
            for name in ("random", "immediate", "reserve", "evidence"):
                cells = [r for r in data["results"] if r["control"] == name and r["scenario"]["objective"] == objective and r["metrics"]["episodes"]]
                labels.append(name+" / "+objective)
                completions.append(np.mean([r["metrics"]["success"] for r in cells]) if cells else 0)
                utility.append(np.mean([r["metrics"]["original_utility"] for r in cells]) if cells else 0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, values, title in zip(axes, (completions, utility), ("Correct completion", "Original outcome utility")):
            ax.barh(labels, values, color="#719bd1")
            ax.set_title(title)
        fig.suptitle("Simple controls: objective and completion are different quantities")
        save(fig, "13_control_objectives", "pilot result", "unweighted mean of four family rates, not pooled training-seed evidence")

    audited = next((data for name, data in training.items() if name.startswith("audited_td")), None)
    if audited:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        history = audited["history"]
        axes[0].plot([h["episode"] for h in history], [h["gate_after"] for h in history], marker="o")
        axes[0].set(xlabel="Inner episodes", ylabel="Sigmoid trust", title="Finite-TD auditor gate (tiny pilot)")
        axes[1].plot([h["episode"] for h in history], [h["gate_gradient"] for h in history], marker="o")
        axes[1].set(xlabel="Inner episodes", ylabel="Outer gate gradient", title="Includes explicitly recorded gate regularization")
        save(fig, "14_auditor_gradient_trace", "pilot diagnostic", "passing a meta-gradient does not establish improvement")
    initial = sorted(runs.glob("initial_*.json"))
    if initial:
        fig, ax = plt.subplots(figsize=(8, 5))
        for path in initial:
            data = json.loads(path.read_text())
            name = data["configuration"]
            before = data["result"]["metrics"]["success"]
            after = next(r["metrics"]["success"] for r in evaluations[name]["results"] if r["label"] == "noisy")
            ax.plot((0, data["comparison_training_episodes"]), (before, after), marker="o", label=name)
            manifest["data_sha256"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        ax.set(ylim=(-.03, 1.03), xlabel="Training episodes", ylabel="Greedy correct-completion fraction",
               title="Seeded initialization versus short training (12 paired episode seeds)")
        ax.legend()
        save(fig, "15_initial_vs_trained", "pilot diagnostic", "same architecture seed/test starts; one training seed and tiny evaluation sample")
    write_json(output/"manifest.json", manifest)
    return {"font": family, "figures": len(manifest["figures"]), "formats": ["svg", "pdf", "png"]}
