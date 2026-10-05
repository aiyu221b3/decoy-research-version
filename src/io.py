"""atomic artifacts and fingerprints preserve run-to-configuration lineage."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    core = {"actions.py", "budget.py", "config.py", "encoding.py", "environment.py", "language.py",
            "learning.py", "models.py", "populations.py", "rollouts.py", "tabular.py", "training.py", "io.py"}
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py")) if path.name in core}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_checkpoint(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def runtime():
    import platform
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}


def configure_cpu(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
