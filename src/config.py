"""mechanics and learner choices are separately versioned."""

from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import TypeVar

T = TypeVar("T")


def from_dict(cls: type[T], values: dict) -> T:
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class Scenario:
    version: str = "rich_v2"
    family: str = "noisy"
    attention: int = 10
    horizon: int = 18
    focus: int = 2
    audits: int = 2
    communication: int = 3
    objective: str = "success"
    statements: bool = True
    records: bool = True
    background: bool = True
    noise: float = 0.2
    witnesses: int = 2
    record_delay: int = 2

    def __post_init__(self):
        if self.version != "rich_v2":
            raise ValueError("this standalone engine implements rich_v2")
        if self.family not in ("conventional", "noisy", "imitative", "strategic"):
            raise ValueError("unknown opponent family")
        if self.objective not in ("success", "original"):
            raise ValueError("unknown objective")
        for name in ("attention", "horizon", "focus", "audits", "communication", "witnesses", "record_delay"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.attention < 1 or self.horizon < 2 or self.witnesses > 2:
            raise ValueError("positive attention, horizon >=2, at most two witness processes")
        if not 0 <= self.noise <= 1:
            raise ValueError("noise must be a probability")
        if self.record_delay < 1 or self.record_delay > 4:
            raise ValueError("record delay must be one to four market ticks")
        for name in ("statements", "records", "background"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    algorithm: str = "ppo"
    memory: str = "gru"
    hidden: int = 32
    seed: int = 71
    episodes: int = 128
    batch_episodes: int = 16
    epochs: int = 3
    learning_rate: float = 0.0007
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.02
    role_weight: float = 0.0
    success_weight: float = 0.0
    exact_resources: bool = True
    joint: bool = False
    checkpoint_every: int = 10000
    eval_episodes: int = 32
    shaping_alpha: float = 0.2
    gate_lr: float = 0.05
    outer_episodes: int = 2
    audit_gate: bool = True
    fixed_gate: float = 1.0
    epsilon: float = 0.2

    def __post_init__(self):
        for name in ("learning_rate", "gamma", "gae_lambda", "clip", "entropy", "role_weight",
                     "success_weight", "shaping_alpha", "gate_lr", "fixed_gate", "epsilon"):
            if not isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.algorithm not in ("ppo", "a2c_td", "abps_td", "q_learning", "monte_carlo"):
            raise ValueError("unknown algorithm")
        if self.memory not in ("ff", "gru") or self.hidden < 8 or self.hidden > 128:
            raise ValueError("memory must be ff/gru with hidden size 8..128")
        for name in ("episodes", "batch_episodes", "epochs", "checkpoint_every", "eval_episodes", "outer_episodes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1 or not 0 < self.clip < 1:
            raise ValueError("invalid discount, trace or clip")
        if self.learning_rate <= 0 or min(self.entropy, self.role_weight, self.success_weight, self.shaping_alpha, self.gate_lr) < 0:
            raise ValueError("learning rate positive and loss weights nonnegative")
        if self.joint and self.algorithm != "ppo":
            raise ValueError("joint training is implemented with ppo only")
        if self.algorithm == "abps_td" and self.role_weight:
            raise ValueError("audited branch keeps its inner supervision fixed to continuation prediction")
        if not 0 <= self.fixed_gate <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("fixed gate and epsilon must be probabilities")
        for name in ("exact_resources", "joint", "audit_gate"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def as_dict(self):
        return asdict(self)
