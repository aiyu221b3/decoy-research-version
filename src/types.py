"""public snapshots and training-only labels have different types."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Event:
    sequence: int
    tick: int
    speaker: int
    kind: str
    target: int | None
    value: float
    text: str


@dataclass(frozen=True)
class Evidence:
    market: float
    volume: float
    inspected: bool
    mean: float
    trend: float
    record_score: float
    record_count: int
    pending: bool
    pressure: float
    speech_count: int
    defend_rate: float
    accuse_rate: float
    support_rate: float


@dataclass(frozen=True)
class Observation:
    traders: tuple[Evidence, ...]
    events: tuple[Event, ...]
    attention: int
    initial_attention: int
    tick: int
    horizon: int
    focus: int
    audits: int
    communication: int
    credibility: float
    accusation: int | None
    macro: float
    liquidity: float
    news: float
    action_mask: tuple[bool, ...]

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TraderView:
    public: Observation
    identity: int
    own_role: int
    own_latent: float


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float
    reference_reward: float
    terminated: bool
    truncated: bool
    outcome: str | None
    reason: str | None
    trader_rewards: tuple[float, ...]
