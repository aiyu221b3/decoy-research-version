"""stable ids distinguish legal support from behavior probabilities."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class Action:
    kind: str
    target: int | None
    intent: str | None = None


INV_ACTIONS = tuple(Action(kind, i) for kind in ("observe", "accuse", "vote", "focus") for i in range(3)) + (
    *tuple(Action("audit", i) for i in range(3)),
    *tuple(Action("statement", i, intent) for intent in ("question", "warn", "publish") for i in range(3)),
    Action("wait", None),
)


def decode(action_id: int) -> Action:
    if isinstance(action_id, bool) or not isinstance(action_id, Integral):
        raise TypeError("action id must be an integer")
    if not 0 <= action_id < len(INV_ACTIONS):
        raise ValueError("action id out of range")
    return INV_ACTIONS[int(action_id)]


def encode(action: Action) -> int:
    return INV_ACTIONS.index(action)


@dataclass(frozen=True)
class TraderAction:
    market: str
    speech: str
    relative_target: int | None


TRADER_ACTIONS = tuple(TraderAction(market, speech, target)
    for market in ("normal", "conceal", "signal")
    for speech, target in (("neutral", None), ("defend", 0), ("explain", 0),
                          ("provide", 0), ("delay", 0), ("accuse", 1),
                          ("accuse", 2), ("support", 1), ("support", 2)))


def trader_target(identity: int, relative: int | None):
    return None if relative is None else (identity + relative) % 3
