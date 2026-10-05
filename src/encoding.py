"""feature construction never requests evaluation or supervision labels."""

import numpy as np

from .actions import INV_ACTIONS
from .language import TEMPLATES

EVENT_KINDS = tuple(TEMPLATES)


def one_hot(value, size):
    return [float(value == i) for i in range(size)]


def encode_public(obs, previous=-1, exact=True):
    values = []
    for evidence in obs.traders:
        values.extend((evidence.market, evidence.volume, float(evidence.inspected), evidence.mean,
            evidence.trend, evidence.record_score, min(evidence.record_count, 4)/4,
            float(evidence.pending), evidence.pressure, min(evidence.speech_count, 60)/60,
            evidence.defend_rate, evidence.accuse_rate, evidence.support_rate))
    attention = obs.attention if exact else min(obs.attention, 3)
    tick = obs.tick if exact else min(obs.tick//5, 3)
    values.extend((attention/10, obs.initial_attention/10, tick/20, obs.horizon/20,
        obs.focus/2, obs.audits/2, obs.communication/3, obs.credibility, obs.macro,
        obs.liquidity, obs.news))
    values.extend(one_hot(obs.accusation, 3))
    events = obs.events[-6:]
    for index in range(6):
        if index < len(events):
            event = events[index]
            values.extend(one_hot(event.speaker, 6))
            values.extend(one_hot(EVENT_KINDS.index(event.kind), len(EVENT_KINDS)))
            values.extend(one_hot(event.target, 3))
            values.extend((event.value, min(obs.tick-event.tick, 20)/20, 1.0))
        else:
            values.extend([0.0] * (6+len(EVENT_KINDS)+3+3))
    values.extend(one_hot(previous, len(INV_ACTIONS)))
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite public evidence")
    return result


def encode_trader(view, previous=-1, exact=True, phase="market"):
    return np.concatenate((encode_public(view.public, -1, exact),
        np.asarray(one_hot(view.identity, 3)+one_hot(view.own_role, 3)+[view.own_latent], dtype=np.float32),
        np.asarray(one_hot(previous, 27)+[float(phase == "response")], dtype=np.float32)))


def tabular_state(obs, exact=True):
    state = []
    for item in obs.traders:
        residual = item.market-obs.macro-obs.news
        state.extend((int(np.digitize(residual, (-0.2, 0.2, 0.6))),
                      int(item.record_count > 0), int(np.digitize(item.record_score, (0.2, 0.6))),
                      int(item.inspected)))
    state.extend((obs.attention if exact else min(obs.attention, 3),
                  obs.tick if exact else min(obs.tick//5, 3), obs.focus, obs.audits,
                  obs.communication, -1 if obs.accusation is None else obs.accusation))
    return tuple(state)
