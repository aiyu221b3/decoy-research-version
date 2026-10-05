"""templates render shared semantic events without revealing hidden labels."""

TEMPLATES = {
    "neutral": "I have no further comment about the current case.",
    "defend": "I dispute the concern about trader {target}; review the records.",
    "explain": "Trader {target}'s activity includes ordinary market fluctuations.",
    "provide": "I can provide supporting records for trader {target}.",
    "delay": "The records for trader {target} need more time to reconcile.",
    "accuse": "The Investigator should review trader {target}'s activity.",
    "support": "Trader {target}'s explanation deserves consideration.",
    "question": "Trader {target}, explain this activity and identify supporting records.",
    "warn": "Trader {target}, further irregular activity will receive closer scrutiny.",
    "publish": "I am publishing the available evidence summary for trader {target}.",
    "audit": "A records check for trader {target} has been requested.",
    "record": "Records review for trader {target} reports discrepancy {value:.2f}.",
    "witness": "A background observer reports trader {target}'s activity score {value:.2f}.",
    "vote": "The Investigator closes the case against trader {target}.",
}


def render(kind, target, value=0.0):
    return TEMPLATES[kind].format(target=target, value=value)
