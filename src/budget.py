"""counted work includes opponents and background processes, not just probes."""

import json
from collections import Counter
from pathlib import Path


class BudgetExhausted(RuntimeError):
    pass


class WorkBudget:
    def __init__(self, limit: int | None = 85000, path: Path | None = None):
        self.limit, self.path = limit, path
        self.counts = Counter()
        if path is not None and path.exists():
            self.counts.update(json.loads(path.read_text())["counts"])
        if limit is not None and self.used > limit:
            raise BudgetExhausted("saved usage already exceeds the requested ceiling")

    @property
    def used(self):
        return sum(self.counts.values())

    def can_reserve(self, count):
        return self.limit is None or self.used + count <= self.limit

    def charge(self, category, count=1):
        if not self.can_reserve(count):
            raise BudgetExhausted("shared counted-work ceiling reached")
        self.counts[category] += count

    def save(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
            temporary.replace(self.path)

    def as_dict(self):
        return {"counts": dict(self.counts), "used": self.used, "limit": self.limit,
                "definition": "investigator actions, trader decisions, market updates, background process updates"}
