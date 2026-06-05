"""Operator feedback store.

Captures corrections from operators (via API or Label Studio export) into a
single JSONL store. Downstream ``RetrainTrigger`` consumes this to schedule
fine-tuning runs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class OperatorFeedback:
    inspection_id: str
    defect_id: str
    corrected_defect_type: Optional[str] = None
    corrected_severity: Optional[str] = None
    operator_id: str = "anonymous"
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> dict:
        return asdict(self)


class FeedbackStore:
    def __init__(self, path: str = "data/feedback.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, feedback: OperatorFeedback) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(feedback.to_dict()) + "\n")

    def all(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def count(self) -> int:
        return len(self.all())
