"""Retraining trigger.

Decides whether the YOLOv8 model should be re-trained based on accumulated
operator feedback. Returns a structured plan that an external scheduler
(GitHub Actions, Airflow, Kubernetes Job) can execute.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .feedback import FeedbackStore

logger = logging.getLogger(__name__)


class RetrainTrigger:
    def __init__(
        self,
        store: Optional[FeedbackStore] = None,
        min_corrections: int = 25,
    ):
        self.store = store or FeedbackStore()
        self.min_corrections = min_corrections

    def should_retrain(self) -> bool:
        return self.store.count() >= self.min_corrections

    def build_plan(self) -> Dict:
        feedback = self.store.all()
        per_class: Dict[str, int] = {}
        for f in feedback:
            cls = f.get("corrected_defect_type") or "unknown"
            per_class[cls] = per_class.get(cls, 0) + 1
        return {
            "trigger_at": datetime.utcnow().isoformat() + "Z",
            "total_corrections": len(feedback),
            "per_class": per_class,
            "should_retrain": self.should_retrain(),
            "min_corrections": self.min_corrections,
            "next_steps": [
                "Export Label Studio corrections to YOLO format.",
                "Mix with base dataset (MVTec + NEU).",
                "Fine-tune YOLOv8 for 50 epochs with MLflow tracking.",
                "Evaluate mAP@0.5 — promote to champion if Δ ≥ +1.0pp.",
            ],
        }
