"""End-to-end vision pipeline.

Detector → severity → Grad-CAM overlay. Returns a structured payload that
downstream RAG and agent layers consume.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

from .detector import DefectDetector, Detection
from .gradcam import gradcam_overlay
from .severity import SeverityClassifier, SeverityResult

logger = logging.getLogger(__name__)


@dataclass
class DefectFinding:
    defect_id: str
    defect_type: str
    confidence: float
    bbox: List[int]
    severity: str
    severity_score: float
    severity_probs: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InspectionResult:
    image_id: str
    findings: List[DefectFinding] = field(default_factory=list)
    overlay_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "findings": [f.to_dict() for f in self.findings],
            "overlay_path": self.overlay_path,
        }


class VisionPipeline:
    def __init__(
        self,
        detector: Optional[DefectDetector] = None,
        severity: Optional[SeverityClassifier] = None,
    ):
        self.detector = detector or DefectDetector()
        self.severity = severity or SeverityClassifier()

    def run(
        self,
        image: np.ndarray,
        image_id: str = "frame",
        overlay_output: Optional[str] = None,
    ) -> InspectionResult:
        detections: List[Detection] = self.detector.detect(image)
        findings: List[DefectFinding] = []
        for idx, det in enumerate(detections):
            sev: SeverityResult = self.severity.classify(image, det.bbox)
            findings.append(
                DefectFinding(
                    defect_id=f"{image_id}-D{idx + 1:03d}",
                    defect_type=det.defect_type,
                    confidence=round(det.confidence, 4),
                    bbox=det.bbox,
                    severity=sev.level,
                    severity_score=sev.score,
                    severity_probs=sev.probabilities,
                )
            )

        overlay_path = None
        if overlay_output and findings:
            import cv2
            overlay = gradcam_overlay(image, [f.bbox for f in findings])
            cv2.imwrite(overlay_output, overlay)
            overlay_path = overlay_output

        return InspectionResult(image_id=image_id, findings=findings, overlay_path=overlay_path)
