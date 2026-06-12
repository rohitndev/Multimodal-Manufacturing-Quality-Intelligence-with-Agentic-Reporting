"""YOLOv8 defect detection module.

Wraps Ultralytics YOLOv8 for defect bounding-box detection. Falls back to a
heuristic OpenCV-based detector if Ultralytics is not installed, so the
pipeline always returns a valid result.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFECT_CLASSES = ["scratch", "crack", "dent", "void", "stain", "burr"]


@dataclass
class Detection:
    defect_type: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]

    def to_dict(self) -> dict:
        return asdict(self)


class DefectDetector:
    """YOLOv8 wrapper for surface defect detection."""

    def __init__(self, weights: Optional[str] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.weights = weights or os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights)
            logger.info("YOLOv8 model loaded: %s", self.weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLOv8 unavailable (%s) — using heuristic detector", exc)
            self.model = None

    def detect(self, image: np.ndarray) -> List[Detection]:
        if self.model is not None:
            return self._yolo_detect(image)
        return self._heuristic_detect(image)

    def _yolo_detect(self, image: np.ndarray) -> List[Detection]:
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                name = r.names.get(cls_idx, f"class_{cls_idx}")
                if name not in DEFECT_CLASSES:
                    name = DEFECT_CLASSES[cls_idx % len(DEFECT_CLASSES)]
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                detections.append(
                    Detection(
                        defect_type=name,
                        confidence=float(box.conf[0]),
                        bbox=xyxy,
                    )
                )
        return detections

    def _heuristic_detect(self, image: np.ndarray) -> List[Detection]:
        """Edge / contour based fallback so the pipeline always runs."""
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape[:2]
        frame_mean = float(blurred.mean())
        min_area = (h * w) * 0.0005
        detections: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)
            extent = area / max(bw * bh, 1)  # fraction of the bbox the blob fills
            if extent < 0.15 or aspect >= 4 or aspect <= 0.25:
                defect = "scratch"           # thin, elongated line
            elif 0.6 <= aspect <= 1.6 and extent >= 0.5:
                defect = "dent"              # compact, round, filled
            else:
                defect = "stain"             # filled irregular blob
            roi = blurred[y:y + bh, x:x + bw]
            roi_edges = edges[y:y + bh, x:x + bw]
            if roi.size:
                darkness = (frame_mean - float(roi.min())) / max(frame_mean, 1.0)
                contrast = abs(float(roi.mean()) - frame_mean) / 128.0
                edge_density = float((roi_edges > 0).sum()) / max(roi_edges.size, 1)
            else:
                darkness = contrast = edge_density = 0.0
            confidence = float(np.clip(0.6 + 0.2 * darkness + 0.35 * contrast + 0.6 * edge_density, 0.6, 0.95))
            detections.append(
                Detection(
                    defect_type=defect,
                    confidence=round(confidence, 4),
                    bbox=[x, y, x + bw, y + bh],
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:5]
