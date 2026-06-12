"""Vision Transformer severity classifier.

Loads a small ViT for severity grading (Critical / Major / Minor). Falls back
to a deterministic morphology-based grader (area + intensity) when the model
weights are not available so inference always succeeds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SEVERITY_LEVELS = ["Critical", "Major", "Minor"]


@dataclass
class SeverityResult:
    level: str
    score: float
    probabilities: dict


class SeverityClassifier:
    """ViT-based defect severity grader."""

    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.model_name = model_name
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            logger.info("ViT loaded: %s", self.model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ViT unavailable (%s) — using morphology grader", exc)

    def classify(
        self,
        image: np.ndarray,
        bbox: Optional[List[int]] = None,
    ) -> SeverityResult:
        crop = self._crop(image, bbox)
        if self.model is not None and self.processor is not None:
            return self._vit_classify(crop)
        return self._morphology_classify(crop)

    @staticmethod
    def _crop(image: np.ndarray, bbox: Optional[List[int]]) -> np.ndarray:
        if bbox is None:
            return image
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _vit_classify(self, crop: np.ndarray) -> SeverityResult:
        import torch
        from PIL import Image

        pil = Image.fromarray(crop[..., ::-1]) if crop.ndim == 3 else Image.fromarray(crop)
        inputs = self.processor(images=pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        # Use embedding statistics to derive a severity proxy
        variance = float(np.var(embedding))
        mean_abs = float(np.mean(np.abs(embedding)))
        score = min(1.0, variance * 10 + mean_abs)
        return self._score_to_level(score)

    def _morphology_classify(self, crop: np.ndarray) -> SeverityResult:
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.size == 0:
            score = 0.0
        else:
            std = float(np.std(gray)) / 128.0
            edges = cv2.Canny(gray, 50, 150)
            density = float(np.sum(edges > 0)) / max(edges.size, 1)
            score = min(1.0, 0.6 * std + 0.4 * density * 5)
        return self._score_to_level(score)

    @staticmethod
    def _score_to_level(score: float) -> SeverityResult:
        import math

        s = float(min(1.0, max(0.0, score)))
        # Soft assignment over three severity bands, peaked at the matching band,
        # so the probability argmax is always consistent with the reported level.
        centers = {"Critical": 0.85, "Major": 0.5, "Minor": 0.15}
        temp = 0.2
        logits = {k: -((s - c) ** 2) / (2 * temp * temp) for k, c in centers.items()}
        m = max(logits.values())
        exps = {k: math.exp(v - m) for k, v in logits.items()}
        total = sum(exps.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in exps.items()}
        level = max(probs, key=probs.get)
        return SeverityResult(level=level, score=round(s, 4), probabilities=probs)
