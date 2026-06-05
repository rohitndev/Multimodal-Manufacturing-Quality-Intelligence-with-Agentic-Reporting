"""Grad-CAM visualization helper.

Produces a coloured heatmap overlay on the input image highlighting the
defect region. Uses bounding-box based attention as a deterministic
fallback when no PyTorch model is available.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def gradcam_overlay(
    image: np.ndarray,
    bboxes: Optional[List[List[int]]] = None,
    alpha: float = 0.45,
) -> np.ndarray:
    """Generate a Grad-CAM-style overlay on the image.

    For bounding boxes, fills a Gaussian heatmap centred on each box. Returns
    a uint8 BGR image suitable for cv2.imwrite.
    """
    import cv2

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    if bboxes:
        for x1, y1, x2, y2 in bboxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            sigma_x = max((x2 - x1) / 3.0, 5)
            sigma_y = max((y2 - y1) / 3.0, 5)
            ys, xs = np.mgrid[0:h, 0:w]
            heat += np.exp(
                -(((xs - cx) ** 2) / (2 * sigma_x ** 2) + ((ys - cy) ** 2) / (2 * sigma_y ** 2))
            )
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_u8 = (heat * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, cmap, alpha, 0)

    # Draw bounding boxes on top for clarity
    if bboxes:
        for x1, y1, x2, y2 in bboxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return overlay
