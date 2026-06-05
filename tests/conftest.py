import numpy as np
import pytest


@pytest.fixture
def synthetic_image():
    """640x480 BGR image with a high-contrast streak (simulated scratch)."""
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    img[230:250, 100:540] = 30  # horizontal dark streak
    img[100:140, 300:340] = 60  # small dark blob
    return img
