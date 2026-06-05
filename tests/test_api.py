import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app


def _png_bytes(shape=(480, 640, 3)) -> bytes:
    img = np.full(shape, 200, dtype=np.uint8)
    img[230:250, 100:540] = 30
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_inspect_endpoint():
    with TestClient(app) as client:
        files = {"image": ("test.png", io.BytesIO(_png_bytes()), "image/png")}
        data = {"product": "Metal Panel"}
        r = client.post("/inspect", files=files, data=data)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["inspection_id"].startswith("INSP-")
        assert body["decision"] in {"PASS", "REWORK", "QUARANTINE", "FAIL"}
        assert "rationale" in body


def test_feedback_and_retrain_status(tmp_path):
    with TestClient(app) as client:
        fb = {
            "inspection_id": "INSP-X",
            "defect_id": "D001",
            "corrected_defect_type": "crack",
            "corrected_severity": "Critical",
            "operator_id": "tester",
            "notes": "edge case",
        }
        r = client.post("/feedback", json=fb)
        assert r.status_code == 200
        r = client.get("/retrain/status")
        assert r.status_code == 200
        body = r.json()
        assert "should_retrain" in body
