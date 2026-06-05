"""MLflow helpers for YOLOv8 fine-tune tracking."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Optional


@contextmanager
def mlflow_run(experiment: str = "ml-quality-inspection", run_name: Optional[str] = None):
    """Context manager that gracefully degrades if MLflow isn't installed."""
    try:
        import mlflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception:  # noqa: BLE001
        yield None


def log_yolo_metrics(metrics: Dict[str, float]) -> None:
    try:
        import mlflow
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
    except Exception:  # noqa: BLE001
        pass


def log_yolo_artifacts(weights_path: str) -> None:
    try:
        import mlflow
        mlflow.log_artifact(weights_path, artifact_path="weights")
    except Exception:  # noqa: BLE001
        pass
