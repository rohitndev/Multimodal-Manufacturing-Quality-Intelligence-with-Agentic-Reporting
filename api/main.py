"""FastAPI inspection endpoint.

Routes:
    GET  /health
    POST /inspect              (multipart image upload, optional product field)
    POST /feedback             (operator correction)
    GET  /retrain/status       (active-learning trigger status)
    POST /specs/ingest         (re-index spec PDFs)
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src import __version__
from src.active_learning import FeedbackStore, OperatorFeedback, RetrainTrigger
from src.agent import ERPClient, ISO9001ReportGenerator, QualityAgent
from src.rag import SpecIngestion, SpecRetriever
from src.vision import VisionPipeline

from .schemas import (
    FeedbackRequest,
    HealthResponse,
    InspectionResponse,
    RetrainStatusResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

OVERLAY_DIR = Path(os.getenv("OVERLAY_DIR", "data/overlays"))
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Multimodal Manufacturing Quality Intelligence with Agentic Reporting",
    description="YOLOv8 + ViT + RAG + LangGraph quality-inspection backend.",
    version=__version__,
)


class _Deps:
    """Lazy-initialised singletons."""

    vision: Optional[VisionPipeline] = None
    retriever: Optional[SpecRetriever] = None
    agent: Optional[QualityAgent] = None
    feedback: Optional[FeedbackStore] = None
    retrain: Optional[RetrainTrigger] = None


def _vision() -> VisionPipeline:
    if _Deps.vision is None:
        _Deps.vision = VisionPipeline()
    return _Deps.vision


def _retriever() -> SpecRetriever:
    if _Deps.retriever is None:
        _Deps.retriever = SpecRetriever()
    return _Deps.retriever


def _agent() -> QualityAgent:
    if _Deps.agent is None:
        _Deps.agent = QualityAgent(
            retriever=_retriever(),
            report_generator=ISO9001ReportGenerator(),
            erp_client=ERPClient(),
        )
    return _Deps.agent


def _feedback() -> FeedbackStore:
    if _Deps.feedback is None:
        _Deps.feedback = FeedbackStore()
    return _Deps.feedback


def _retrain() -> RetrainTrigger:
    if _Deps.retrain is None:
        _Deps.retrain = RetrainTrigger(store=_feedback())
    return _Deps.retrain


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        components={
            "vision": "ready",
            "rag": "ready",
            "agent": "ready",
            "feedback": "ready",
        },
    )


@app.post("/inspect", response_model=InspectionResponse)
async def inspect(
    image: UploadFile = File(...),
    product: str = Form("Generic Component"),
) -> InspectionResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    inspection_id = f"INSP-{uuid.uuid4().hex[:10].upper()}"
    overlay_path = str(OVERLAY_DIR / f"{inspection_id}.png")

    vision_result = _vision().run(frame, image_id=inspection_id, overlay_output=overlay_path)
    findings = [f.to_dict() for f in vision_result.findings]

    decision = _agent().run(
        inspection_id=inspection_id,
        findings=findings,
        product=product,
    )

    return InspectionResponse(
        inspection_id=inspection_id,
        product=product,
        findings=findings,
        decision=decision.decision,
        rationale=decision.rationale,
        report=decision.report,
        erp_response=decision.erp_response,
        overlay_path=vision_result.overlay_path,
    )


@app.get("/inspect/overlay/{inspection_id}")
def get_overlay(inspection_id: str) -> FileResponse:
    path = OVERLAY_DIR / f"{inspection_id}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Overlay not found.")
    return FileResponse(path, media_type="image/png")


@app.post("/feedback")
def submit_feedback(body: FeedbackRequest) -> JSONResponse:
    fb = OperatorFeedback(
        inspection_id=body.inspection_id,
        defect_id=body.defect_id,
        corrected_defect_type=body.corrected_defect_type,
        corrected_severity=body.corrected_severity,
        operator_id=body.operator_id,
        notes=body.notes,
    )
    _feedback().add(fb)
    return JSONResponse({"status": "stored", "feedback": fb.to_dict()})


@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status() -> RetrainStatusResponse:
    plan = _retrain().build_plan()
    return RetrainStatusResponse(**plan)


@app.post("/specs/ingest")
def specs_ingest(specs_dir: str = "data/sample_specs") -> JSONResponse:
    ingestion = SpecIngestion(specs_dir=specs_dir)
    n = ingestion.ingest()
    # Reset retriever cache so it picks up the new corpus
    _Deps.retriever = None
    return JSONResponse({"status": "ingested", "documents": n})
