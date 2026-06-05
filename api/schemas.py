"""Pydantic request/response schemas for the inspection API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    inspection_id: str
    defect_id: str
    corrected_defect_type: Optional[str] = None
    corrected_severity: Optional[str] = None
    operator_id: str = "anonymous"
    notes: str = ""


class FindingSchema(BaseModel):
    defect_id: str
    defect_type: str
    confidence: float
    bbox: List[int]
    severity: str
    severity_score: float
    severity_probs: Dict[str, float]


class InspectionResponse(BaseModel):
    inspection_id: str
    product: str
    findings: List[FindingSchema] = Field(default_factory=list)
    decision: str
    rationale: str
    report: Dict[str, Any]
    erp_response: Dict[str, Any]
    overlay_path: Optional[str] = None


class RetrainStatusResponse(BaseModel):
    trigger_at: str
    total_corrections: int
    per_class: Dict[str, int]
    should_retrain: bool
    min_corrections: int
    next_steps: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]
