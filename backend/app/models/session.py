from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.app.models.constraints import Constraints
from backend.app.models.partspec import PartSpec


SessionStatus = Literal["draft", "concept", "approved", "final"]


class SessionRecord(BaseModel):
    id: str = Field(..., min_length=6)
    created_at: str
    updated_at: str
    status: SessionStatus = "draft"

    name: Optional[str] = None

    current_revision: Optional[int] = None
    approved_revision: Optional[int] = None
    final_revision: Optional[int] = None

    # Session-level constraints (enforced on approval/finalize)
    constraints: Constraints = Field(default_factory=Constraints)

    metadata: Dict[str, Any] = Field(default_factory=dict)


RevisionSource = Literal["spec", "text", "finalize", "ai_change", "ai_clarify"]


class RevisionRecord(BaseModel):
    session_id: str
    rev: int = Field(..., ge=1)
    created_at: str
    source: RevisionSource

    note: Optional[str] = None
    input_description: Optional[str] = None
    input_sketch_asset: Optional[str] = None
    change_request: Optional[str] = None

    stage: Literal["concept", "final"] = "concept"

    spec: PartSpec
    validation: Dict[str, Any]
    artifacts: Dict[str, Any]  # {run_id, urls, reports, ...}

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ListRevisionsResponse(BaseModel):
    session: SessionRecord
    revisions: List[RevisionRecord]
