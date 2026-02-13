from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CreateRevisionFromSpecRequest(BaseModel):
    mode: Literal["from_spec"] = "from_spec"
    spec: Dict[str, Any]
    note: Optional[str] = None


class CreateRevisionFromTextRequest(BaseModel):
    mode: Literal["from_text"] = "from_text"
    description: str = Field(..., min_length=1)
    sketch_asset_id: Optional[str] = None
    note: Optional[str] = None

    # Optional OpenAI params (local/self-host use)
    csv_path: str = "user_api.csv"
    model: str = "gpt-4.1"


CreateRevisionRequest = Annotated[Union[CreateRevisionFromSpecRequest, CreateRevisionFromTextRequest], Field(discriminator="mode")]


class ApproveRevisionResponse(BaseModel):
    session_id: str
    approved_revision: int


class FinalizeSessionRequest(BaseModel):
    # Reduce grid_pitch by this factor for final (smaller => higher resolution).
    # If it would exceed max_voxels, the system auto-adjusts to a safe pitch.
    target_pitch_factor: float = Field(0.5, gt=0.0, lt=1.0)
    note: Optional[str] = None


class FinalizeSessionResponse(BaseModel):
    session_id: str
    final_revision: int


class ApplyChangeRequest(BaseModel):
    change_request: str = Field(..., min_length=1)
    note: Optional[str] = None

    # Optional OpenAI params (local/self-host use)
    csv_path: str = "user_api.csv"
    model: str = "gpt-4.1"


class ClarificationAnswer(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class AnswerClarificationsRequest(BaseModel):
    answers: list[ClarificationAnswer] = Field(..., min_length=1)
    note: Optional[str] = None

    # Optional OpenAI params (local/self-host use)
    csv_path: str = "user_api.csv"
    model: str = "gpt-4.1"
