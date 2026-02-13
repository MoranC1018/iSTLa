from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import ARTIFACTS_DIR
from backend.app.models.constraints import ConstraintsPatch
from backend.app.models.partspec import PartSpec
from backend.app.models.api import (
    CreateSessionRequest,
    CreateRevisionRequest,
    FinalizeSessionRequest,
    ApplyChangeRequest,
    AnswerClarificationsRequest,
)
from backend.app.models.session import SessionRecord, RevisionRecord
from backend.app.services.validation_engine import validate_partspec
from backend.app.services.session_store import SessionStore, generate_artifacts_for_spec


app = FastAPI(title="AI STL Builder MVP", version="0.4.2")

# Serve a tiny no-build web UI (vanilla HTML/JS)
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

store = SessionStore()


@app.get("/")
def root() -> FileResponse:
    return FileResponse((STATIC_DIR / "index.html").as_posix())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# ----------------------------
# Stateless (spec -> artifacts)
# ----------------------------

@app.post("/validate_spec")
def validate_spec(spec: PartSpec) -> Dict[str, Any]:
    return validate_partspec(spec)


@app.post("/generate_artifacts")
def generate_artifacts(spec: PartSpec) -> Dict[str, Any]:
    validation = validate_partspec(spec)
    if validation.get("errors"):
        raise HTTPException(status_code=400, detail={"message": "Spec validation failed", **validation})

    try:
        artifact = generate_artifacts_for_spec(spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact generation failed: {e}")

    return artifact.payload


@app.get("/artifacts/{run_id}/{filename}")
def get_artifact(run_id: str, filename: str) -> FileResponse:
    path = ARTIFACTS_DIR / run_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path)


# ----------------------------
# Sessions + revisions (MVP)
# ----------------------------

@app.post("/sessions", response_model=SessionRecord)
def create_session(req: CreateSessionRequest) -> SessionRecord:
    return store.create_session(name=req.name, metadata=req.metadata)


@app.get("/sessions", response_model=List[SessionRecord])
def list_sessions() -> List[SessionRecord]:
    sessions: List[SessionRecord] = []
    for p in sorted((store.base_dir).glob("*")):
        if not p.is_dir():
            continue
        try:
            sessions.append(store.get_session(p.name))
        except Exception:
            continue
    return sessions


@app.get("/sessions/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    try:
        return store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/sessions/{session_id}/constraints")
def get_constraints(session_id: str) -> Dict[str, Any]:
    """Get session constraints + their hash."""
    try:
        s = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "constraints": s.constraints.model_dump(exclude_none=True),
        "constraints_hash": store.constraints_hash(session_id),
    }


@app.put("/sessions/{session_id}/constraints")
def update_constraints(session_id: str, patch: ConstraintsPatch) -> Dict[str, Any]:
    """Update session constraints (partial update).

    Changing constraints invalidates approvals/finals.
    """
    try:
        s = store.update_constraints(session_id, patch)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "session": s.model_dump(),
        "constraints_hash": store.constraints_hash(session_id),
    }


@app.get("/sessions/{session_id}/revisions", response_model=List[RevisionRecord])
def list_revisions(session_id: str) -> List[RevisionRecord]:
    try:
        _ = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    return store.list_revisions(session_id)


@app.get("/sessions/{session_id}/revisions/{rev}", response_model=RevisionRecord)
def get_revision(session_id: str, rev: int) -> RevisionRecord:
    try:
        _ = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return store.get_revision(session_id, rev)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Revision not found")


@app.post("/sessions/{session_id}/revisions", response_model=RevisionRecord)
def create_revision(session_id: str, req: CreateRevisionRequest) -> RevisionRecord:
    try:
        session = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.mode == "from_spec":
        try:
            spec = PartSpec.model_validate(req.spec)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid PartSpec: {e}")

        try:
            return store.create_revision_from_spec(
                session_id,
                spec=spec,
                source="spec",
                note=req.note,
                input_description=None,
                input_sketch_asset=None,
                stage="concept",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if req.mode == "from_text":
        description = req.description.strip()
        if not description:
            raise HTTPException(status_code=400, detail="Missing description")

        schema_path = Path(__file__).resolve().parent / "schema" / "partspec_v1.schema.json"
        partspec_schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))

        # Optional: if a sketch is available, include it as an image input.
        sketch_path: Path | None = None
        if req.sketch_asset_id:
            asset_id = req.sketch_asset_id
            if "/" in asset_id or "\\" in asset_id:
                raise HTTPException(status_code=400, detail="Invalid sketch_asset_id")
            candidate = store.uploads_dir(session_id) / asset_id
            if not candidate.exists():
                raise HTTPException(status_code=404, detail="Sketch asset not found")
            sketch_path = candidate

        try:
            from backend.app.services.openai_orchestrator import (
                generate_partspec_from_text,
                generate_partspec_from_text_and_sketch,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI integration not available: {e}")

        try:
            if sketch_path is not None:
                spec = generate_partspec_from_text_and_sketch(
                    description,
                    sketch_path,
                    csv_path=req.csv_path,
                    model=req.model,
                    partspec_schema=partspec_schema,
                    constraints=session.constraints,
                )
            else:
                spec = generate_partspec_from_text(
                    description,
                    csv_path=req.csv_path,
                    model=req.model,
                    partspec_schema=partspec_schema,
                    constraints=session.constraints,
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

        try:
            return store.create_revision_from_spec(
                session_id,
                spec=spec,
                source="text",
                note=req.note,
                input_description=description,
                input_sketch_asset=req.sketch_asset_id,
                stage="concept",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=400, detail=f"Unknown mode: {getattr(req, 'mode', None)}")


@app.post("/sessions/{session_id}/revisions/{rev}/approve", response_model=SessionRecord)
def approve_revision(session_id: str, rev: int) -> SessionRecord:
    try:
        return store.approve_revision(session_id, rev)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session or revision not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sessions/{session_id}/finalize", response_model=RevisionRecord)
def finalize_session(session_id: str, req: FinalizeSessionRequest) -> RevisionRecord:
    try:
        return store.finalize_session(
            session_id,
            target_pitch_factor=req.target_pitch_factor,
            note=req.note,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sessions/{session_id}/revisions/{rev}/change", response_model=RevisionRecord)
def apply_change_request(session_id: str, rev: int, req: ApplyChangeRequest) -> RevisionRecord:
    try:
        session = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        base_rev = store.get_revision(session_id, rev)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Revision not found")

    change_request = req.change_request.strip()
    if not change_request:
        raise HTTPException(status_code=400, detail="Missing change_request")

    schema_path = Path(__file__).resolve().parent / "schema" / "partspec_v1.schema.json"
    partspec_schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))

    try:
        from backend.app.services.openai_orchestrator import revise_partspec_with_change_request
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI integration not available: {e}")

    try:
        new_spec = revise_partspec_with_change_request(
            base_rev.spec,
            change_request,
            csv_path=req.csv_path,
            model=req.model,
            partspec_schema=partspec_schema,
            constraints=session.constraints,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI change request failed: {e}")

    try:
        return store.create_revision_from_spec(
            session_id,
            spec=new_spec,
            source="ai_change",
            note=req.note,
            input_description=base_rev.input_description,
            input_sketch_asset=base_rev.input_sketch_asset,
            change_request=change_request,
            stage="concept",
            metadata={"change_from_revision": base_rev.rev},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sessions/{session_id}/revisions/{rev}/clarify", response_model=RevisionRecord)
def answer_clarifications(session_id: str, rev: int, req: AnswerClarificationsRequest) -> RevisionRecord:
    try:
        session = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        base_rev = store.get_revision(session_id, rev)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Revision not found")

    if not req.answers:
        raise HTTPException(status_code=400, detail="Missing answers")

    schema_path = Path(__file__).resolve().parent / "schema" / "partspec_v1.schema.json"
    partspec_schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))

    sketch_path: Path | None = None
    if base_rev.input_sketch_asset:
        asset_id = base_rev.input_sketch_asset
        if "/" in asset_id or "\\" in asset_id:
            raise HTTPException(status_code=400, detail="Invalid sketch asset id in base revision")
        candidate = store.uploads_dir(session_id) / asset_id
        if candidate.exists():
            sketch_path = candidate

    qa_pairs = [{"question": a.question, "answer": a.answer} for a in req.answers]

    try:
        from backend.app.services.openai_orchestrator import revise_partspec_with_clarification_answers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI integration not available: {e}")

    try:
        new_spec = revise_partspec_with_clarification_answers(
            base_rev.spec,
            qa_pairs,
            original_description=base_rev.input_description,
            sketch_path=sketch_path,
            csv_path=req.csv_path,
            model=req.model,
            partspec_schema=partspec_schema,
            constraints=session.constraints,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI clarification failed: {e}")

    try:
        return store.create_revision_from_spec(
            session_id,
            spec=new_spec,
            source="ai_clarify",
            note=req.note,
            input_description=base_rev.input_description,
            input_sketch_asset=base_rev.input_sketch_asset,
            stage="concept",
            metadata={"clarify_from_revision": base_rev.rev, "answers": qa_pairs},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ----------------------------
# Uploads (sketches) - MVP
# ----------------------------

@app.post("/sessions/{session_id}/uploads/sketch")
def upload_sketch(session_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        _ = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    filename = (file.filename or "").lower()
    ext = Path(filename).suffix
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Only .png, .jpg, .jpeg, .webp are supported for sketches in MVP.")

    asset_id = __import__("uuid").uuid4().hex + ext
    out_path = store.uploads_dir(session_id) / asset_id
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        f.write(file.file.read())

    return {
        "asset_id": asset_id,
        "url": f"/sessions/{session_id}/uploads/{asset_id}",
        "filename": file.filename,
        "content_type": file.content_type,
    }


@app.get("/sessions/{session_id}/uploads/{asset_id}")
def get_upload(session_id: str, asset_id: str) -> FileResponse:
    try:
        _ = store.get_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")

    if "/" in asset_id or "\\" in asset_id:
        raise HTTPException(status_code=400, detail="Invalid asset_id")

    path = store.uploads_dir(session_id) / asset_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload not found")
    return FileResponse(path)


# ----------------------------
# Back-compat: AI scaffold endpoint
# ----------------------------

@app.post("/ai/generate_partspec_from_text")
def ai_generate_partspec_from_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    description = (payload.get("description") or "").strip()
    if not description:
        raise HTTPException(status_code=400, detail="Missing 'description'")

    schema_path = Path(__file__).resolve().parent / "schema" / "partspec_v1.schema.json"
    partspec_schema = __import__("json").loads(schema_path.read_text(encoding="utf-8"))

    try:
        from backend.app.services.openai_orchestrator import generate_partspec_from_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI integration not available: {e}")

    try:
        spec = generate_partspec_from_text(
            description,
            partspec_schema=partspec_schema,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI call failed: {e}")

    return spec.model_dump()
