from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.core.config import DATA_DIR, ARTIFACTS_DIR
from backend.app.models.constraints import Constraints, ConstraintsPatch
from backend.app.models.partspec import PartSpec
from backend.app.models.session import SessionRecord, RevisionRecord
from backend.app.services.validation_engine import validate_partspec
from backend.app.services.cad_engine import export_stl
from backend.app.services.drawing_engine import generate_orthographic_svg
from backend.app.services.constraints_engine import (
    apply_export_constraints,
    constraints_hash,
    hole_export_diameters_from_effects,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class ArtifactResult:
    run_id: str
    stl_path: Path
    drawing_path: Path
    dimensions_path: Path
    payload: Dict[str, Any]


def generate_artifacts_for_spec(spec: PartSpec, *, constraints: Optional[Constraints] = None) -> ArtifactResult:
    """Deterministically generate STL + drawing + dimensions.json for a spec.

    - Hard validation errors block artifact generation (e.g. too-large grid).
    - Constraint errors do NOT block artifact generation (they block approval/finalize).
    """
    validation = validate_partspec(spec, constraints=constraints)
    if validation.get("errors"):
        raise ValueError(f"Spec validation failed: {validation['errors']}")

    run_id = uuid.uuid4().hex
    out_dir = ARTIFACTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stl_path = out_dir / "model.stl"
    drawing_path = out_dir / "drawing.svg"
    dims_path = out_dir / "dimensions.json"

    # Apply export-only effects (e.g., hole offset) for STL export + reports
    eff_spec = spec
    effects: Dict[str, Any] = {}
    if constraints is not None:
        eff_spec, effects = apply_export_constraints(spec, constraints)

    hole_export_diameters = hole_export_diameters_from_effects(effects)

    notes: List[str] = []
    if constraints is not None and abs(float(constraints.hole_diameter_offset or 0.0)) > 1e-12:
        notes.append(
            f"Hole diameter offset {float(constraints.hole_diameter_offset):+g} {spec.units} applied at STL export."
        )

    max_overhang = None
    if constraints is not None and bool(constraints.must_be_printable):
        max_overhang = float(constraints.max_overhang_angle_deg)

    mesh_report = export_stl(eff_spec, stl_path, max_overhang_angle_deg=max_overhang)
    drawing_report = generate_orthographic_svg(spec, drawing_path, hole_export_diameters=hole_export_diameters, notes=notes)

    dims = {
        "constraints": constraints.model_dump(exclude_none=True) if constraints is not None else None,
        "constraints_hash": constraints_hash(constraints) if constraints is not None else None,
        "constraints_effects": effects,
        "validation": validation,
        "mesh_report": mesh_report,
        "drawing_report": drawing_report,
    }
    dims_path.write_text(json.dumps(dims, indent=2), encoding="utf-8")

    payload = {
        "run_id": run_id,
        "stl_url": f"/artifacts/{run_id}/model.stl",
        "drawing_url": f"/artifacts/{run_id}/drawing.svg",
        "dimensions_url": f"/artifacts/{run_id}/dimensions.json",
        "mesh_report": mesh_report,
        "drawing_report": drawing_report,
        "validation": validation,
        "constraints": dims["constraints"],
        "constraints_hash": dims["constraints_hash"],
        "constraints_effects": effects,
    }

    return ArtifactResult(
        run_id=run_id,
        stl_path=stl_path,
        drawing_path=drawing_path,
        dimensions_path=dims_path,
        payload=payload,
    )


class SessionStore:
    """Filesystem-backed session + revision store (MVP).

    Directory layout:
      data/
        sessions/
          {session_id}/
            session.json
            revisions/
              0001.json
              0002.json
            uploads/
              ...
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or (DATA_DIR / "sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---------- paths ----------

    def session_dir(self, session_id: str) -> Path:
        return self.base_dir / session_id

    def session_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def revisions_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "revisions"

    def revision_path(self, session_id: str, rev: int) -> Path:
        return self.revisions_dir(session_id) / f"{rev:04d}.json"

    def uploads_dir(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "uploads"

    # ---------- sessions ----------

    def create_session(self, *, name: str | None = None, metadata: Dict[str, Any] | None = None) -> SessionRecord:
        session_id = uuid.uuid4().hex
        now = _now_iso()
        rec = SessionRecord(
            id=session_id,
            created_at=now,
            updated_at=now,
            status="draft",
            name=name,
            metadata=metadata or {},
            current_revision=None,
            approved_revision=None,
            final_revision=None,
            # constraints default is provided by SessionRecord
        )
        self._write_session(rec)
        return rec

    def get_session(self, session_id: str) -> SessionRecord:
        path = self.session_path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def update_constraints(self, session_id: str, patch: ConstraintsPatch) -> SessionRecord:
        session = self.get_session(session_id)
        session.constraints = patch.apply_to(session.constraints)
        # Changing constraints invalidates approvals/finals (requirements changed)
        session.approved_revision = None
        session.final_revision = None
        if session.current_revision is not None:
            session.status = "concept"
        self._write_session(session)
        return session

    def constraints_hash(self, session_id: str) -> str:
        session = self.get_session(session_id)
        return constraints_hash(session.constraints)

    def _write_session(self, rec: SessionRecord) -> None:
        rec.updated_at = _now_iso()
        _atomic_write_text(self.session_path(rec.id), rec.model_dump_json(indent=2))

    # ---------- revisions ----------

    def list_revision_numbers(self, session_id: str) -> List[int]:
        rdir = self.revisions_dir(session_id)
        if not rdir.exists():
            return []
        nums: List[int] = []
        for p in sorted(rdir.glob("*.json")):
            try:
                nums.append(int(p.stem))
            except Exception:
                continue
        return sorted(nums)

    def next_revision_number(self, session_id: str) -> int:
        nums = self.list_revision_numbers(session_id)
        return (max(nums) + 1) if nums else 1

    def get_revision(self, session_id: str, rev: int) -> RevisionRecord:
        path = self.revision_path(session_id, rev)
        if not path.exists():
            raise FileNotFoundError(f"Revision not found: session={session_id}, rev={rev}")
        return RevisionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list_revisions(self, session_id: str) -> List[RevisionRecord]:
        nums = self.list_revision_numbers(session_id)
        return [self.get_revision(session_id, n) for n in nums]

    def _write_revision(self, rec: RevisionRecord) -> None:
        _atomic_write_text(self.revision_path(rec.session_id, rec.rev), rec.model_dump_json(indent=2))

    def create_revision_from_spec(
        self,
        session_id: str,
        *,
        spec: PartSpec,
        source: str,
        note: str | None = None,
        input_description: str | None = None,
        input_sketch_asset: str | None = None,
        change_request: str | None = None,
        stage: str = "concept",
        metadata: Dict[str, Any] | None = None,
    ) -> RevisionRecord:
        # Ensure session exists
        session = self.get_session(session_id)

        rev = self.next_revision_number(session_id)
        now = _now_iso()

        # Generate artifacts (deterministic)
        artifact = generate_artifacts_for_spec(spec, constraints=session.constraints)

        validation = artifact.payload["validation"]

        # Stamp constraints snapshot/hash onto the revision
        meta = dict(metadata or {})
        meta.setdefault("constraints", session.constraints.model_dump(exclude_none=True))
        meta.setdefault("constraints_hash", constraints_hash(session.constraints))

        rec = RevisionRecord(
            session_id=session_id,
            rev=rev,
            created_at=now,
            source=source,  # type: ignore[arg-type]
            note=note,
            input_description=input_description,
            input_sketch_asset=input_sketch_asset,
            change_request=change_request,
            stage=stage,  # type: ignore[arg-type]
            spec=spec,
            validation=validation,
            artifacts=artifact.payload,
            metadata=meta,
        )
        self._write_revision(rec)

        # Update session pointers/status
        session.current_revision = rev

        # Any new concept revision invalidates prior approvals/finals.
        if stage == "concept":
            session.status = "concept"
            session.approved_revision = None
            session.final_revision = None

        self._write_session(session)

        return rec

    def approve_revision(self, session_id: str, rev: int) -> SessionRecord:
        session = self.get_session(session_id)
        r = self.get_revision(session_id, rev)

        # Block approval if constraints changed since the revision was produced
        current_hash = constraints_hash(session.constraints)
        rev_hash = (r.metadata or {}).get("constraints_hash")
        if rev_hash and rev_hash != current_hash:
            raise ValueError(
                "Session constraints changed since this revision was generated. "
                "Regenerate a revision under the current constraints before approving."
            )

        # Block approval on constraint violations
        c_errs = r.validation.get("constraint_errors") if isinstance(r.validation, dict) else None
        if c_errs:
            raise ValueError(f"Cannot approve: revision violates constraints: {c_errs}")

        session.approved_revision = rev
        session.status = "approved"
        self._write_session(session)
        return session

    def finalize_session(
        self,
        session_id: str,
        *,
        target_pitch_factor: float = 0.5,
        note: str | None = None,
    ) -> RevisionRecord:
        session = self.get_session(session_id)
        if session.approved_revision is None:
            raise ValueError("Session has no approved_revision. Approve a revision first.")

        approved = self.get_revision(session_id, session.approved_revision)

        # Ensure constraints hash still matches
        current_hash = constraints_hash(session.constraints)
        rev_hash = (approved.metadata or {}).get("constraints_hash")
        if rev_hash and rev_hash != current_hash:
            raise ValueError(
                "Session constraints changed since the approved revision. "
                "Regenerate and re-approve before finalizing."
            )

        # Copy spec and refine tessellation for final
        spec_dict = approved.spec.model_dump()
        spec_dict.setdefault("metadata", {})
        spec_dict["metadata"] = dict(spec_dict.get("metadata") or {})
        spec_dict["metadata"]["stage"] = "final"

        old_pitch = float(spec_dict["tessellation"]["grid_pitch"])
        desired = old_pitch * float(target_pitch_factor)

        def estimate_voxels(pitch: float) -> int:
            tmp = spec_dict.copy()
            tmp["tessellation"] = dict(tmp["tessellation"])
            tmp["tessellation"]["grid_pitch"] = pitch
            tmp_spec = PartSpec.model_validate(tmp)
            v = validate_partspec(tmp_spec, constraints=session.constraints)
            if v.get("errors"):
                raise ValueError(f"Spec became invalid during finalize: {v['errors']}")
            return int(v["estimated_grid"]["voxels"])

        max_vox = int(spec_dict["tessellation"]["max_voxels"])
        pitch = max(desired, 1e-6)

        # If too big, relax pitch until it fits
        for _ in range(30):
            vox = estimate_voxels(pitch)
            if vox <= max_vox:
                break
            pitch *= 1.25
        else:
            raise ValueError("Could not find a safe grid_pitch for finalize within iteration limit.")

        spec_dict["tessellation"]["grid_pitch"] = pitch
        final_spec = PartSpec.model_validate(spec_dict)

        final_rev = self.create_revision_from_spec(
            session_id,
            spec=final_spec,
            source="finalize",
            note=note,
            input_description=approved.input_description,
            input_sketch_asset=approved.input_sketch_asset,
            stage="final",
            metadata={
                "finalize": {
                    "approved_revision": approved.rev,
                    "old_grid_pitch": old_pitch,
                    "target_pitch_factor": target_pitch_factor,
                    "final_grid_pitch": pitch,
                }
            },
        )

        session = self.get_session(session_id)
        session.status = "final"
        session.final_revision = final_rev.rev
        self._write_session(session)

        return final_rev
