from __future__ import annotations

import base64
import io
import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


def _deep_merge_json_schema(a: Any, b: Any) -> Any:
    """Best-effort deep merge for JSON Schema dicts.
    - dicts: recursively merge
    - lists: for 'required' we union unique items; otherwise prefer b
    - scalars: prefer b
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = _deep_merge_json_schema(out[k], v)
            else:
                out[k] = v
        # Special-case: required should be a union
        if isinstance(a.get("required"), list) and isinstance(b.get("required"), list):
            seen = set()
            req = []
            for item in a["required"] + b["required"]:
                if item not in seen:
                    seen.add(item)
                    req.append(item)
            out["required"] = req
        return out
    if isinstance(a, list) and isinstance(b, list):
        return b
    return b


def sanitize_json_schema_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI Structured Outputs supports only a subset of JSON Schema.
    In particular, `allOf` is rejected. Pydantic often emits `allOf` wrappers.
    This function recursively flattens/removes `allOf` while preserving intent
    as well as we can (best-effort merge).
    """
    def _sanitize(node: Any) -> Any:
        if isinstance(node, list):
            return [_sanitize(x) for x in node]
        if not isinstance(node, dict):
            return node

        # First sanitize children
        node = {k: _sanitize(v) for k, v in node.items()}

        if "allOf" in node:
            subs = node.pop("allOf") or []
            merged: Any = {}
            for sub in subs:
                merged = _deep_merge_json_schema(merged, _sanitize(sub))
            merged = _deep_merge_json_schema(merged, node)
            return merged

        return node

    cleaned = _sanitize(copy.deepcopy(schema))
    # Ensure result is a dict
    if not isinstance(cleaned, dict):
        raise ValueError("Sanitized schema must be an object")
    return cleaned


from backend.app.core.api_key import load_openai_credentials_from_csv
from backend.app.models.constraints import Constraints
from backend.app.models.partspec import PartSpec
from backend.app.services.validation_engine import validate_partspec


PARTSPEC_SYSTEM_PROMPT = """You are a mechanical CAD assistant.

You must output a JSON object that conforms to the provided PartSpec schema.
- Use only supported primitives and operations.
- Use consistent units.
- Do not add rotation (rotation must be [0,0,0] in this MVP schema).
- Prefer to create a simple, blocky concept first (no fillets).
- Ensure the PartSpec can be built deterministically (all ids referenced exist).

Also include a short planning summary and assumptions in the PartSpec metadata:
- metadata.plan_summary: a short string describing the intended geometry
- metadata.assumptions: list of strings for any inferred/missing dimensions
- metadata.questions_for_user: list of strings for any questions the user should answer to improve accuracy

If session constraints are provided, you MUST obey them. If the constraints conflict with the user's request, propose the closest design and add targeted questions under metadata.questions_for_user.
"""


PARTSPEC_REVISION_SYSTEM_PROMPT = """You are a mechanical CAD assistant.

You will receive:
1) an existing PartSpec JSON that already conforms to the schema, and
2) a change request in plain English.

You must output a NEW PartSpec JSON that conforms to the provided schema.

Rules:
- Only use supported primitives and operations.
- Rotation is not implemented: rotation must remain [0,0,0].
- Preserve the existing design intent and dimensions unless the change request explicitly modifies them.
- Keep existing solid/operation ids stable when possible. If you add new solids/ops, give them unique ids.
- If the change request asks for geometry you cannot represent in the MVP schema, record the request under metadata.unimplemented_requests as a list of strings, and make the closest possible representable change.

If session constraints are provided, you MUST obey them. If the constraints conflict with the change request, propose the closest design and add targeted questions under metadata.questions_for_user.
"""


PARTSPEC_CLARIFICATION_SYSTEM_PROMPT = """You are a mechanical CAD assistant.

You will receive:
1) an existing PartSpec JSON that already conforms to the schema,
2) the original user description (if available),
3) a list of clarification questions and the user's answers, and
4) optionally, the original sketch image.

You must output a NEW PartSpec JSON that conforms to the provided schema.

Rules:
- Only use supported primitives and operations.
- Rotation is not implemented: rotation must remain [0,0,0].
- Use the user's answers to replace assumptions and resolve ambiguities.
- Preserve existing design intent and dimensions unless the answers explicitly change them.
- Keep existing solid/operation ids stable when possible. If you add new solids/ops, give them unique ids.

Metadata requirements:
- metadata.plan_summary: update if the concept changes materially
- metadata.assumptions: remove items that are now resolved; add any new assumptions
- metadata.questions_for_user: if anything is still ambiguous, ask follow-up questions (keep short and specific)

If session constraints are provided, you MUST obey them. If the constraints conflict with the answers, propose the closest design and add targeted questions under metadata.questions_for_user.
"""


def _guess_mime_from_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _encode_sketch_as_data_url(
    sketch_path: Path,
    *,
    max_side_px: int = 1024,
    jpeg_quality: int = 85,
    max_bytes: int = 5_000_000,
) -> str:
    """Encode a local image file as a data URL for Responses API input_image.

    - If Pillow is available, we downscale and convert to JPEG to keep tokens/cost reasonable.
    - Otherwise we base64 the raw file.
    """
    if not sketch_path.exists():
        raise FileNotFoundError(f"Sketch not found: {sketch_path}")

    if Image is not None:
        with Image.open(sketch_path) as im:
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.paste(rgba, mask=rgba.split()[-1])
                im2 = bg.convert("RGB")
            else:
                im2 = im.convert("RGB")

            im2.thumbnail((max_side_px, max_side_px))

            buf = io.BytesIO()
            im2.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
            data = buf.getvalue()

        if len(data) > max_bytes:
            raise ValueError(
                f"Sketch is too large after compression ({len(data)} bytes). "
                f"Please upload a smaller image (try < {max_bytes/1_000_000:.1f}MB)."
            )

        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    raw = sketch_path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(
            f"Sketch is too large ({len(raw)} bytes). "
            f"Please upload a smaller image (try < {max_bytes/1_000_000:.1f}MB)."
        )
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = _guess_mime_from_path(sketch_path)
    return f"data:{mime};base64,{b64}"


def get_client(csv_path: str | Path = "user_api.csv") -> "OpenAI":
    if OpenAI is None:
        raise ImportError("openai package not installed. Install requirements.txt (pip install -r requirements.txt).")
    creds = load_openai_credentials_from_csv(csv_path)
    return OpenAI(api_key=creds.api_key, organization=creds.organization, project=creds.project)


def _extract_output_text(resp: Any) -> str:
    """Extract the model's text output from an OpenAI Responses API object."""
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) in ("output_text", "text"):
                txt = getattr(c, "text", None)
                if isinstance(txt, str) and txt.strip():
                    return txt

    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    raise RuntimeError("Could not find JSON text in OpenAI response.")


def _format_constraints(constraints: Constraints) -> str:
    return json.dumps(constraints.model_dump(exclude_none=True), indent=2)


def _constraint_questions_from_validation(v: Dict[str, Any], constraints: Constraints) -> List[str]:
    """Turn constraint validation failures into user-facing questions."""
    qs: List[str] = []
    report = (v.get("constraints_report") or {}) if isinstance(v, dict) else {}

    # Fit within
    mf = report.get("must_fit_within")
    if isinstance(mf, dict) and not mf.get("ok", True):
        limit = mf.get("limit")
        margins = mf.get("margins")
        qs.append(
            f"Constraints require the part to fit within {limit} {constraints.preferred_units}, but current margins are {margins}. Which dimension(s) should I reduce, or should the fit-within limit be increased?"
        )

    # Target overall
    tgt = report.get("target_overall_size")
    if isinstance(tgt, dict) and not tgt.get("ok", True):
        target = tgt.get("target")
        tol = tgt.get("tolerance")
        deltas = tgt.get("deltas")
        qs.append(
            f"Constraints specify target overall size {target} ±{tol} {constraints.preferred_units}, but deltas are {deltas}. Should I match the target exactly, or should the tolerance/target be adjusted?"
        )

    # Min wall
    mw = report.get("min_wall_thickness")
    if isinstance(mw, dict):
        if not mw.get("supported", True):
            qs.append(
                "Minimum wall thickness constraint is set, but the current geometry isn't in the MVP-supported pattern (box minus hole cylinders). Should I simplify the design to that pattern, or relax min_wall_thickness?"
            )
        elif mw.get("supported") and not mw.get("ok", True):
            req = mw.get("required")
            actual = mw.get("min_wall_actual")
            worst = (mw.get("min_wall_worst_case") or {}).get("kind")
            qs.append(
                f"Minimum wall thickness must be ≥ {req} {constraints.preferred_units}, but current worst case is ~{actual} ({worst}). Should I move holes inward, reduce hole diameter, or relax min_wall_thickness?"
            )

    # Min feature
    mfs = report.get("min_feature_size")
    if isinstance(mfs, dict) and not mfs.get("ok", True):
        req = mfs.get("required")
        actual = mfs.get("actual_min")
        qs.append(
            f"Minimum feature size must be ≥ {req} {constraints.preferred_units}, but current smallest feature is ~{actual}. Should I enlarge the small feature(s), or relax min_feature_size?"
        )

    # Units
    u = report.get("units")
    if isinstance(u, dict) and not u.get("ok", True):
        qs.append(
            f"Preferred units are {u.get('preferred')}, but the spec is in {u.get('actual')}. Should I convert everything to {u.get('preferred')}?"
        )

    if not qs:
        errs = v.get("constraint_errors") if isinstance(v, dict) else None
        if errs:
            qs.append(
                "Some constraints are not currently satisfiable. Which constraint(s) are flexible, and which must be strictly enforced?"
            )

    # Cap to avoid runaway UI
    return qs[:6]


def _inject_questions(spec: PartSpec, questions: List[str]) -> None:
    if not questions:
        return
    spec.metadata = dict(spec.metadata or {})
    existing = spec.metadata.get("questions_for_user")
    if not isinstance(existing, list):
        existing_list: List[str] = []
    else:
        existing_list = [str(x) for x in existing if str(x).strip()]

    # merge unique
    out: List[str] = existing_list[:]
    for q in questions:
        q = str(q).strip()
        if not q:
            continue
        if q not in out:
            out.append(q)
    spec.metadata["questions_for_user"] = out


def generate_partspec_from_text(
    description: str,
    *,
    csv_path: str | Path = "user_api.csv",
    model: str = "gpt-4.1",
    partspec_schema: Dict[str, Any] | None = None,
    constraints: Constraints | None = None,
    max_attempts: int = 2,
) -> PartSpec:
    """Create a PartSpec from text only using Structured Outputs.

    If constraints are provided, we validate and retry. If constraint violations remain
    after retries, we return the best-effort spec with targeted questions in metadata.
    """
    client = get_client(csv_path)
    if partspec_schema is None:
        raise ValueError("partspec_schema is required for structured output.")

    constraint_block = ""
    if constraints is not None:
        constraint_block = "\n\nSESSION CONSTRAINTS (must obey):\n" + _format_constraints(constraints) + "\n"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": PARTSPEC_SYSTEM_PROMPT},
        {"role": "user", "content": (description.strip() + constraint_block)},
    ]

    last_spec: PartSpec | None = None
    last_v: Dict[str, Any] | None = None
    last_err: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "PartSpec",
                    "schema": sanitize_json_schema_for_openai(partspec_schema),
                    "strict": True,
                }
            },
            store=False,
        )
        out_text = _extract_output_text(resp)

        try:
            spec = PartSpec.model_validate_json(out_text)
        except Exception as e:
            last_err = f"Schema/Pydantic validation failed: {e}"
            messages.append({"role": "user", "content": f"Your JSON did not validate. Error: {e}. Please try again."})
            continue

        v = validate_partspec(spec, constraints=constraints)
        last_spec, last_v = spec, v

        if not v.get("errors") and not v.get("constraint_errors"):
            return spec

        # Hard errors must be fixed
        if v.get("errors"):
            last_err = f"Internal validation failed: {v['errors']}"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The PartSpec failed internal validation and cannot be meshed/exported.\n"
                        f"Errors: {v['errors']}\n"
                        "Please revise the PartSpec to fix these errors."
                    ),
                }
            )
            continue

        # Constraint violations: ask model to fix
        if v.get("constraint_errors"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The PartSpec violates session constraints. Please revise it to satisfy the constraints.\n"
                        f"Constraint errors: {v['constraint_errors']}\n"
                        "If constraints conflict with the user's request, make the closest possible design and add targeted questions under metadata.questions_for_user."
                    ),
                }
            )

    # exhausted attempts
    if last_v and last_v.get("errors"):
        raise ValueError(last_err or f"Generated spec failed validation: {last_v.get('errors')}")

    if last_spec is None:
        raise ValueError(last_err or "Failed to generate PartSpec.")

    # still constraint errors: inject questions
    if constraints is not None and last_v and last_v.get("constraint_errors"):
        qs = _constraint_questions_from_validation(last_v, constraints)
        _inject_questions(last_spec, qs)

    return last_spec


def generate_partspec_from_text_and_sketch(
    description: str,
    sketch_path: str | Path,
    *,
    csv_path: str | Path = "user_api.csv",
    model: str = "gpt-4.1",
    partspec_schema: Dict[str, Any] | None = None,
    constraints: Constraints | None = None,
    max_attempts: int = 2,
) -> PartSpec:
    """Create a PartSpec using BOTH a text description and a sketch image."""
    client = get_client(csv_path)
    if partspec_schema is None:
        raise ValueError("partspec_schema is required for structured output.")

    sketch_path = Path(sketch_path)
    image_data_url = _encode_sketch_as_data_url(sketch_path)

    constraint_block = ""
    if constraints is not None:
        constraint_block = "\n\nSESSION CONSTRAINTS (must obey):\n" + _format_constraints(constraints) + "\n"

    user_content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Design request:\n"
                f"{description.strip()}\n"
                + constraint_block
                + "\nA sketch image is provided after this text. Use it to infer overall shape and feature placement. "
                "If dimensions are missing or unclear, record assumptions under metadata.assumptions. "
                "If something is ambiguous, add questions under metadata.questions_for_user.\n"
            ),
        },
        {"type": "input_image", "image_url": image_data_url},
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": PARTSPEC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_spec: PartSpec | None = None
    last_v: Dict[str, Any] | None = None
    last_err: Optional[str] = None

    for _attempt in range(1, max_attempts + 1):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "PartSpec",
                    "schema": sanitize_json_schema_for_openai(partspec_schema),
                    "strict": True,
                }
            },
            store=False,
        )
        out_text = _extract_output_text(resp)

        try:
            spec = PartSpec.model_validate_json(out_text)
        except Exception as e:
            last_err = f"Schema/Pydantic validation failed: {e}"
            messages.append({"role": "user", "content": f"Your JSON did not validate. Error: {e}. Please try again."})
            continue

        v = validate_partspec(spec, constraints=constraints)
        last_spec, last_v = spec, v

        if not v.get("errors") and not v.get("constraint_errors"):
            return spec

        if v.get("errors"):
            last_err = f"Internal validation failed: {v['errors']}"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The PartSpec failed internal validation and cannot be meshed/exported.\n"
                        f"Errors: {v['errors']}\n"
                        "Please revise the PartSpec to fix these errors."
                    ),
                }
            )
            continue

        if v.get("constraint_errors"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The PartSpec violates session constraints. Please revise it to satisfy the constraints.\n"
                        f"Constraint errors: {v['constraint_errors']}\n"
                        "If constraints conflict with the user's request, make the closest possible design and add targeted questions under metadata.questions_for_user."
                    ),
                }
            )

    if last_v and last_v.get("errors"):
        raise ValueError(last_err or f"Generated spec failed validation: {last_v.get('errors')}")

    if last_spec is None:
        raise ValueError(last_err or "Failed to generate PartSpec.")

    if constraints is not None and last_v and last_v.get("constraint_errors"):
        qs = _constraint_questions_from_validation(last_v, constraints)
        _inject_questions(last_spec, qs)

    return last_spec


def revise_partspec_with_change_request(
    existing: PartSpec,
    change_request: str,
    *,
    csv_path: str | Path = "user_api.csv",
    model: str = "gpt-4.1",
    partspec_schema: Dict[str, Any] | None = None,
    constraints: Constraints | None = None,
    max_attempts: int = 2,
) -> PartSpec:
    """Revise an existing PartSpec given a change request."""
    client = get_client(csv_path)
    if partspec_schema is None:
        raise ValueError("partspec_schema is required for structured output.")

    existing_json = existing.model_dump_json(indent=2)

    constraint_block = ""
    if constraints is not None:
        constraint_block = "\n\nSESSION CONSTRAINTS (must obey):\n" + _format_constraints(constraints) + "\n"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": PARTSPEC_REVISION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Existing PartSpec JSON:\n"
                f"{existing_json}\n\n"
                "Change request:\n"
                f"{change_request.strip()}\n"
                + constraint_block
                + "\nReturn the updated PartSpec JSON only."
            ),
        },
    ]

    last_spec: PartSpec | None = None
    last_v: Dict[str, Any] | None = None
    last_err: Optional[str] = None

    for _attempt in range(1, max_attempts + 1):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "PartSpec",
                    "schema": sanitize_json_schema_for_openai(partspec_schema),
                    "strict": True,
                }
            },
            store=False,
        )
        out_text = _extract_output_text(resp)

        try:
            spec = PartSpec.model_validate_json(out_text)
        except Exception as e:
            last_err = f"Schema/Pydantic validation failed: {e}"
            messages.append({"role": "user", "content": f"Your JSON did not validate. Error: {e}. Please try again."})
            continue

        v = validate_partspec(spec, constraints=constraints)
        last_spec, last_v = spec, v

        if not v.get("errors") and not v.get("constraint_errors"):
            return spec

        if v.get("errors"):
            last_err = f"Internal validation failed: {v['errors']}"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The updated PartSpec failed internal validation.\n"
                        f"Errors: {v['errors']}\n"
                        "Please revise the PartSpec to fix these errors while still satisfying the change request."
                    ),
                }
            )
            continue

        if v.get("constraint_errors"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The updated PartSpec violates session constraints. Please revise it to satisfy constraints.\n"
                        f"Constraint errors: {v['constraint_errors']}\n"
                        "If constraints conflict with the change request, make the closest possible design and add targeted questions under metadata.questions_for_user."
                    ),
                }
            )

    if last_v and last_v.get("errors"):
        raise ValueError(last_err or f"Failed to revise PartSpec: {last_v.get('errors')}")

    if last_spec is None:
        raise ValueError(last_err or "Failed to revise PartSpec.")

    if constraints is not None and last_v and last_v.get("constraint_errors"):
        qs = _constraint_questions_from_validation(last_v, constraints)
        _inject_questions(last_spec, qs)

    return last_spec


def revise_partspec_with_clarification_answers(
    existing: PartSpec,
    qa_pairs: List[Dict[str, str]],
    *,
    original_description: Optional[str] = None,
    sketch_path: str | Path | None = None,
    csv_path: str | Path = "user_api.csv",
    model: str = "gpt-4.1",
    partspec_schema: Dict[str, Any] | None = None,
    constraints: Constraints | None = None,
    max_attempts: int = 2,
) -> PartSpec:
    """Revise an existing PartSpec using user-provided answers."""
    client = get_client(csv_path)
    if partspec_schema is None:
        raise ValueError("partspec_schema is required for structured output.")

    existing_json = existing.model_dump_json(indent=2)

    qa_lines: List[str] = []
    for i, qa in enumerate(qa_pairs, start=1):
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q or not a:
            continue
        qa_lines.append(f"{i}. Q: {q}\n   A: {a}")
    qa_block = "\n".join(qa_lines) if qa_lines else "(none)"

    desc = (original_description or "").strip() or "(not provided)"

    constraint_block = ""
    if constraints is not None:
        constraint_block = "\n\nSESSION CONSTRAINTS (must obey):\n" + _format_constraints(constraints) + "\n"

    base_text = (
        "Existing PartSpec JSON:\n"
        f"{existing_json}\n\n"
        "Original user description:\n"
        f"{desc}\n\n"
        "Clarification Q/A (use these answers to update the design):\n"
        f"{qa_block}\n"
        + constraint_block
        + "\nReturn the updated PartSpec JSON only."
    )

    user_content: Any
    if sketch_path is not None:
        sp = Path(sketch_path)
        image_data_url = _encode_sketch_as_data_url(sp)
        user_content = [
            {"type": "input_text", "text": base_text},
            {"type": "input_image", "image_url": image_data_url},
        ]
    else:
        user_content = base_text

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": PARTSPEC_CLARIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    last_spec: PartSpec | None = None
    last_v: Dict[str, Any] | None = None
    last_err: Optional[str] = None

    for _attempt in range(1, max_attempts + 1):
        resp = client.responses.create(
            model=model,
            input=messages,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "PartSpec",
                    "schema": sanitize_json_schema_for_openai(partspec_schema),
                    "strict": True,
                }
            },
            store=False,
        )
        out_text = _extract_output_text(resp)

        try:
            spec = PartSpec.model_validate_json(out_text)
        except Exception as e:
            last_err = f"Schema/Pydantic validation failed: {e}"
            messages.append({"role": "user", "content": f"Your JSON did not validate. Error: {e}. Please try again."})
            continue

        v = validate_partspec(spec, constraints=constraints)
        last_spec, last_v = spec, v

        if not v.get("errors") and not v.get("constraint_errors"):
            return spec

        if v.get("errors"):
            last_err = f"Internal validation failed: {v['errors']}"
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The updated PartSpec failed internal validation.\n"
                        f"Errors: {v['errors']}\n"
                        "Please revise the PartSpec to fix these errors while still applying the clarification answers."
                    ),
                }
            )
            continue

        if v.get("constraint_errors"):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The updated PartSpec violates session constraints. Please revise it to satisfy constraints while applying the answers.\n"
                        f"Constraint errors: {v['constraint_errors']}\n"
                        "If constraints conflict with the answers, make the closest possible design and add targeted questions under metadata.questions_for_user."
                    ),
                }
            )

    if last_v and last_v.get("errors"):
        raise ValueError(last_err or f"Failed to revise PartSpec: {last_v.get('errors')}")

    if last_spec is None:
        raise ValueError(last_err or "Failed to revise PartSpec with clarification answers.")

    if constraints is not None and last_v and last_v.get("constraint_errors"):
        qs = _constraint_questions_from_validation(last_v, constraints)
        _inject_questions(last_spec, qs)

    return last_spec
