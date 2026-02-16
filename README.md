# AI STL Builder (MVP Skeleton)

This repo is an early build step for a system that:
1) takes a rough sketch + description,
2) turns it into a **parametric PartSpec** (JSON),
3) generates a **dimensioned orthographic drawing** for approval,
4) then produces an STL.

## What exists in this MVP (v0.4.2)

- **PartSpec v1** JSON Schema (`backend/app/schema/partspec_v1.schema.json`)
- A deterministic **CAD engine** that can build **CSG from primitives** (box + cylinder) using an SDF grid + marching cubes
- STL export via `trimesh`
- A simple orthographic **drawing generator** (SVG) for the common MVP case: *box base with cylindrical cut holes*
- **Session constraints** (fit envelope, target size/tolerance, min wall thickness, min feature size, hole diameter offset, and a simple printability heuristic).
  - Enforced on **approve/finalize** and shown as pass/fail badges per revision in the UI.
  - API: `GET /sessions/{id}/constraints` and `PUT /sessions/{id}/constraints`
- FastAPI endpoints:
  - stateless: validate a spec and generate artifacts
  - stateful: **sessions + revisions** (filesystem persistence under `data/`)
  - sketch upload endpoint (stores under `data/sessions/{id}/uploads/`)
- `user_api.csv.example` and a small loader module (for later OpenAI integration)
- **Sketch+text → PartSpec** (OpenAI vision input via Responses API)
- **Clarification loop**: answer AI questions in the UI and regenerate a new revision (OpenAI)

## Quickstart (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn backend.app.main:app --reload
```

Then open:
- http://127.0.0.1:8000/  (simple web UI)
- http://127.0.0.1:8000/docs  (API docs)

## Stateless usage (PartSpec -> STL + drawing)

```bash
python scripts/generate_from_spec.py backend/app/examples/plate_with_hole.json --out out/
```

Outputs:
- `out/model.stl`
- `out/drawing.svg`
- `out/dimensions.json`

Or via API: POST the PartSpec JSON to:
- `POST /generate_artifacts`

## Session workflow (revisioned, approval-first)

### 1) Create a session

```bash
curl -X POST http://127.0.0.1:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"name":"Demo session"}'
```

Response includes `id` (your session id).

### 2) Create a concept revision (from an explicit PartSpec)

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions \
  -H "Content-Type: application/json" \
  -d @backend/app/examples/plate_with_hole_create_revision_payload.json
```

(That payload file is included as an example.)

### 3) Approve a revision

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions/1/approve
```

### 4) Finalize (higher-res STL)

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/finalize \
  -H "Content-Type: application/json" \
  -d '{"target_pitch_factor":0.5, "note":"final STL"}'
```

This creates a **new revision** with `stage="final"` and sets the session status to `final`.

### 5) Apply a change request (AI edits the spec)

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions/1/change \
  -H "Content-Type: application/json" \
  -d '{"change_request":"Move hole H1 5mm to the left and make all holes Ø8mm.","note":"tweak"}'
```

This will create a new concept revision (and will invalidate any prior approvals).

### 6) Answer clarification questions (AI)

If a revision's `spec.metadata.questions_for_user` contains questions, you can send answers to create a new concept revision:

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions/1/clarify \
  -H "Content-Type: application/json" \
  -d '{
    "answers": [
      {"question": "Is the hole through-all?", "answer": "Yes, through-all"},
      {"question": "What is the wall thickness?", "answer": "3mm"}
    ],
    "note": "answered clarifications"
  }'
```

This will re-use the same sketch (if the revision had one) to preserve layout intent.

## Sketch uploads (MVP)

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/uploads/sketch \
  -F "file=@sketch.png"
```

Response includes `asset_id` and a `url` you can view.

## Optional: OpenAI (ChatGPT) text -> PartSpec

1) Provide credentials (recommended: env var so no key file is written):

```bash
export OPENAI_API_KEY="sk-..."
# optional:
# export OPENAI_ORGANIZATION="org_..."
# export OPENAI_PROJECT="proj_..."
```

Alternative file mode (also safe to keep local):

```bash
cp user_api.csv.example user_api.csv
# edit user_api.csv
```

`user_api.csv` is ignored by git via `.gitignore` and should never be committed.

2) Create a concept revision from text:

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions \
  -H "Content-Type: application/json" \
  -d '{"mode":"from_text","description":"Create a 60mm x 40mm x 5mm plate with two Ø6mm through holes, centered at x=-15 and x=+15, y=0.","note":"first concept"}'
```

That uses Structured Outputs (JSON schema strict) in the OpenAI call and stores the generated PartSpec as revision 1.

## Optional: OpenAI sketch + text -> PartSpec

1) Upload a sketch (PNG/JPG/WebP):

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/uploads/sketch \
  -F "file=@sketch.png"
```

The response includes an `asset_id`.

2) Create a concept revision from text **and** the sketch:

```bash
curl -X POST http://127.0.0.1:8000/sessions/<SESSION_ID>/revisions \
  -H "Content-Type: application/json" \
  -d '{
    "mode":"from_text",
    "description":"Make a simple wall-mount bracket like in the sketch. Overall width 60mm, height 40mm, thickness 6mm. Two through-holes Ø5mm.",
    "sketch_asset_id":"<ASSET_ID>",
    "note":"concept from sketch"
  }'
```

Notes:
- The backend will pass the sketch as an `input_image` in a Responses API request, alongside your text.
- If Pillow is installed, the sketch is automatically downscaled/compressed for cheaper/faster calls.

## Notes / current limitations (intentional for MVP)

- Only **axis-aligned** primitives (no rotations yet).
- Supported primitives: `box`, `cylinder_z`.
- Supported CSG ops: `union`, `difference`, `intersection` (difference is the main one used so far).
- Drawing engine only produces detailed feature dimensioning for the pattern:
  `difference(base_box, [cylinders...])`.

Next build steps:
- add change requests (AI spec editing loop),
- add more primitives (extrude from 2D, chamfers/fillets),
- add refinement pipeline (blocky concept -> smoothed final) without breaking approved dims.
