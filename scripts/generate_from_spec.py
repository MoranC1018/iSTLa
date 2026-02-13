#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import backend...` works when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from backend.app.models.partspec import PartSpec  # noqa: E402
from backend.app.services.validation_engine import validate_partspec  # noqa: E402
from backend.app.services.cad_engine import export_stl  # noqa: E402
from backend.app.services.drawing_engine import generate_orthographic_svg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate STL + drawing from a PartSpec JSON file.")
    parser.add_argument("spec_path", type=str, help="Path to PartSpec JSON")
    parser.add_argument("--out", type=str, default="out", help="Output directory")
    args = parser.parse_args()

    spec_path = Path(args.spec_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = PartSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))

    validation = validate_partspec(spec)
    if validation.get("errors"):
        raise SystemExit(f"Spec validation failed: {validation['errors']}")

    stl_report = export_stl(spec, out_dir / "model.stl")
    drawing_report = generate_orthographic_svg(spec, out_dir / "drawing.svg")

    out = {
        "validation": validation,
        "mesh_report": stl_report,
        "drawing_report": drawing_report,
    }
    (out_dir / "dimensions.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote:")
    print(" -", (out_dir / "model.stl").resolve())
    print(" -", (out_dir / "drawing.svg").resolve())
    print(" -", (out_dir / "dimensions.json").resolve())


if __name__ == "__main__":
    main()
