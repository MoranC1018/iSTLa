from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.models.constraints import Constraints
from backend.app.models.partspec import PartSpec, BoxSolid, CylinderZSolid
from backend.app.services.cad_engine import compute_bounds
from backend.app.services.constraints_engine import apply_export_constraints, constraints_hash


def _try_extract_base_and_holes(spec: PartSpec) -> Tuple[Optional[BoxSolid], List[CylinderZSolid], Optional[str]]:
    """MVP wall-thickness extractor.

    Supported pattern:
      final = difference(base_box, [cylinder_z tools...])

    Returns (base_box_or_none, hole_tools, error_reason_or_none)
    """
    op = next((o for o in spec.operations if o.id == spec.final), None)
    if op is None or op.op != "difference":
        return None, [], "Unsupported: final is not a difference operation"

    solid_map = {s.id: s for s in spec.solids}
    base = solid_map.get(op.a)
    if not isinstance(base, BoxSolid):
        return None, [], "Unsupported: op.a is not a box"

    holes: List[CylinderZSolid] = []
    for bid in op.b:
        s = solid_map.get(bid)
        if isinstance(s, CylinderZSolid) and s.role == "tool":
            holes.append(s)

    if not holes:
        return base, [], "No tool cylinders found in difference op"

    return base, holes, None


def _wall_thickness_report(base: BoxSolid, holes: List[CylinderZSolid]) -> Dict[str, Any]:
    """Compute a simple min wall thickness in XY for a base box with cylindrical hole tools."""
    cx, cy, _cz = base.center
    sx, sy, sz = base.size
    min_x, max_x = cx - sx / 2.0, cx + sx / 2.0
    min_y, max_y = cy - sy / 2.0, cy + sy / 2.0

    per_hole: List[Dict[str, Any]] = []
    worst = {"kind": None, "value": float("inf"), "detail": None}

    # margins to outer edges
    for h in holes:
        hx, hy, _ = h.center
        r = float(h.radius)
        margins = {
            "left": (hx - min_x) - r,
            "right": (max_x - hx) - r,
            "bottom": (hy - min_y) - r,
            "top": (max_y - hy) - r,
        }
        min_edge = min(margins.values())
        per_hole.append({"hole_id": h.id, "radius": r, "margins": margins, "min_edge_margin": min_edge})
        if min_edge < worst["value"]:
            worst = {"kind": "edge", "value": float(min_edge), "detail": {"hole_id": h.id, **margins}}

    # margins between holes
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            h1, h2 = holes[i], holes[j]
            x1, y1, _ = h1.center
            x2, y2, _ = h2.center
            d = float(np.hypot(x1 - x2, y1 - y2)) - float(h1.radius) - float(h2.radius)
            if d < worst["value"]:
                worst = {"kind": "between_holes", "value": float(d), "detail": {"a": h1.id, "b": h2.id}}

    # also consider Z thickness of the base (common interpretation for plates)
    if float(sz) < worst["value"]:
        worst = {"kind": "base_thickness_z", "value": float(sz), "detail": {"sz": float(sz)}}

    return {
        "supported": True,
        "base_size": {"sx": float(sx), "sy": float(sy), "sz": float(sz)},
        "per_hole": per_hole,
        "min_wall_actual": float(worst["value"]),
        "min_wall_worst_case": worst,
    }


def validate_partspec(spec: PartSpec, constraints: Optional[Constraints] = None) -> Dict[str, Any]:
    """Validate a PartSpec.

    Returns:
      - errors/warnings: hard issues that may prevent artifact generation
      - constraint_errors/constraint_warnings: requirement violations (block approval/finalize)
      - constraints_report: structured data for UI badges/deltas
    """
    errors: List[str] = []
    warnings: List[str] = []

    constraint_errors: List[str] = []
    constraint_warnings: List[str] = []

    # --- hard geometry/bounds/grid checks ---
    try:
        b = compute_bounds(spec)
    except Exception as e:
        errors.append(f"Failed to compute bounds: {e}")
        return {
            "errors": errors,
            "warnings": warnings,
            "constraint_errors": constraint_errors,
            "constraint_warnings": constraint_warnings,
        }

    size = b.size
    if np.any(size <= 0):
        errors.append(
            f"Computed bounds are non-positive in at least one axis: size={size.tolist()}. "
            "This can happen with intersection ops that don't overlap."
        )

    pitch = float(spec.tessellation.grid_pitch)
    pad = float(spec.tessellation.padding)
    padded_size = size + 2 * pad

    nx = int(np.ceil(padded_size[0] / pitch)) + 1
    ny = int(np.ceil(padded_size[1] / pitch)) + 1
    nz = int(np.ceil(padded_size[2] / pitch)) + 1
    voxels = int(nx * ny * nz)

    if voxels > int(spec.tessellation.max_voxels):
        errors.append(
            f"Meshing grid too large: {voxels:,} voxels (nx={nx}, ny={ny}, nz={nz}). "
            f"Increase grid_pitch or reduce part size; max_voxels={spec.tessellation.max_voxels:,}."
        )
    elif voxels > 1_500_000:
        warnings.append(f"Large grid: {voxels:,} voxels. This may be slow; consider increasing grid_pitch.")

    if pitch > 0:
        min_feature = 2.5 * pitch
        warnings.append(
            f"Resolution note: features smaller than ~{min_feature:.3g} {spec.units} may not appear clearly in the STL (grid_pitch={pitch})."
        )

    # If hard errors exist, we can still include constraint report, but most flows should stop.

    # --- constraints checks ---
    c_report: Dict[str, Any] = {}
    c_effects: Dict[str, Any] = {}

    if constraints is not None:
        c_hash = constraints_hash(constraints)
        c_report["constraints_hash"] = c_hash

        # Units
        units_ok = spec.units == constraints.preferred_units
        c_report["units"] = {"preferred": constraints.preferred_units, "actual": spec.units, "ok": units_ok}
        if not units_ok:
            constraint_errors.append(
                f"Units mismatch: preferred {constraints.preferred_units}, but spec.units is {spec.units}."
            )

        # Bounds size
        c_report["bounds"] = {"size": size.tolist(), "min": b.min_xyz.tolist(), "max": b.max_xyz.tolist()}

        # must_fit_within
        if constraints.must_fit_within is not None:
            mf = np.array(constraints.must_fit_within, dtype=float)
            margins = mf - size
            fits = bool(np.all(margins >= -1e-9))
            c_report["must_fit_within"] = {
                "limit": list(mf.tolist()),
                "margins": list(margins.tolist()),
                "ok": fits,
            }
            if not fits:
                constraint_errors.append(
                    f"Does not fit within {mf.tolist()} {spec.units}. Actual size {size.tolist()} {spec.units}."
                )

        # target_overall_size
        if constraints.target_overall_size is not None:
            tgt = np.array(constraints.target_overall_size, dtype=float)
            tol = float(constraints.overall_size_tolerance)
            deltas = size - tgt
            within = bool(np.all(np.abs(deltas) <= tol + 1e-9))
            c_report["target_overall_size"] = {
                "target": list(tgt.tolist()),
                "tolerance": tol,
                "deltas": list(deltas.tolist()),
                "ok": within,
            }
            if not within:
                constraint_errors.append(
                    f"Overall size differs from target {tgt.tolist()} ±{tol} {spec.units}. Deltas: {deltas.tolist()}."
                )

        # Apply export-only effects for checks that depend on final hole size
        eff_spec, effects = apply_export_constraints(spec, constraints)
        c_effects = effects
        if effects:
            c_report["export_effects"] = effects

        # min_feature_size
        if constraints.min_feature_size is not None:
            min_req = float(constraints.min_feature_size)
            actual_min = float("inf")
            # scan solids
            for s in eff_spec.solids:
                if isinstance(s, BoxSolid):
                    actual_min = min(actual_min, float(min(s.size)))
                elif isinstance(s, CylinderZSolid):
                    actual_min = min(actual_min, float(2.0 * s.radius), float(s.height))
            if actual_min == float("inf"):
                actual_min = 0.0
            ok = bool(actual_min >= min_req - 1e-9)
            c_report["min_feature_size"] = {"required": min_req, "actual_min": actual_min, "ok": ok}
            if not ok:
                constraint_errors.append(
                    f"Minimum feature size violated: required >= {min_req} {spec.units}, but found ~{actual_min:.3g} {spec.units}."
                )

        # min_wall_thickness (MVP heuristic)
        if constraints.min_wall_thickness is not None:
            min_wall_req = float(constraints.min_wall_thickness)
            base, holes, reason = _try_extract_base_and_holes(eff_spec)
            if base is None or not holes:
                c_report["min_wall_thickness"] = {
                    "required": min_wall_req,
                    "supported": False,
                    "reason": reason,
                }
                constraint_warnings.append(
                    "min_wall_thickness check not supported for this geometry (MVP supports box-minus-holes pattern only)."
                )
            else:
                rep = _wall_thickness_report(base, holes)
                rep["required"] = min_wall_req
                rep["ok"] = bool(float(rep["min_wall_actual"]) >= min_wall_req - 1e-9)
                c_report["min_wall_thickness"] = rep
                if not rep["ok"]:
                    constraint_errors.append(
                        f"Minimum wall thickness violated: required >= {min_wall_req} {spec.units}, actual ~{rep['min_wall_actual']:.3g} {spec.units}."
                    )

        # printability flag is reported; actual metric comes from mesh export
        c_report["must_be_printable"] = {
            "enabled": bool(constraints.must_be_printable),
            "max_overhang_angle_deg": float(constraints.max_overhang_angle_deg),
        }

    return {
        "errors": errors,
        "warnings": warnings,
        "bounds": {"min": b.min_xyz.tolist(), "max": b.max_xyz.tolist(), "size": size.tolist()},
        "estimated_grid": {"nx": nx, "ny": ny, "nz": nz, "voxels": voxels},
        "constraint_errors": constraint_errors,
        "constraint_warnings": constraint_warnings,
        "constraints_report": c_report,
        "constraints_effects": c_effects,
    }
