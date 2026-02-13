from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Tuple

from backend.app.models.constraints import Constraints
from backend.app.models.partspec import PartSpec, CylinderZSolid


_HOLE_TAGS = {"hole", "through_hole", "bore"}


def constraints_hash(constraints: Constraints) -> str:
    """Stable hash for a constraints object.

    Used to detect when constraints changed since a revision was generated/approved.
    """
    canon = json.dumps(constraints.model_dump(exclude_none=True), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def apply_export_constraints(spec: PartSpec, constraints: Optional[Constraints]) -> Tuple[PartSpec, Dict[str, Any]]:
    """Apply export-only constraints to a PartSpec.

    Currently implemented:
    - hole_diameter_offset: adjusts radius of tool cylinders tagged as holes.

    Returns (effective_spec, effects).
    """
    if constraints is None:
        return spec, {}

    offset = float(constraints.hole_diameter_offset or 0.0)
    if abs(offset) < 1e-12:
        return spec, {}

    hole_export_diameters: Dict[str, float] = {}
    new_solids = []

    for s in spec.solids:
        if isinstance(s, CylinderZSolid) and s.role == "tool":
            tags = {t.lower() for t in (s.tags or [])}
            if tags & _HOLE_TAGS:
                new_radius = float(s.radius) + offset / 2.0
                # Guard against non-positive radii
                if new_radius <= 0:
                    # If the offset would invert the hole, skip applying and let validation catch it.
                    new_solids.append(s)
                    continue
                if abs(new_radius - float(s.radius)) > 1e-12:
                    hole_export_diameters[s.id] = 2.0 * new_radius
                    new_solids.append(s.model_copy(update={"radius": new_radius}))
                    continue
        new_solids.append(s)

    if not hole_export_diameters:
        return spec, {}

    eff = spec.model_copy(update={"solids": new_solids})
    effects = {
        "hole_diameter_offset": offset,
        "hole_export_diameters": hole_export_diameters,
    }
    return eff, effects


def hole_export_diameters_from_effects(effects: Dict[str, Any]) -> Dict[str, float]:
    """Helper to pull hole export diameter mapping from an effects dict."""
    if not effects:
        return {}
    m = effects.get("hole_export_diameters")
    if isinstance(m, dict):
        # ensure float values
        out: Dict[str, float] = {}
        for k, v in m.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    return {}
