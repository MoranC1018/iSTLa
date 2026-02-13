from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import svgwrite

from backend.app.models.partspec import PartSpec, BoxSolid, CylinderZSolid


@dataclass(frozen=True)
class ViewRect:
    x: float
    y: float
    w: float
    h: float


def _find_base_and_holes(spec: PartSpec) -> tuple[BoxSolid, List[CylinderZSolid]]:
    """Heuristic MVP extractor:
    Expect final op is difference(base_box, cylinders...)."""
    op = next((o for o in spec.operations if o.id == spec.final), None)
    if op is None or op.op != "difference":
        raise ValueError("Drawing MVP expects final operation to be a 'difference' op.")

    solid_map = {s.id: s for s in spec.solids}
    base = solid_map.get(op.a)
    if not isinstance(base, BoxSolid):
        raise ValueError("Drawing MVP expects op.a to be a box base.")

    holes: List[CylinderZSolid] = []
    for bid in op.b:
        s = solid_map.get(bid)
        if isinstance(s, CylinderZSolid):
            holes.append(s)

    return base, holes


def _add_arrow_markers(dwg: svgwrite.Drawing) -> None:
    # Simple triangular arrow markers
    marker_end = dwg.marker(insert=(3, 3), size=(6, 6), orient="auto", id="arrow_end")
    marker_end.add(dwg.path(d="M0,0 L6,3 L0,6 Z", fill="black"))
    dwg.defs.add(marker_end)

    marker_start = dwg.marker(insert=(3, 3), size=(6, 6), orient="auto", id="arrow_start")
    marker_start.add(dwg.path(d="M6,0 L0,3 L6,6 Z", fill="black"))
    dwg.defs.add(marker_start)

    marker_dot = dwg.marker(insert=(2, 2), size=(4, 4), orient="auto", id="dot")
    marker_dot.add(dwg.circle(center=(2, 2), r=2, fill="black"))
    dwg.defs.add(marker_dot)


def _dim_line(
    dwg: svgwrite.Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    text: str,
    text_pos: Tuple[float, float],
) -> None:
    dwg.add(
        dwg.line(
            start=(x1, y1),
            end=(x2, y2),
            stroke="black",
            stroke_width=1,
            marker_start="url(#arrow_start)",
            marker_end="url(#arrow_end)",
        )
    )
    dwg.add(dwg.text(text, insert=text_pos, font_size=12, font_family="Arial"))


def generate_orthographic_svg(
    spec: PartSpec,
    out_path: str | Path,
    *,
    hole_export_diameters: Optional[Dict[str, float]] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a dimensioned orthographic drawing as SVG for MVP parts.

    MVP supported geometry:
    - base: box
    - features: cylindrical cut holes along Z (from difference op)

    Optionally:
    - hole_export_diameters: mapping hole_id -> export diameter to display as "Øx (export Øy)"
    - notes: lines displayed under a NOTES section

    Returns metadata (sizes, hole table).
    """
    base, holes = _find_base_and_holes(spec)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hole_export_diameters = hole_export_diameters or {}
    notes = notes or []

    # World dims
    cx, cy, cz = base.center
    sx, sy, sz = base.size
    min_x, max_x = cx - sx / 2, cx + sx / 2
    min_y, max_y = cy - sy / 2, cy + sy / 2
    min_z, max_z = cz - sz / 2, cz + sz / 2

    # Canvas
    canvas_w = 1200
    canvas_h = 520
    margin = 30
    gap = 50

    # Choose scale so largest dimension fits ~300px
    max_dim = max(sx, sy, sz)
    scale = 300.0 / max_dim if max_dim > 0 else 1.0

    # View rectangles on canvas
    top_view = ViewRect(x=margin, y=margin, w=sx * scale, h=sy * scale)
    front_view = ViewRect(x=margin + top_view.w + gap, y=margin, w=sx * scale, h=sz * scale)
    right_view = ViewRect(x=margin + top_view.w + gap + front_view.w + gap, y=margin, w=sy * scale, h=sz * scale)

    dwg = svgwrite.Drawing(out_path.as_posix(), size=(canvas_w, canvas_h))
    _add_arrow_markers(dwg)

    # Titles
    dwg.add(dwg.text(spec.name or "Part", insert=(margin, 20), font_size=16, font_family="Arial", font_weight="bold"))
    dwg.add(dwg.text(f"Units: {spec.units}", insert=(canvas_w - 130, 20), font_size=12, font_family="Arial"))

    def w2s_top(x: float, y: float) -> Tuple[float, float]:
        sxp = top_view.x + (x - min_x) * scale
        syp = top_view.y + (y - min_y) * scale
        return sxp, syp

    def w2s_front(x: float, z: float) -> Tuple[float, float]:
        sxp = front_view.x + (x - min_x) * scale
        syp = front_view.y + (max_z - z) * scale  # flip so +z up
        return sxp, syp

    def w2s_right(y: float, z: float) -> Tuple[float, float]:
        sxp = right_view.x + (y - min_y) * scale
        syp = right_view.y + (max_z - z) * scale
        return sxp, syp

    # ---- Draw top view ----
    dwg.add(dwg.rect(insert=(top_view.x, top_view.y), size=(top_view.w, top_view.h), fill="none", stroke="black"))
    dwg.add(dwg.text("TOP", insert=(top_view.x, top_view.y + top_view.h + 20), font_size=12, font_family="Arial"))

    # holes in top view
    for i, h in enumerate(holes, start=1):
        hx, hy, _ = h.center
        (cxp, cyp) = w2s_top(hx, hy)
        dwg.add(dwg.circle(center=(cxp, cyp), r=h.radius * scale, fill="none", stroke="black"))
        dwg.add(dwg.text(f"H{i}", insert=(cxp + 4, cyp - 4), font_size=10, font_family="Arial"))

    # ---- Draw front view ----
    dwg.add(dwg.rect(insert=(front_view.x, front_view.y), size=(front_view.w, front_view.h), fill="none", stroke="black"))
    dwg.add(dwg.text("FRONT", insert=(front_view.x, front_view.y + front_view.h + 20), font_size=12, font_family="Arial"))

    # ---- Draw right view ----
    dwg.add(dwg.rect(insert=(right_view.x, right_view.y), size=(right_view.w, right_view.h), fill="none", stroke="black"))
    dwg.add(dwg.text("RIGHT", insert=(right_view.x, right_view.y + right_view.h + 20), font_size=12, font_family="Arial"))

    # ---- Dimensioning (overall) ----
    dim_off = 25
    # Width on top view
    x1, y1 = w2s_top(min_x, max_y)
    x2, y2 = w2s_top(max_x, max_y)
    _dim_line(
        dwg,
        x1, y1 + dim_off,
        x2, y2 + dim_off,
        text=f"{sx:.3g}",
        text_pos=((x1 + x2) / 2 - 10, y1 + dim_off - 5),
    )
    # Depth on top view
    x1, y1 = w2s_top(max_x, min_y)
    x2, y2 = w2s_top(max_x, max_y)
    _dim_line(
        dwg,
        x1 + dim_off, y1,
        x2 + dim_off, y2,
        text=f"{sy:.3g}",
        text_pos=(x1 + dim_off + 5, (y1 + y2) / 2),
    )

    # Height on front view
    x1, y1 = w2s_front(max_x, min_z)
    x2, y2 = w2s_front(max_x, max_z)
    _dim_line(
        dwg,
        x1 + dim_off, y1,
        x2 + dim_off, y2,
        text=f"{sz:.3g}",
        text_pos=(x1 + dim_off + 5, (y1 + y2) / 2),
    )

    # ---- Hole callout table ----
    table_x = margin
    table_y = canvas_h - 170
    dwg.add(dwg.text("HOLES:", insert=(table_x, table_y), font_size=12, font_family="Arial", font_weight="bold"))
    row_y = table_y + 18

    hole_rows: List[Dict[str, Any]] = []
    for i, h in enumerate(holes, start=1):
        hx, hy, _ = h.center
        x_off = hx - min_x
        y_off = hy - min_y
        dia = 2 * h.radius
        export_dia = hole_export_diameters.get(h.id)

        hole_rows.append({
            "id": f"H{i}",
            "solid_id": h.id,
            "diameter_nominal": float(dia),
            "diameter_export": float(export_dia) if export_dia is not None else None,
            "x_from_left": float(x_off),
            "y_from_bottom": float(y_off),
        })

        if export_dia is not None and abs(float(export_dia) - float(dia)) > 1e-9:
            dia_text = f"Ø{dia:.3g} (export Ø{float(export_dia):.3g})"
        else:
            dia_text = f"Ø{dia:.3g}"

        dwg.add(
            dwg.text(
                f"H{i}: {dia_text} at ({x_off:.3g}, {y_off:.3g}) from (left, bottom)",
                insert=(table_x, row_y),
                font_size=11,
                font_family="Arial",
            )
        )
        row_y += 16

    # ---- Notes ----
    if notes:
        row_y += 10
        dwg.add(dwg.text("NOTES:", insert=(table_x, row_y), font_size=12, font_family="Arial", font_weight="bold"))
        row_y += 18
        for n in notes:
            dwg.add(dwg.text(str(n), insert=(table_x, row_y), font_size=11, font_family="Arial"))
            row_y += 16

    dwg.save()
    return {
        "overall": {"width": float(sx), "depth": float(sy), "height": float(sz), "units": spec.units},
        "holes": hole_rows,
        "notes": notes,
        "scale_px_per_unit": float(scale),
    }
