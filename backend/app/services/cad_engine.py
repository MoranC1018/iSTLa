from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Any, Optional

import numpy as np
import trimesh
from skimage import measure

from backend.app.models.partspec import PartSpec, Solid, BoxSolid, CylinderZSolid, Operation


@dataclass(frozen=True)
class Bounds:
    min_xyz: np.ndarray  # shape (3,)
    max_xyz: np.ndarray  # shape (3,)

    def padded(self, pad: float) -> "Bounds":
        return Bounds(self.min_xyz - pad, self.max_xyz + pad)

    @property
    def size(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz


def _bounds_box(s: BoxSolid) -> Bounds:
    c = np.array(s.center, dtype=float)
    size = np.array(s.size, dtype=float)
    half = size / 2.0
    return Bounds(c - half, c + half)


def _bounds_cylinder_z(s: CylinderZSolid) -> Bounds:
    c = np.array(s.center, dtype=float)
    r = float(s.radius)
    h = float(s.height)
    half = np.array([r, r, h / 2.0], dtype=float)
    return Bounds(c - half, c + half)


def compute_bounds(spec: PartSpec) -> Bounds:
    """Compute an axis-aligned bounding box for the final id using primitive + op bounds.

    Notes:
    - This is exact for axis-aligned boxes/cylinders, and exact for union/diff given those.
    - For intersection, it's still correct as a bounding box but may be loose in edge cases.
    """
    solid_map: Dict[str, Solid] = {s.id: s for s in spec.solids}
    op_map: Dict[str, Operation] = {op.id: op for op in spec.operations}

    bounds_cache: Dict[str, Bounds] = {}

    def get_bounds(node_id: str) -> Bounds:
        if node_id in bounds_cache:
            return bounds_cache[node_id]

        if node_id in solid_map:
            s = solid_map[node_id]
            if isinstance(s, BoxSolid):
                b = _bounds_box(s)
            elif isinstance(s, CylinderZSolid):
                b = _bounds_cylinder_z(s)
            else:
                raise TypeError(f"Unsupported solid type: {type(s)}")
            bounds_cache[node_id] = b
            return b

        if node_id in op_map:
            op = op_map[node_id]
            a = get_bounds(op.a)
            if op.op == "difference":
                # difference can't expand beyond A
                b = a
            else:
                # union / intersection: combine bounds
                if not op.b:
                    b = a
                else:
                    others = [get_bounds(x) for x in op.b]
                    mins = np.vstack([a.min_xyz] + [o.min_xyz for o in others]).min(axis=0)
                    maxs = np.vstack([a.max_xyz] + [o.max_xyz for o in others]).max(axis=0)
                    if op.op == "intersection":
                        # tighter intersection bounds: overlap of A with others
                        mins = np.vstack([a.min_xyz] + [o.min_xyz for o in others]).max(axis=0)
                        maxs = np.vstack([a.max_xyz] + [o.max_xyz for o in others]).min(axis=0)
                    b = Bounds(mins, maxs)
            bounds_cache[node_id] = b
            return b

        raise KeyError(f"Unknown id '{node_id}'")

    return get_bounds(spec.final)


# -----------------------
# Signed distance fields
# -----------------------

def sdf_box_xyz(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: Tuple[float, float, float], size: Tuple[float, float, float]) -> np.ndarray:
    cx, cy, cz = center
    sx, sy, sz = size
    # axis-aligned box SDF (Inigo Quilez)
    qx = np.abs(X - cx) - sx / 2.0
    qy = np.abs(Y - cy) - sy / 2.0
    qz = np.abs(Z - cz) - sz / 2.0

    # outside distance
    ox = np.maximum(qx, 0.0)
    oy = np.maximum(qy, 0.0)
    oz = np.maximum(qz, 0.0)
    outside = np.sqrt(ox * ox + oy * oy + oz * oz)

    # inside distance
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return outside + inside


def sdf_cylinder_z_xyz(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, center: Tuple[float, float, float], radius: float, height: float) -> np.ndarray:
    cx, cy, cz = center
    # finite cylinder along Z
    dx = X - cx
    dy = Y - cy
    dz = Z - cz
    d_xy = np.sqrt(dx * dx + dy * dy) - radius
    d_z = np.abs(dz) - height / 2.0

    # combine (Inigo Quilez)
    ax = np.maximum(d_xy, 0.0)
    ay = np.maximum(d_z, 0.0)
    outside = np.sqrt(ax * ax + ay * ay)
    inside = np.minimum(np.maximum(d_xy, d_z), 0.0)
    return outside + inside


def build_sdf_grid(spec: PartSpec, *, bounds: Optional[Bounds] = None) -> tuple[np.ndarray, Bounds, float]:
    """Return (sdf_grid, bounds_used, pitch) for the final object."""
    solid_map: Dict[str, Solid] = {s.id: s for s in spec.solids}
    op_map: Dict[str, Operation] = {op.id: op for op in spec.operations}

    if bounds is None:
        bounds = compute_bounds(spec)
    bounds = bounds.padded(spec.tessellation.padding)

    pitch = float(spec.tessellation.grid_pitch)
    min_x, min_y, min_z = bounds.min_xyz
    max_x, max_y, max_z = bounds.max_xyz

    # build grid axes
    xs = np.arange(min_x, max_x + pitch * 0.5, pitch, dtype=np.float32)
    ys = np.arange(min_y, max_y + pitch * 0.5, pitch, dtype=np.float32)
    zs = np.arange(min_z, max_z + pitch * 0.5, pitch, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    voxels = nx * ny * nz
    if voxels > spec.tessellation.max_voxels:
        raise ValueError(
            f"Grid too large: {voxels:,} voxels (nx={nx}, ny={ny}, nz={nz}). "
            f"Increase grid_pitch or reduce part size; max_voxels={spec.tessellation.max_voxels:,}."
        )

    # use broadcasting grids to avoid storing full meshgrid coords separately
    X = xs[:, None, None]
    Y = ys[None, :, None]
    Z = zs[None, None, :]

    # cache grid SDFs per node id
    grid_cache: Dict[str, np.ndarray] = {}

    def get_grid(node_id: str) -> np.ndarray:
        if node_id in grid_cache:
            return grid_cache[node_id]

        if node_id in solid_map:
            s = solid_map[node_id]
            if isinstance(s, BoxSolid):
                g = sdf_box_xyz(X, Y, Z, s.center, s.size).astype(np.float32, copy=False)
            elif isinstance(s, CylinderZSolid):
                g = sdf_cylinder_z_xyz(X, Y, Z, s.center, float(s.radius), float(s.height)).astype(np.float32, copy=False)
            else:
                raise TypeError(f"Unsupported solid type: {type(s)}")
            grid_cache[node_id] = g
            return g

        if node_id in op_map:
            op = op_map[node_id]
            a = get_grid(op.a)
            if not op.b:
                g = a
            else:
                bs = [get_grid(x) for x in op.b]
                if op.op == "union":
                    g = np.minimum.reduce([a, *bs])
                elif op.op == "intersection":
                    g = np.maximum.reduce([a, *bs])
                elif op.op == "difference":
                    # subtract union of B from A
                    union_b = np.minimum.reduce(bs)
                    g = np.maximum(a, -union_b)
                else:
                    raise ValueError(f"Unknown op '{op.op}'")
            grid_cache[node_id] = g.astype(np.float32, copy=False)
            return grid_cache[node_id]

        raise KeyError(f"Unknown id '{node_id}'")

    sdf_grid = get_grid(spec.final)
    return sdf_grid, bounds, pitch


def mesh_from_partspec(spec: PartSpec) -> trimesh.Trimesh:
    sdf_grid, bounds_used, pitch = build_sdf_grid(spec)

    # marching cubes expects axis order (x,y,z) as we constructed it
    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid,
        level=0.0,
        spacing=(pitch, pitch, pitch),
        allow_degenerate=False
    )

    # shift to world coords by adding min bounds (origin of the grid)
    verts = verts + bounds_used.min_xyz.astype(np.float32)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    # basic cleanup
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.merge_vertices()

    return mesh



def _printability_report(mesh: trimesh.Trimesh, max_overhang_angle_deg: float) -> dict:
    """Very fast printability heuristic.

    We approximate "support-needed" area as downward-facing triangles whose normals are
    more horizontal than the allowed overhang angle.

    - Build direction is +Z.
    - We only consider downward faces (normal_z < 0).
    - If |normal_z| < cos(max_overhang_angle_deg), we mark it as needing support.

    This is not slicer-accurate, but it's a useful MVP signal.
    """
    try:
        normals = mesh.face_normals
        areas = mesh.area_faces
    except Exception:
        return {}

    if normals is None or areas is None or len(areas) == 0:
        return {}

    nz = normals[:, 2]
    down = nz < 0

    import numpy as _np

    thresh = float(_np.cos(_np.deg2rad(float(max_overhang_angle_deg))))
    needs_support = down & (_np.abs(nz) < thresh)

    total_area = float(mesh.area) if hasattr(mesh, 'area') else float(_np.sum(areas))
    downward_area = float(_np.sum(areas[down]))
    support_area = float(_np.sum(areas[needs_support]))

    return {
        "max_overhang_angle_deg": float(max_overhang_angle_deg),
        "total_surface_area": total_area,
        "downward_surface_area": downward_area,
        "support_surface_area_estimate": support_area,
        "support_fraction_of_total_area": (support_area / total_area) if total_area > 0 else 0.0,
        "support_fraction_of_downward_area": (support_area / downward_area) if downward_area > 0 else 0.0,
    }



def export_stl(spec: PartSpec, out_path: str | Path, *, max_overhang_angle_deg: float | None = None) -> dict:
    """Build and export STL. Returns a report.

    Args:
        max_overhang_angle_deg: if provided, include a printability heuristic report.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = mesh_from_partspec(spec)

    # Export STL
    mesh.export(out_path.as_posix(), file_type="stl")

    report = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "is_watertight": bool(mesh.is_watertight),
        "bounds": mesh.bounds.tolist(),  # [[minx,miny,minz],[maxx,maxy,maxz]]
        "extents": mesh.extents.tolist(),
    }

    if max_overhang_angle_deg is not None:
        report["printability"] = _printability_report(mesh, float(max_overhang_angle_deg))

    return report

