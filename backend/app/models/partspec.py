from __future__ import annotations

from typing import Literal, Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, model_validator, field_validator


Vec3 = Tuple[float, float, float]


class Tessellation(BaseModel):
    grid_pitch: float = Field(..., gt=0, description="Sampling pitch in model units.")
    padding: float = Field(2.0, ge=0, description="Extra margin around bounds.")
    max_voxels: int = Field(5_000_000, ge=1000, description="Safety cap on grid size.")


class SolidBase(BaseModel):
    id: str = Field(..., min_length=1)
    type: Literal["box", "cylinder_z"]
    center: Vec3
    tags: List[str] = Field(default_factory=list)
    role: Literal["solid", "tool"] = "solid"
    rotation: Vec3 = (0.0, 0.0, 0.0)

    @field_validator("rotation")
    @classmethod
    def rotation_must_be_zero_for_mvp(cls, v: Vec3) -> Vec3:
        # Rotation support will come later. For now, enforce [0,0,0].
        if any(abs(x) > 1e-9 for x in v):
            raise ValueError("rotation is not implemented in MVP; must be [0,0,0].")
        return v


class BoxSolid(SolidBase):
    type: Literal["box"] = "box"
    size: Tuple[float, float, float] = Field(..., description="Box size [sx, sy, sz].")

    @field_validator("size")
    @classmethod
    def size_positive(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if any(x <= 0 for x in v):
            raise ValueError("box.size must be > 0 in all dimensions.")
        return v


class CylinderZSolid(SolidBase):
    type: Literal["cylinder_z"] = "cylinder_z"
    radius: float = Field(..., gt=0)
    height: float = Field(..., gt=0)


Solid = BoxSolid | CylinderZSolid


class Operation(BaseModel):
    id: str = Field(..., min_length=1)
    op: Literal["union", "difference", "intersection"]
    a: str
    b: List[str] = Field(default_factory=list)


class CoordinateSystem(BaseModel):
    convention: str = "x=width, y=depth, z=height"
    right_handed: bool = True
    origin: Optional[Vec3] = None


class PartSpec(BaseModel):
    version: Literal["1.0"]
    name: Optional[str] = None
    units: Literal["mm", "cm", "in"] = "mm"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    coordinate_system: Optional[CoordinateSystem] = None

    solids: List[Solid]
    operations: List[Operation]
    final: str

    tessellation: Tessellation

    @model_validator(mode="after")
    def check_references(self) -> "PartSpec":
        ids = {s.id for s in self.solids} | {op.id for op in self.operations}
        solid_ids = {s.id for s in self.solids}
        op_ids = {op.id for op in self.operations}

        # final must reference an operation or solid that exists
        if self.final not in ids:
            raise ValueError(f"final='{self.final}' does not reference a known solid/operation id.")

        # operations must reference existing ids
        for op in self.operations:
            if op.a not in ids:
                raise ValueError(f"Operation '{op.id}' references missing a='{op.a}'.")
            for b in op.b:
                if b not in ids:
                    raise ValueError(f"Operation '{op.id}' references missing b='{b}'.")

        # disallow operations that depend on themselves directly
        for op in self.operations:
            if op.id == op.a or op.id in op.b:
                raise ValueError(f"Operation '{op.id}' cannot reference itself.")

        # helpful: ensure no duplicate ids
        if len(ids) != (len(self.solids) + len(self.operations)):
            # find duplicates
            all_ids = [s.id for s in self.solids] + [op.id for op in self.operations]
            dups = {x for x in all_ids if all_ids.count(x) > 1}
            raise ValueError(f"Duplicate ids found: {sorted(dups)}")

        # basic MVP guardrail: require at least one operation
        if not self.operations:
            raise ValueError("At least one operation is required.")

        return self
