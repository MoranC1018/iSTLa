from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


Vec3 = Tuple[float, float, float]
Units = Literal["mm", "cm", "in"]


class Constraints(BaseModel):
    """Session-level constraints that the generated part should obey.

    These are *product requirements*, not part of the PartSpec itself.
    We store them on the session and enforce them during validation/approval/finalize.
    """

    preferred_units: Units = "mm"

    # Hard envelope constraint (overall bounding box must be <= these dimensions)
    must_fit_within: Optional[Vec3] = None

    # If provided, overall size must match within tolerance.
    target_overall_size: Optional[Vec3] = None
    overall_size_tolerance: float = Field(0.2, ge=0.0)

    # Manufacturing / print constraints
    min_wall_thickness: Optional[float] = Field(None, gt=0.0)
    min_feature_size: Optional[float] = Field(None, gt=0.0)

    # Export-only compensation for holes. Positive => enlarge holes, negative => shrink.
    hole_diameter_offset: float = 0.0

    must_be_printable: bool = False
    max_overhang_angle_deg: float = Field(45.0, ge=0.0, le=90.0)

    notes: Optional[str] = None

    @field_validator("must_fit_within")
    @classmethod
    def _must_fit_positive(cls, v: Optional[Vec3]) -> Optional[Vec3]:
        if v is None:
            return v
        if any(x <= 0 for x in v):
            raise ValueError("must_fit_within must be > 0 in all dimensions")
        return v

    @field_validator("target_overall_size")
    @classmethod
    def _target_positive(cls, v: Optional[Vec3]) -> Optional[Vec3]:
        if v is None:
            return v
        if any(x <= 0 for x in v):
            raise ValueError("target_overall_size must be > 0 in all dimensions")
        return v


class ConstraintsPatch(BaseModel):
    """Partial update payload for constraints."""

    preferred_units: Optional[Units] = None
    must_fit_within: Optional[Vec3] = None
    target_overall_size: Optional[Vec3] = None
    overall_size_tolerance: Optional[float] = Field(None, ge=0.0)
    min_wall_thickness: Optional[float] = Field(None, gt=0.0)
    min_feature_size: Optional[float] = Field(None, gt=0.0)
    hole_diameter_offset: Optional[float] = None
    must_be_printable: Optional[bool] = None
    max_overhang_angle_deg: Optional[float] = Field(None, ge=0.0, le=90.0)
    notes: Optional[str] = None

    def apply_to(self, base: Constraints) -> Constraints:
        """Apply this patch to a base Constraints object.

        Unlike a naive model_dump(exclude_none=True), this preserves explicit
        nulls so the UI can clear optional fields (e.g. set must_fit_within = null).
        """
        data = {}
        for field in self.model_fields_set:
            data[field] = getattr(self, field)
        return base.model_copy(update=data)
