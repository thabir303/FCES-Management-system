"""§5b. Floor plans and pins (D-series extension, §5.6).

A "pin" is a `Location` row carrying `floorplan_id` + percentage coordinates rather than a
separate table -- `locations` already exists precisely so an asset's `location_id` can point
either at a plain building/floor/room record or at a plotted pin, without two parallel
concepts of "where an asset is." Percentage coordinates (`x_pct`/`y_pct`, already
CHECK-constrained to [0, 100] in the migration) survive the plan image being re-exported at a
different resolution -- pixel coordinates would not.
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.config import get_settings
from fcesapi.db import get_db
from fcesapi.models import Asset, FloorPlan, Location, User, UserRole
from fcesapi.schemas import FloorPlanOut, PinCreate, PinOut, PinUpdate
from fcesapi.security import get_current_user, require_role

router = APIRouter(tags=["floorplans"])


def _get_floorplan_or_404(db: Session, floorplan_id: int) -> FloorPlan:
    plan = db.get(FloorPlan, floorplan_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "floorplan_not_found", "message": f"no floor plan with id {floorplan_id}"},
        )
    return plan


def _get_pin_or_404(db: Session, floorplan_id: int, pin_id: int) -> Location:
    pin = db.get(Location, pin_id)
    if pin is None or pin.floorplan_id != floorplan_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "pin_not_found",
                "message": f"no pin {pin_id} on floor plan {floorplan_id}",
            },
        )
    return pin


@router.post("/floorplans", response_model=FloorPlanOut, status_code=status.HTTP_201_CREATED)
def upload_floorplan(
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
    building: Annotated[str, Form()],
    floor: Annotated[str, Form()],
    image: UploadFile,
    name: Annotated[str | None, Form()] = None,
) -> FloorPlanOut:
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "wrong_mime",
                "message": f"floor plan upload requires an image/* file, got {image.content_type!r}",
            },
        )
    contents = image.file.read()
    dims = Image.open(io.BytesIO(contents)).size  # (width, height)

    d = Path(get_settings().storage_root) / "floorplans"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{uuid.uuid4().hex}_{image.filename or 'plan'}"
    dest.write_bytes(contents)

    plan = FloorPlan(
        building=building, floor=floor, name=name,
        image_path=str(dest), image_w=dims[0], image_h=dims[1], uploaded_by=user.id,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return FloorPlanOut.model_validate(plan)


@router.get("/floorplans", response_model=list[FloorPlanOut])
def list_floorplans(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> list[FloorPlanOut]:
    rows = db.scalars(select(FloorPlan).order_by(FloorPlan.id)).all()
    return [FloorPlanOut.model_validate(r) for r in rows]


@router.post(
    "/floorplans/{floorplan_id}/pins", response_model=PinOut, status_code=status.HTTP_201_CREATED
)
def create_pin(
    floorplan_id: int,
    body: PinCreate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.technician))],
) -> PinOut:
    plan = _get_floorplan_or_404(db, floorplan_id)
    if not (0 <= body.x_pct <= 100 and 0 <= body.y_pct <= 100):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "coordinate_out_of_range", "message": "x_pct and y_pct must be within 0-100"},
        )
    pin = Location(
        building=plan.building, floor=plan.floor, label=body.label,
        floorplan_id=plan.id, x_pct=body.x_pct, y_pct=body.y_pct,
    )
    db.add(pin)
    db.flush()
    if body.asset_id is not None:
        asset = db.get(Asset, body.asset_id)
        if asset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "asset_not_found", "message": f"no asset with id {body.asset_id}"},
            )
        asset.location_id = pin.id
    db.commit()
    db.refresh(pin)
    return PinOut.model_validate(pin)


@router.get("/floorplans/{floorplan_id}/pins", response_model=list[PinOut])
def list_pins(
    floorplan_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> list[PinOut]:
    _get_floorplan_or_404(db, floorplan_id)
    rows = db.scalars(
        select(Location).where(Location.floorplan_id == floorplan_id).order_by(Location.id)
    ).all()
    return [PinOut.model_validate(r) for r in rows]


@router.patch("/floorplans/{floorplan_id}/pins/{pin_id}", response_model=PinOut)
def move_pin(
    floorplan_id: int,
    pin_id: int,
    body: PinUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.technician))],
) -> PinOut:
    pin = _get_pin_or_404(db, floorplan_id, pin_id)
    changes = body.model_dump(exclude_unset=True, exclude={"asset_id"})
    for field, value in changes.items():
        if field in ("x_pct", "y_pct") and value is not None and not (0 <= value <= 100):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "coordinate_out_of_range", "message": f"{field} must be within 0-100"},
            )
        setattr(pin, field, value)
    if "asset_id" in body.model_fields_set:
        if body.asset_id is not None:
            asset = db.get(Asset, body.asset_id)
            if asset is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"code": "asset_not_found", "message": f"no asset with id {body.asset_id}"},
                )
            asset.location_id = pin.id
    db.commit()
    db.refresh(pin)
    return PinOut.model_validate(pin)


@router.delete("/floorplans/{floorplan_id}/pins/{pin_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_pin(
    floorplan_id: int,
    pin_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.technician))],
) -> None:
    pin = _get_pin_or_404(db, floorplan_id, pin_id)
    # Any asset pointing at this pin is set to location_id=NULL by the FK's own
    # ON DELETE SET NULL (models.py/migration) -- no orphaned reference survives this delete.
    db.delete(pin)
    db.commit()
