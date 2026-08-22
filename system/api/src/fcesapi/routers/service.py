"""§5d. Service events and the due/overdue view (D7).

Logging a service event moves `assets.last_serviced_at` forward, which recomputes
`next_due_at` at the database level (a GENERATED column) -- this router never writes
`next_due_at` itself, only the input the database derives it from.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.db import get_db
from fcesapi.models import Asset, ServiceEvent, User, UserRole
from fcesapi.schemas import AssetOut, ServiceEventCreate, ServiceEventOut
from fcesapi.security import get_current_user, require_role
from fcesapi.services.notifications import DEFAULT_DUE_SOON_DAYS

router = APIRouter(tags=["service"])


def _get_asset_or_404(db: Session, asset_id: int) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": f"no asset with id {asset_id}"},
        )
    return asset


@router.post(
    "/assets/{asset_id}/service-events",
    response_model=ServiceEventOut,
    status_code=status.HTTP_201_CREATED,
)
def log_service_event(
    asset_id: int,
    body: ServiceEventCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
) -> ServiceEventOut:
    asset = _get_asset_or_404(db, asset_id)
    event = ServiceEvent(
        asset_id=asset.id, performed_at=body.performed_at,
        performed_by=user.id, provider=body.provider, notes=body.notes,
    )
    db.add(event)
    if asset.last_serviced_at is None or body.performed_at > asset.last_serviced_at:
        asset.last_serviced_at = body.performed_at
    db.commit()
    db.refresh(event)
    return ServiceEventOut.model_validate(event)


@router.get("/assets/{asset_id}/service-events", response_model=list[ServiceEventOut])
def list_service_events(
    asset_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> list[ServiceEventOut]:
    _get_asset_or_404(db, asset_id)
    rows = db.scalars(
        select(ServiceEvent).where(ServiceEvent.asset_id == asset_id).order_by(ServiceEvent.performed_at.desc())
    ).all()
    return [ServiceEventOut.model_validate(r) for r in rows]


@router.get("/service/due")
def service_due(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    window_days: Annotated[int, Query(ge=1, le=365)] = DEFAULT_DUE_SOON_DAYS,
) -> dict:
    today = date.today()
    horizon = today + timedelta(days=window_days)
    rows = db.scalars(
        select(Asset)
        .where(Asset.next_due_at.isnot(None), Asset.next_due_at <= horizon)
        .order_by(Asset.next_due_at)
    ).all()
    overdue = [AssetOut.model_validate(a) for a in rows if a.next_due_at < today]
    due_soon = [AssetOut.model_validate(a) for a in rows if a.next_due_at >= today]
    return {"overdue": overdue, "due_soon": due_soon}
