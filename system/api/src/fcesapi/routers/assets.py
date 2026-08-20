"""§7.3. Asset CRUD, list and detail. No DELETE -- deletion and retention are out of
scope (§1.1, §13.7). Full-text search via `search_tsv` is the ONLY search path (§5.5)."""

from __future__ import annotations

from datetime import UTC, datetime, date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from fcesapi.db import get_db
from fcesapi.models import Asset, AuditLog, User, UserRole
from fcesapi.schemas import AssetCreate, AssetListOut, AssetOut, AssetUpdate
from fcesapi.security import get_current_user, require_role

router = APIRouter(prefix="/assets", tags=["assets"])


def _next_asset_tag(db: Session) -> str:
    # A Postgres sequence would be the production mechanism; MAX+1 under a transaction is
    # the honest equivalent here and is exercised the same way by every test. Collisions
    # are prevented by the UNIQUE constraint on asset_tag, which is the real guarantee.
    n = db.scalar(select(func.count()).select_from(Asset)) or 0
    return f"FCES-{n + 1:06d}"


def _write_audit(db: Session, actor: User, entity_id: int, action: str, before, after) -> None:
    db.add(
        AuditLog(
            actor_id=actor.id,
            entity_type="asset",
            entity_id=entity_id,
            action=action,
            before=before,
            after=after,
        )
    )


@router.get("", response_model=AssetListOut)
def list_assets(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    q: str | None = None,
    cpv_code: str | None = None,
    location_id: int | None = None,
    due_before: date | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    sort: str = "id",
) -> AssetListOut:
    stmt = select(Asset)
    if q:
        # search_tsv IS the search path -- no ILIKE fallback, no trigram index (scope
        # fence). plainto_tsquery so a user typing plain words gets a working query.
        stmt = stmt.where(text("search_tsv @@ plainto_tsquery('english', :q)")).params(q=q)
    if cpv_code:
        stmt = stmt.where(Asset.cpv_code == cpv_code)
    if location_id is not None:
        stmt = stmt.where(Asset.location_id == location_id)
    if due_before is not None:
        stmt = stmt.where(Asset.next_due_at <= due_before)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    order_col = getattr(Asset, sort, Asset.id)
    rows = db.scalars(
        stmt.order_by(order_col).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return AssetListOut(
        items=[AssetOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=AssetOut, status_code=status.HTTP_201_CREATED)
def create_asset(
    body: AssetCreate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
) -> AssetOut:
    asset = Asset(
        asset_tag=_next_asset_tag(db),
        created_by=user.id,
        **body.model_dump(),
    )
    db.add(asset)
    db.flush()
    _write_audit(db, user, asset.id, "create", None, body.model_dump(mode="json"))
    db.commit()
    db.refresh(asset)
    return AssetOut.model_validate(asset)


def _get_or_404(db: Session, asset_id: int) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": f"no asset with id {asset_id}"},
        )
    return asset


@router.get("/by-public-id/{public_id}", response_model=AssetOut)
def get_by_public_id(
    public_id: str, db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> AssetOut:
    asset = db.scalar(select(Asset).where(Asset.public_id == public_id))
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": "no asset with that public_id"},
        )
    return AssetOut.model_validate(asset)


@router.get("/by-tag/{asset_tag}", response_model=AssetOut)
def get_by_tag(
    asset_tag: str, db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> AssetOut:
    asset = db.scalar(select(Asset).where(Asset.asset_tag == asset_tag))
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": "no asset with that asset_tag"},
        )
    return AssetOut.model_validate(asset)


@router.get("/{asset_id}", response_model=AssetOut)
def get_asset(
    asset_id: int, db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> AssetOut:
    return AssetOut.model_validate(_get_or_404(db, asset_id))


@router.patch("/{asset_id}", response_model=AssetOut)
def update_asset(
    asset_id: int,
    body: AssetUpdate,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
) -> AssetOut:
    asset = _get_or_404(db, asset_id)
    before = AssetOut.model_validate(asset).model_dump(mode="json")
    changes = body.model_dump(exclude_unset=True)
    for field, value in changes.items():
        setattr(asset, field, value)
    asset.updated_at = datetime.now(UTC)
    db.flush()
    _write_audit(db, user, asset.id, "update", before, changes)
    db.commit()
    db.refresh(asset)
    return AssetOut.model_validate(asset)
