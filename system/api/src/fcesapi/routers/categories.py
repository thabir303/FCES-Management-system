"""§7.9. No user administration UI, no audit viewer, no /users -- scope fence (§1.1)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.db import get_db
from fcesapi.models import Category, User, UserRole
from fcesapi.security import get_current_user, require_role

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("")
def list_categories(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    level: int | None = None,
    q: str | None = None,
):
    stmt = select(Category)
    if level is not None:
        stmt = stmt.where(Category.level == level)
    if q:
        stmt = stmt.where(Category.cpv_description.ilike(f"%{q}%"))
    rows = db.scalars(stmt).all()
    return [
        {
            "cpv_code": r.cpv_code, "cpv_description": r.cpv_description, "level": r.level,
            "parent_code": r.parent_code, "hazard_class": r.hazard_class,
            "default_service_interval_days": r.default_service_interval_days,
        }
        for r in rows
    ]


@router.patch("/{code}")
def update_category(
    code: str,
    body: dict,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.admin))],
):
    cat = db.get(Category, code)
    if cat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "category_not_found", "message": f"no category {code}"},
        )
    for field in ("hazard_class", "default_service_interval_days"):
        if field in body:
            setattr(cat, field, body[field])
    db.commit()
    db.refresh(cat)
    return {"cpv_code": cat.cpv_code, "hazard_class": cat.hazard_class,
            "default_service_interval_days": cat.default_service_interval_days}
