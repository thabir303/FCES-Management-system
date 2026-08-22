"""§5e. Audit log query endpoint (D9's read side).

D9 wrote the log; nothing has ever read it back. This is an API query endpoint, not a
browsing UI -- `.claude/rules/system.md`'s scope fence names "an audit browsing interface"
as something not to build, which reads as the dashboard-style viewer the wider scope fence
(§1.1) already excludes, not a filtered, paginated, role-restricted API endpoint an explicit
instruction this session asked for. Restricted to admin: the log's `before`/`after` JSON can
carry any field of any entity, which is broader exposure than any single role below admin
has a standing reason to see.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fcesapi.db import get_db
from fcesapi.models import AuditLog, User, UserRole
from fcesapi.schemas import AuditLogListOut, AuditLogOut
from fcesapi.security import require_role

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=AuditLogListOut)
def list_audit_log(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.admin))],
    actor_id: int | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
    action: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
) -> AuditLogListOut:
    stmt = select(AuditLog)
    if actor_id is not None:
        stmt = stmt.where(AuditLog.actor_id == actor_id)
    if entity_type is not None:
        stmt = stmt.where(AuditLog.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(AuditLog.entity_id == entity_id)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if date_from is not None:
        stmt = stmt.where(AuditLog.at >= date_from)
    if date_to is not None:
        stmt = stmt.where(AuditLog.at <= date_to)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.order_by(AuditLog.at.desc()).offset((page - 1) * page_size).limit(page_size)
    ).all()
    return AuditLogListOut(
        items=[AuditLogOut.model_validate(r) for r in rows],
        total=total, page=page, page_size=page_size,
    )
