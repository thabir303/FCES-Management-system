"""§5.8. The daily reminder job.

Computes `due_soon`/`overdue` items from `assets.next_due_at` and writes `notifications`
rows. Idempotent by construction: a row is only inserted if the exact
``(asset_id, kind, due_at)`` triple is not already present, which is also the table's own
UNIQUE constraint (`models.Notification.__table_args__`) -- running this twice on the same
day, before any due date has moved, finds every row already there and inserts nothing more.
No email, no external service: rows only, surfaced by ``GET /service/due`` and read on login,
per the scope ruling in `.claude/rules/system.md`.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.models import Asset, Notification

#: Matches the `?window_days=30` default documented for `GET /service/due` (PROJECT_PLAN.md
#: §7.6) -- a due date inside this many days from today is "due soon", not yet overdue.
DEFAULT_DUE_SOON_DAYS = 30


def generate_due_notifications(
    db: Session, today: date | None = None, window_days: int = DEFAULT_DUE_SOON_DAYS
) -> int:
    """Insert any missing `due_soon`/`overdue` rows. Returns the count actually inserted."""
    today = today or date.today()
    horizon = today + timedelta(days=window_days)
    assets = db.scalars(
        select(Asset).where(Asset.next_due_at.isnot(None), Asset.next_due_at <= horizon)
    ).all()

    inserted = 0
    for asset in assets:
        kind = "overdue" if asset.next_due_at < today else "due_soon"
        already = db.scalar(
            select(Notification.id).where(
                Notification.asset_id == asset.id,
                Notification.kind == kind,
                Notification.due_at == asset.next_due_at,
            )
        )
        if already is not None:
            continue
        db.add(Notification(asset_id=asset.id, kind=kind, due_at=asset.next_due_at))
        inserted += 1

    db.commit()
    return inserted
