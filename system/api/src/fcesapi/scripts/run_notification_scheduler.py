"""The daily reminder job (§5.8, D7b): container-level cron or an in-process scheduler
invokes this script once a day.

    python -m fcesapi.scripts.run_notification_scheduler
"""

from __future__ import annotations

from fcesapi.db import get_sessionmaker
from fcesapi.services.notifications import generate_due_notifications


def main() -> int:
    db = get_sessionmaker()()
    try:
        inserted = generate_due_notifications(db)
        print(f"inserted {inserted} new notification row(s)")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
