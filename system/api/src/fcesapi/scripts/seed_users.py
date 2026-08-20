"""Seeds one admin, one technician, one readonly user (§7.1). No `/users` API -- this
script is the only way accounts are created. Passwords come from `Settings`, never
`os.getenv` inline (§12.3), so a committed seed script cannot leak a real credential."""

from __future__ import annotations

import sys

from sqlalchemy import select

from fcesapi.config import get_settings
from fcesapi.db import get_sessionmaker
from fcesapi.models import User, UserRole
from fcesapi.security import hash_password


def _seeds() -> list[tuple[str, str, UserRole, str]]:
    settings = get_settings()
    return [
        ("admin@fces.internal", "Admin", UserRole.admin, settings.seed_admin_password),
        (
            "technician@fces.internal", "Technician", UserRole.technician,
            settings.seed_technician_password,
        ),
        (
            "readonly@fces.internal", "Readonly", UserRole.readonly,
            settings.seed_readonly_password,
        ),
    ]


def main() -> int:
    db = get_sessionmaker()()
    try:
        for email, name, role, password in _seeds():
            if not password:
                print(f"seed_users: password for {email} not set, skipping", file=sys.stderr)
                continue
            if db.scalar(select(User).where(User.email == email)) is not None:
                print(f"seed_users: {email} already exists, skipping")
                continue
            db.add(
                User(email=email, name=name, role=role, password_hash=hash_password(password))
            )
            print(f"seed_users: created {email} ({role.value})")
        db.commit()
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
