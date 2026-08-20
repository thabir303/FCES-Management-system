"""Shared fixtures. DB-dependent tests are skipped, with a reason, when Postgres is not
reachable -- mirroring the research side's `conftest.requires(...)` philosophy: a clean
checkout without `make bootstrap` run shows skips, not failures."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from fcesapi.config import get_settings
from fcesapi.db import get_engine


def _db_reachable() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable on DATABASE_URL -- run `make db && make migrate` first",
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
