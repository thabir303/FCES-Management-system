"""§5e acceptance: the audit log query endpoint, against real Postgres.

`create_asset`/`update_asset` in routers/assets.py already write audit rows on every write --
this exercises that the query endpoint reads them back correctly, filters, paginates, and is
restricted to admin.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import Asset, AuditLog, User, UserRole
from fcesapi.security import hash_password

from conftest import requires_db

pytestmark = requires_db


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def users():
    db = get_sessionmaker()()
    tag = uuid.uuid4().hex[:8]
    made = {}
    try:
        for role in UserRole:
            u = User(
                email=f"{role.value}-{tag}@fces.internal", name=role.value, role=role,
                password_hash=hash_password("testpass123"),
            )
            db.add(u)
            made[role] = u
        db.commit()
        for u in made.values():
            db.refresh(u)
        yield made
    finally:
        ids = [u.id for u in made.values()]
        asset_ids = [a.id for a in db.query(Asset.id).filter(Asset.created_by.in_(ids)).all()]
        db.execute(delete(AuditLog).where(AuditLog.actor_id.in_(ids)))
        if asset_ids:
            db.execute(delete(AuditLog).where(AuditLog.entity_id.in_(asset_ids)))
            db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
        db.execute(delete(User).where(User.id.in_(ids)))
        db.commit()
        db.close()


def _token(client: TestClient, users, role: UserRole) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": users[role].email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestAuditQuery:
    def test_admin_sees_the_create_row_it_caused(self, client, users):
        h_tech = _token(client, users, UserRole.technician)
        created = client.post(
            "/api/v1/assets", json={"name": "Audit test asset"}, headers=h_tech
        )
        assert created.status_code == 201
        asset_id = created.json()["id"]

        h_admin = _token(client, users, UserRole.admin)
        r = client.get(f"/api/v1/audit?entity_type=asset&entity_id={asset_id}", headers=h_admin)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 1
        assert body["items"][0]["action"] == "create"
        assert body["items"][0]["actor_id"] == users[UserRole.technician].id
        assert body["items"][0]["after"]["name"] == "Audit test asset"

    def test_filters_by_action(self, client, users):
        h_tech = _token(client, users, UserRole.technician)
        created = client.post("/api/v1/assets", json={"name": "Filter test asset"}, headers=h_tech)
        asset_id = created.json()["id"]
        client.patch(f"/api/v1/assets/{asset_id}", json={"name": "Renamed"}, headers=h_tech)

        h_admin = _token(client, users, UserRole.admin)
        r = client.get(
            f"/api/v1/audit?entity_type=asset&entity_id={asset_id}&action=update", headers=h_admin
        )
        assert r.status_code == 200
        assert r.json()["total"] == 1
        assert r.json()["items"][0]["action"] == "update"

    def test_technician_is_403_not_401(self, client, users):
        h_tech = _token(client, users, UserRole.technician)
        r = client.get("/api/v1/audit", headers=h_tech)
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "forbidden"

    def test_readonly_is_403(self, client, users):
        h_ro = _token(client, users, UserRole.readonly)
        r = client.get("/api/v1/audit", headers=h_ro)
        assert r.status_code == 403

    def test_pagination_page_size(self, client, users):
        h_admin = _token(client, users, UserRole.admin)
        r = client.get("/api/v1/audit?page=1&page_size=1", headers=h_admin)
        assert r.status_code == 200
        assert len(r.json()["items"]) <= 1
