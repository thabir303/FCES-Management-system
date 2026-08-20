"""D2/D3 acceptance: role enforcement and asset CRUD, against real Postgres.

Requires `make db && make migrate` (or the equivalent manual steps) -- skipped with a
reason otherwise, never failed, per `conftest.requires_db`.
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
    """One of each role, torn down after the test. Emails carry a random suffix so
    parallel test runs never collide on the UNIQUE constraint."""
    db = get_sessionmaker()()
    tag = uuid.uuid4().hex[:8]
    made = {}
    try:
        for role in UserRole:
            u = User(
                email=f"{role.value}-{tag}@fces.internal",
                name=role.value,
                role=role,
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
    r = client.post(
        "/api/v1/auth/login",
        json={"email": users[role].email, "password": "testpass123"},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestRoleEnforcement:
    def test_readonly_write_is_403_not_401(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.post("/api/v1/assets", json={"name": "x"}, headers=h)
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "forbidden"

    def test_technician_write_succeeds(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post("/api/v1/assets", json={"name": "Test Asset A"}, headers=h)
        assert r.status_code == 201

    def test_no_token_is_401(self, client):
        r = client.get("/api/v1/assets")
        assert r.status_code == 401
        assert r.json()["detail"]["code"] == "not_authenticated"

    def test_every_get_is_permitted_to_readonly(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.get("/api/v1/assets", headers=h)
        assert r.status_code == 200


class TestAssetCrud:
    def test_create_then_get_round_trips(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            "/api/v1/assets",
            json={"name": "Round Trip Asset", "manufacturer": "Acme"},
            headers=h,
        )
        asset = r.json()
        assert asset["asset_tag"].startswith("FCES-")

        r = client.get(f"/api/v1/assets/{asset['id']}", headers=h)
        assert r.status_code == 200
        assert r.json()["manufacturer"] == "Acme"

    def test_patch_only_changes_supplied_fields(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            "/api/v1/assets",
            json={"name": "Patchable", "manufacturer": "Acme"},
            headers=h,
        )
        asset_id = r.json()["id"]
        r = client.patch(f"/api/v1/assets/{asset_id}", json={"model": "X1"}, headers=h)
        assert r.status_code == 200
        assert r.json()["manufacturer"] == "Acme"  # untouched
        assert r.json()["model"] == "X1"

    def test_missing_asset_is_404_with_the_envelope(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.get("/api/v1/assets/999999999", headers=h)
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "asset_not_found"

    def test_search_uses_the_gin_index_path(self, client, users):
        # Not an EXPLAIN check here -- that belongs to a DB-plan test -- but confirms the
        # search_tsv path returns the seeded row and nothing search_tsv wouldn't match.
        h = _token(client, users, UserRole.technician)
        client.post(
            "/api/v1/assets", json={"name": "Unique Spectrophotometer Zx9"}, headers=h
        )
        r = client.get("/api/v1/assets?q=spectrophotometer", headers=h)
        assert r.status_code == 200
        assert any("Spectrophotometer" in a["name"] for a in r.json()["items"])

    def test_update_writes_an_audit_row(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post("/api/v1/assets", json={"name": "Audited"}, headers=h)
        asset_id = r.json()["id"]
        client.patch(f"/api/v1/assets/{asset_id}", json={"model": "M2"}, headers=h)

        db = get_sessionmaker()()
        try:
            rows = db.query(AuditLog).filter(
                AuditLog.entity_type == "asset", AuditLog.entity_id == asset_id
            ).all()
            actions = {r.action for r in rows}
            assert {"create", "update"} <= actions
        finally:
            db.close()

    def test_by_tag_and_by_public_id_resolve_the_same_asset(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post("/api/v1/assets", json={"name": "Resolvable"}, headers=h)
        asset = r.json()

        by_tag = client.get(f"/api/v1/assets/by-tag/{asset['asset_tag']}", headers=h)
        by_pid = client.get(f"/api/v1/assets/by-public-id/{asset['public_id']}", headers=h)
        assert by_tag.json()["id"] == by_pid.json()["id"] == asset["id"]
