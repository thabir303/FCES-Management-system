"""§5b acceptance: floor plan upload and pin CRUD against real Postgres."""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import Asset, AuditLog, FloorPlan, Location, User, UserRole
from fcesapi.security import hash_password

from conftest import requires_db

pytestmark = requires_db

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


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
        plan_ids = [p.id for p in db.query(FloorPlan.id).filter(FloorPlan.uploaded_by.in_(ids)).all()]
        db.execute(delete(AuditLog).where(AuditLog.actor_id.in_(ids)))
        if asset_ids:
            db.execute(delete(AuditLog).where(AuditLog.entity_id.in_(asset_ids)))
            db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
        if plan_ids:
            db.execute(delete(Location).where(Location.floorplan_id.in_(plan_ids)))
            db.execute(delete(FloorPlan).where(FloorPlan.id.in_(plan_ids)))
        db.execute(delete(User).where(User.id.in_(ids)))
        db.commit()
        db.close()


def _token(client: TestClient, users, role: UserRole) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": users[role].email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def floorplan(client, users):
    h = _token(client, users, UserRole.technician)
    r = client.post(
        "/api/v1/floorplans",
        data={"building": "Main", "floor": "1", "name": "Test Lab"},
        files={"image": ("plan.png", io.BytesIO(_PNG), "image/png")},
        headers=h,
    )
    assert r.status_code == 201, r.text
    return r.json()


class TestFloorPlanUpload:
    def test_upload_records_image_dimensions(self, floorplan):
        assert floorplan["image_w"] == 1
        assert floorplan["image_h"] == 1
        assert floorplan["building"] == "Main"

    def test_upload_rejects_non_image_mime(self, client, users):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            "/api/v1/floorplans",
            data={"building": "Main", "floor": "1"},
            files={"image": ("plan.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            headers=h,
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "wrong_mime"

    def test_readonly_upload_is_403(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.post(
            "/api/v1/floorplans",
            data={"building": "Main", "floor": "1"},
            files={"image": ("plan.png", io.BytesIO(_PNG), "image/png")},
            headers=h,
        )
        assert r.status_code == 403

    def test_list_includes_uploaded_plan(self, client, users, floorplan):
        h = _token(client, users, UserRole.readonly)
        r = client.get("/api/v1/floorplans", headers=h)
        assert r.status_code == 200
        assert any(p["id"] == floorplan["id"] for p in r.json())


class TestPins:
    def test_create_list_move_delete_pin(self, client, users, floorplan):
        h = _token(client, users, UserRole.technician)
        created = client.post(
            f"/api/v1/floorplans/{floorplan['id']}/pins",
            json={"x_pct": 25.5, "y_pct": 60.0, "label": "bench 3"},
            headers=h,
        )
        assert created.status_code == 201, created.text
        pin_id = created.json()["id"]
        assert created.json()["floorplan_id"] == floorplan["id"]

        listed = client.get(f"/api/v1/floorplans/{floorplan['id']}/pins", headers=h)
        assert listed.status_code == 200
        assert any(p["id"] == pin_id for p in listed.json())

        moved = client.patch(
            f"/api/v1/floorplans/{floorplan['id']}/pins/{pin_id}",
            json={"x_pct": 80.0, "y_pct": 10.0},
            headers=h,
        )
        assert moved.status_code == 200
        assert float(moved.json()["x_pct"]) == 80.0

        deleted = client.delete(f"/api/v1/floorplans/{floorplan['id']}/pins/{pin_id}", headers=h)
        assert deleted.status_code == 204

        listed_again = client.get(f"/api/v1/floorplans/{floorplan['id']}/pins", headers=h)
        assert all(p["id"] != pin_id for p in listed_again.json())

    def test_pin_out_of_range_coordinate_is_422(self, client, users, floorplan):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            f"/api/v1/floorplans/{floorplan['id']}/pins",
            json={"x_pct": 150.0, "y_pct": 10.0},
            headers=h,
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "coordinate_out_of_range"

    def test_deleting_a_pin_orphans_no_asset_reference(self, client, users, floorplan):
        db = get_sessionmaker()()
        try:
            h = _token(client, users, UserRole.technician)
            asset = Asset(
                asset_tag=f"PIN-{uuid.uuid4().hex[:8]}", name="Pinned asset",
                created_by=users[UserRole.technician].id,
            )
            db.add(asset)
            db.commit()
            db.refresh(asset)

            created = client.post(
                f"/api/v1/floorplans/{floorplan['id']}/pins",
                json={"x_pct": 10.0, "y_pct": 10.0, "asset_id": asset.id},
                headers=h,
            )
            assert created.status_code == 201, created.text
            pin_id = created.json()["id"]

            db.refresh(asset)
            assert asset.location_id == pin_id

            deleted = client.delete(f"/api/v1/floorplans/{floorplan['id']}/pins/{pin_id}", headers=h)
            assert deleted.status_code == 204

            db.expire(asset)
            db.refresh(asset)
            assert asset.location_id is None
        finally:
            db.execute(delete(AuditLog).where(AuditLog.entity_id == asset.id))
            db.execute(delete(Asset).where(Asset.id == asset.id))
            db.commit()
            db.close()
