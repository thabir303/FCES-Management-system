"""§5d acceptance: service event logging, the due/overdue view, and the notification
scheduler's idempotency, against real Postgres."""

from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import Asset, AuditLog, Notification, ServiceEvent, User, UserRole
from fcesapi.security import hash_password
from fcesapi.services.notifications import generate_due_notifications

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
            # ServiceEvent and Notification rows cascade with their asset.
            db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
        db.execute(delete(User).where(User.id.in_(ids)))
        db.commit()
        db.close()


@pytest.fixture
def asset(users):
    db = get_sessionmaker()()
    a = Asset(
        asset_tag=f"SVC-{uuid.uuid4().hex[:8]}", name="Service test asset",
        created_by=users[UserRole.technician].id,
        service_interval_days=30, last_serviced_at=date.today() - timedelta(days=25),
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    yield a
    db.close()


def _token(client: TestClient, users, role: UserRole) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": users[role].email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestServiceEvents:
    def test_logging_a_service_event_recomputes_next_due_at(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        today = date.today()
        r = client.post(
            f"/api/v1/assets/{asset.id}/service-events",
            json={"performed_at": today.isoformat(), "provider": "Acme Servicing", "notes": "annual check"},
            headers=h,
        )
        assert r.status_code == 201, r.text

        db = get_sessionmaker()()
        try:
            db.expire_all()
            refreshed = db.get(Asset, asset.id)
            assert refreshed.last_serviced_at == today
            assert refreshed.next_due_at == today + timedelta(days=30)
        finally:
            db.close()

    def test_readonly_cannot_log_service_event(self, client, users, asset):
        h = _token(client, users, UserRole.readonly)
        r = client.post(
            f"/api/v1/assets/{asset.id}/service-events",
            json={"performed_at": date.today().isoformat()},
            headers=h,
        )
        assert r.status_code == 403

    def test_list_service_events(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        client.post(
            f"/api/v1/assets/{asset.id}/service-events",
            json={"performed_at": date.today().isoformat()},
            headers=h,
        )
        r = client.get(f"/api/v1/assets/{asset.id}/service-events", headers=h)
        assert r.status_code == 200
        assert len(r.json()) == 1


class TestServiceDue:
    def test_overdue_asset_appears_in_overdue_not_due_soon(self, client, users):
        db = get_sessionmaker()()
        overdue_asset = Asset(
            asset_tag=f"OVD-{uuid.uuid4().hex[:8]}", name="Overdue asset",
            created_by=users[UserRole.technician].id,
            service_interval_days=10, last_serviced_at=date.today() - timedelta(days=40),
        )
        db.add(overdue_asset)
        db.commit()
        db.refresh(overdue_asset)
        try:
            h = _token(client, users, UserRole.readonly)
            r = client.get("/api/v1/service/due", headers=h)
            assert r.status_code == 200
            body = r.json()
            assert any(a["id"] == overdue_asset.id for a in body["overdue"])
            assert all(a["id"] != overdue_asset.id for a in body["due_soon"])
        finally:
            db.execute(delete(AuditLog).where(AuditLog.entity_id == overdue_asset.id))
            db.execute(delete(Asset).where(Asset.id == overdue_asset.id))
            db.commit()
            db.close()


class TestNotificationScheduler:
    def test_running_twice_in_one_day_inserts_no_duplicates(self, users, asset):
        db = get_sessionmaker()()
        try:
            # asset fixture is already overdue-in-5-days (interval 30, serviced 25 days ago).
            first = generate_due_notifications(db, today=date.today())
            assert first >= 1
            second = generate_due_notifications(db, today=date.today())
            assert second == 0

            rows = db.scalars(select(Notification).where(Notification.asset_id == asset.id)).all()
            assert len(rows) == 1
        finally:
            db.execute(delete(Notification).where(Notification.asset_id == asset.id))
            db.commit()
            db.close()
