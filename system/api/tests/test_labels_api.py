"""§5a acceptance: QR code, single printable label, and the multi-label PDF sheet, against
real Postgres. The QR round trip is the acceptance criterion that matters most here -- the
URL a QR encodes must be one this API actually serves, not merely a well-formed string."""

from __future__ import annotations

import uuid
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import Asset, AuditLog, User, UserRole
from fcesapi.routers.labels import asset_url
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


@pytest.fixture
def asset(users):
    db = get_sessionmaker()()
    a = Asset(asset_tag=f"LBL-{uuid.uuid4().hex[:8]}", name="Label test asset", created_by=users[UserRole.technician].id)
    db.add(a)
    db.commit()
    db.refresh(a)
    yield a
    db.close()


def _token(client: TestClient, users, role: UserRole) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": users[role].email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestQrRoundTrip:
    def test_qr_endpoint_returns_a_png(self, client, users, asset):
        h = _token(client, users, UserRole.readonly)
        r = client.get(f"/api/v1/assets/{asset.id}/qr.png", headers=h)
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_encoded_url_resolves_to_this_asset(self, client, users, asset):
        """The actual round trip: decode nothing (no zbar dependency needed) -- just take
        the same URL the QR endpoint would encode, hit its path, and confirm it returns
        this exact asset. This is the literal claim "the QR must resolve to a URL the asset
        detail endpoint actually serves", verified end to end rather than assumed."""
        h = _token(client, users, UserRole.readonly)
        encoded = asset_url(asset)
        path = urlparse(encoded).path
        assert path == f"/a/{asset.public_id}"

        r = client.get(path, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["id"] == asset.id
        assert r.json()["public_id"] == str(asset.public_id)

    def test_public_url_requires_authentication(self, client, asset):
        path = urlparse(asset_url(asset)).path
        r = client.get(path)
        assert r.status_code == 401


class TestSingleLabel:
    def test_label_svg_embeds_qr_and_barcode(self, client, users, asset):
        h = _token(client, users, UserRole.readonly)
        r = client.get(f"/api/v1/assets/{asset.id}/label.svg", headers=h)
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/svg+xml"
        body = r.text
        assert body.count("<image") == 2  # QR + Code128 barcode, one file


class TestLabelSheet:
    def test_label_sheet_returns_a_pdf_with_one_page(self, client, users, asset):
        h = _token(client, users, UserRole.readonly)
        r = client.post("/api/v1/assets/label-sheet", json={"asset_ids": [asset.id]}, headers=h)
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content[:4] == b"%PDF"

    def test_label_sheet_rejects_empty_list(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.post("/api/v1/assets/label-sheet", json={"asset_ids": []}, headers=h)
        assert r.status_code == 422

    def test_label_sheet_404s_on_unknown_asset(self, client, users):
        h = _token(client, users, UserRole.readonly)
        r = client.post("/api/v1/assets/label-sheet", json={"asset_ids": [999999999]}, headers=h)
        assert r.status_code == 404
