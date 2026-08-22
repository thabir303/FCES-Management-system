"""§5c acceptance: attachment upload/list/download/delete against real Postgres."""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import Asset, Attachment, AuditLog, User, UserRole
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
            # Attachment rows cascade with their asset (ON DELETE CASCADE).
            db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
        db.execute(delete(User).where(User.id.in_(ids)))
        db.commit()
        db.close()


@pytest.fixture
def asset(users):
    db = get_sessionmaker()()
    a = Asset(asset_tag=f"ATT-{uuid.uuid4().hex[:8]}", name="Attachment test asset", created_by=users[UserRole.technician].id)
    db.add(a)
    db.commit()
    db.refresh(a)
    yield a
    db.close()


def _token(client: TestClient, users, role: UserRole) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": users[role].email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestPhotoAndPdfUpload:
    def test_photo_upload_accepts_image_mime(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "photo", "title": "front view"},
            files={"file": ("photo.png", io.BytesIO(png_bytes), "image/png")},
            headers=h,
        )
        assert r.status_code == 201, r.text
        assert r.json()["kind"] == "photo"
        assert r.json()["mime"] == "image/png"

    def test_photo_upload_rejects_pdf_mime(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "photo"},
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            headers=h,
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "wrong_mime"

    def test_pdf_upload_accepts_pdf_mime(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "pdf"},
            files={"file": ("manual.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
            headers=h,
        )
        assert r.status_code == 201, r.text
        assert r.json()["kind"] == "pdf"

    def test_risk_assessment_kind_accepts_pdf(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "risk_assessment"},
            files={"file": ("ra.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
            headers=h,
        )
        assert r.status_code == 201, r.text
        assert r.json()["kind"] == "risk_assessment"

    def test_readonly_upload_is_403(self, client, users, asset):
        h = _token(client, users, UserRole.readonly)
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "pdf"},
            files={"file": ("m.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            headers=h,
        )
        assert r.status_code == 403


class TestListDownloadDelete:
    def test_list_download_and_delete_round_trip(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        upload = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "pdf", "title": "spec sheet"},
            files={"file": ("spec.pdf", io.BytesIO(b"%PDF-1.4 body"), "application/pdf")},
            headers=h,
        )
        attachment_id = upload.json()["id"]

        listed = client.get(f"/api/v1/assets/{asset.id}/attachments", headers=h)
        assert listed.status_code == 200
        assert any(a["id"] == attachment_id for a in listed.json())

        downloaded = client.get(f"/api/v1/attachments/{attachment_id}/download", headers=h)
        assert downloaded.status_code == 200
        assert downloaded.content == b"%PDF-1.4 body"

        deleted = client.delete(f"/api/v1/attachments/{attachment_id}", headers=h)
        assert deleted.status_code == 204

        listed_again = client.get(f"/api/v1/assets/{asset.id}/attachments", headers=h)
        assert all(a["id"] != attachment_id for a in listed_again.json())

    def test_is_primary_exclusive_per_kind(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        first = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "photo", "is_primary": "true"},
            files={"file": ("a.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
            headers=h,
        )
        assert first.status_code == 201, first.text
        second = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "photo", "is_primary": "true"},
            files={"file": ("b.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
            headers=h,
        )
        assert second.status_code == 409
        assert second.json()["detail"]["code"] == "primary_already_set"

    def test_link_kind_requires_url_not_file(self, client, users, asset):
        h = _token(client, users, UserRole.technician)
        r = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "link"},
            headers=h,
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "url_required"

        ok = client.post(
            f"/api/v1/assets/{asset.id}/attachments",
            data={"kind": "link", "url": "https://example.org/manual"},
            headers=h,
        )
        assert ok.status_code == 201, ok.text
        assert ok.json()["url"] == "https://example.org/manual"
