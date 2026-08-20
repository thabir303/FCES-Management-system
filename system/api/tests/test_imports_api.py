"""E1-E5 acceptance: upload -> map -> process -> resolve -> commit, against real Postgres
and the real fitted pipeline (fcesreg is not mocked -- this is the actual boundary)."""

from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete

from fcesapi.db import get_sessionmaker
from fcesapi.main import app
from fcesapi.models import (
    Asset,
    AuditLog,
    ImportBatch,
    ImportRow,
    ReviewDecision,
    User,
    UserRole,
)
from fcesapi.security import hash_password

from conftest import requires_db

pytestmark = requires_db

def _csv(tag: str) -> bytes:
    # Unique equipment names per call: a real, live-assets dedup check runs against
    # whatever this DB already holds, and a fixed name would make repeat test runs see
    # their own leftovers -- or another test's -- as duplicates and (correctly) refuse to
    # create them. That happened once already; this is the fix, not a workaround.
    return (
        b"Equipment Name,Description,Make\n"
        + f"Centrifuge 5424 {tag},Bench top centrifuge,Eppendorf\n".encode()
        + f"Autoclave Pro 200 {tag},Steam steriliser,Getinge\n".encode()
    )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def technician():
    db = get_sessionmaker()()
    tag = uuid.uuid4().hex[:8]
    user = User(
        email=f"tech-{tag}@fces.internal", name="Tech", role=UserRole.technician,
        password_hash=hash_password("testpass123"),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    yield user
    batches = db.query(ImportBatch).filter(ImportBatch.uploaded_by == user.id).all()
    for batch in batches:
        row_ids = [r.id for r in db.query(ImportRow.id).filter(ImportRow.batch_id == batch.id)]
        if row_ids:
            db.execute(delete(ReviewDecision).where(ReviewDecision.import_row_id.in_(row_ids)))
        db.execute(delete(ImportRow).where(ImportRow.batch_id == batch.id))
    asset_ids = [a.id for a in db.query(Asset.id).filter(Asset.created_by == user.id)]
    if asset_ids:
        db.execute(delete(AuditLog).where(AuditLog.entity_id.in_(asset_ids)))
        db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
    db.execute(delete(AuditLog).where(AuditLog.actor_id == user.id))
    db.execute(delete(ImportBatch).where(ImportBatch.uploaded_by == user.id))
    db.execute(delete(User).where(User.id == user.id))
    db.commit()
    db.close()


def _token(client: TestClient, user: User) -> dict:
    r = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "testpass123"}
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


class TestImportWizard:
    """A 50-row-scale file split auto/review with scores, alternatives and evidence
    populated (E3's acceptance criterion, exercised at CSV-fixture scale rather than 50
    rows -- the code path per row is identical)."""

    def _upload_and_map(self, client, h):
        r = client.post(
            "/api/v1/imports",
            files={"file": ("assets.csv", io.BytesIO(_csv(uuid.uuid4().hex[:8])), "text/csv")},
            headers=h,
        )
        assert r.status_code == 201, r.text
        batch_id = r.json()["batch_id"]
        assert r.json()["detected_columns"] == ["Equipment Name", "Description", "Make"]

        r = client.post(
            f"/api/v1/imports/{batch_id}/mapping",
            json={
                "column_mapping": {
                    "Equipment Name": "name", "Description": "description", "Make": "manufacturer"
                },
                "precision_target": 0.95,
            },
            headers=h,
        )
        assert r.status_code == 200
        return batch_id

    def test_a_mismatched_header_file_still_maps_by_hand(self, client, technician):
        """E2's acceptance criterion: headers matching nothing still map successfully."""
        h = _token(client, technician)
        csv = f"Col A,Col B\nWidget One {uuid.uuid4().hex[:8]},Some text\n".encode()
        r = client.post(
            "/api/v1/imports", files={"file": ("odd.csv", io.BytesIO(csv), "text/csv")},
            headers=h,
        )
        batch_id = r.json()["batch_id"]
        r = client.post(
            f"/api/v1/imports/{batch_id}/mapping",
            json={"column_mapping": {"Col A": "name"}, "precision_target": 0.95},
            headers=h,
        )
        assert r.status_code == 200

    def test_rows_are_split_with_scores_alternatives_and_evidence(self, client, technician):
        h = _token(client, technician)
        batch_id = self._upload_and_map(client, h)

        r = client.get(f"/api/v1/imports/{batch_id}", headers=h)
        assert r.json()["status"] == "ready_for_review"
        assert r.json()["row_count"] == 2

        r = client.get(f"/api/v1/imports/{batch_id}/rows", headers=h)
        rows = r.json()
        assert len(rows) == 2
        for row in rows:
            assert row["class_cpv_code"] is not None
            assert row["class_score"] is not None
            assert row["class_alternatives"]  # non-empty: alternatives are not optional
            assert row["evidence"] is not None
            assert row["route"] in ("auto", "review")

    def test_resolve_writes_the_six_review_queue_fields(self, client, technician):
        """The fields the review queue must never lose: decision_type, action,
        chosen_value, seconds_taken here; dedup_score/class_score already on the row it
        joins to via import_row_id."""
        h = _token(client, technician)
        batch_id = self._upload_and_map(client, h)
        row_id = client.get(f"/api/v1/imports/{batch_id}/rows", headers=h).json()[0]["id"]

        r = client.post(
            f"/api/v1/imports/{batch_id}/rows/{row_id}/resolve",
            json={"action": "accept", "decision_type": "classification", "seconds_taken": 8},
            headers=h,
        )
        assert r.status_code == 200

        db = get_sessionmaker()()
        try:
            decision = db.query(ReviewDecision).filter(
                ReviewDecision.import_row_id == row_id
            ).one()
            assert decision.decision_type == "classification"
            assert decision.action.value == "accept"
            assert decision.seconds_taken == 8
            row = db.get(ImportRow, row_id)
            assert row.resolved_at is not None
        finally:
            db.close()

    def test_an_unknown_action_is_422_not_a_silent_no_op(self, client, technician):
        h = _token(client, technician)
        batch_id = self._upload_and_map(client, h)
        row_id = client.get(f"/api/v1/imports/{batch_id}/rows", headers=h).json()[0]["id"]
        r = client.post(
            f"/api/v1/imports/{batch_id}/rows/{row_id}/resolve",
            json={"action": "explode", "decision_type": "classification", "seconds_taken": 1},
            headers=h,
        )
        assert r.status_code == 422

    def test_commit_writes_assets_with_source_row_id_and_one_audit_row_each(
        self, client, technician
    ):
        """E5's acceptance criterion."""
        h = _token(client, technician)
        batch_id = self._upload_and_map(client, h)

        r = client.post(f"/api/v1/imports/{batch_id}/commit", headers=h)
        assert r.status_code == 200
        assert r.json()["assets_created"] == 2

        db = get_sessionmaker()()
        try:
            rows = db.query(ImportRow).filter(ImportRow.batch_id == batch_id).all()
            for row in rows:
                assert row.final_asset_id is not None
                asset = db.get(Asset, row.final_asset_id)
                assert asset.source_row_id == row.id
                audit = db.query(AuditLog).filter(
                    AuditLog.entity_type == "asset",
                    AuditLog.entity_id == asset.id,
                    AuditLog.action == "import_commit",
                ).one_or_none()
                assert audit is not None
        finally:
            db.close()

    def test_readonly_cannot_upload(self, client):
        db = get_sessionmaker()()
        tag = uuid.uuid4().hex[:8]
        user = User(
            email=f"ro-{tag}@fces.internal", name="RO", role=UserRole.readonly,
            password_hash=hash_password("testpass123"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        try:
            h = _token(client, user)
            r = client.post(
                "/api/v1/imports",
                files={"file": ("x.csv", io.BytesIO(_csv(uuid.uuid4().hex[:8])), "text/csv")},
                headers=h,
            )
            assert r.status_code == 403
        finally:
            db.execute(delete(User).where(User.id == user.id))
            db.commit()
            db.close()
