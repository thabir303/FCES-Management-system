"""Capture real API request/response bodies for appendix_system.tex (supervisor request,
2026-08-27): "API-level evidence in place of screenshots ... real bodies from a live run
against Postgres, not illustrations."

There is no frontend (`system/web` is `.gitkeep` only), so the appendix cannot carry
screenshots; this script drives the same `TestClient`/real-Postgres path the test suite
itself uses (no auth faking, no mocked `fcesreg`) and writes what actually came back to
JSON, so the appendix quotes a real exchange rather than a constructed example.

Every user, import batch, row, decision and asset this script creates is deleted again in
its own `finally` block -- it must leave the database exactly as it found it, the same
discipline every test fixture here already follows.

    .venv/bin/python -m fcesapi.scripts.capture_appendix_evidence
"""

from __future__ import annotations

import io
import json
import sys
import uuid
from pathlib import Path

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

OUT = Path(__file__).resolve().parents[5] / "results" / "appendix_evidence.json"


def _token(client: TestClient, email: str) -> dict:
    r = client.post("/api/v1/auth/login", json={"email": email, "password": "testpass123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def main() -> int:
    db = get_sessionmaker()()
    tag = uuid.uuid4().hex[:8]
    client = TestClient(app)
    evidence: dict = {}
    made_user_ids: list[int] = []

    try:
        technician = User(
            email=f"tech-{tag}@fces.internal", name="Tech", role=UserRole.technician,
            password_hash=hash_password("testpass123"),
        )
        admin = User(
            email=f"admin-{tag}@fces.internal", name="Admin", role=UserRole.admin,
            password_hash=hash_password("testpass123"),
        )
        db.add_all([technician, admin])
        db.commit()
        db.refresh(technician)
        db.refresh(admin)
        made_user_ids = [technician.id, admin.id]

        tech_h = _token(client, technician.email)
        admin_h = _token(client, admin.email)

        # Seed the "existing" asset an incoming duplicate row will be matched against.
        seed_name = f"Centrifuge 5424 {tag}"
        r = client.post(
            "/api/v1/assets",
            json={"name": seed_name, "description": "Bench top centrifuge, Eppendorf"},
            headers=tech_h,
        )
        assert r.status_code == 201, r.text
        seed_asset = r.json()
        evidence["seed_asset_created"] = seed_asset

        # A CSV with one exact-text duplicate of the seed asset (drives dedup_decision to
        # 'duplicate' against a real fitted matcher, not a stubbed one) and one genuinely
        # novel item.
        csv = (
            "Equipment Name,Description,Make\n"
            f"{seed_name},Bench top centrifuge,Eppendorf\n"
            f"Autoclave Pro 200 {tag},Steam steriliser,Getinge\n"
        ).encode()

        r = client.post(
            "/api/v1/imports",
            files={"file": ("assets.csv", io.BytesIO(csv), "text/csv")},
            headers=tech_h,
        )
        assert r.status_code == 201, r.text
        evidence["bulk_import_request"] = {
            "method": "POST", "path": "/api/v1/imports",
            "files": {"file": "assets.csv (2 rows, see body below)"},
        }
        evidence["bulk_import_response"] = r.json()
        batch_id = r.json()["batch_id"]

        r = client.post(
            f"/api/v1/imports/{batch_id}/mapping",
            json={
                "column_mapping": {
                    "Equipment Name": "name", "Description": "description",
                    "Make": "manufacturer",
                },
                "precision_target": 0.95,
            },
            headers=tech_h,
        )
        assert r.status_code == 200, r.text

        rows = client.get(f"/api/v1/imports/{batch_id}/rows", headers=tech_h).json()
        duplicate_row = next(
            r for r in rows if r["dedup_decision"] == "duplicate"
        )
        novel_row = next(r for r in rows if r["dedup_decision"] != "duplicate")
        evidence["flagged_duplicate_row"] = duplicate_row

        # The reviewer confirms the automated duplicate call.
        r = client.post(
            f"/api/v1/imports/{batch_id}/rows/{duplicate_row['id']}/resolve",
            json={"action": "accept", "decision_type": "dedup", "seconds_taken": 6},
            headers=tech_h,
        )
        assert r.status_code == 200, r.text
        evidence["review_decision_response"] = r.json()

        # The novel row: accept its classification suggestion so it commits as a new asset.
        r = client.post(
            f"/api/v1/imports/{batch_id}/rows/{novel_row['id']}/resolve",
            json={"action": "accept", "decision_type": "classification", "seconds_taken": 4},
            headers=tech_h,
        )
        assert r.status_code == 200, r.text

        decision = db.query(ReviewDecision).filter(
            ReviewDecision.import_row_id == duplicate_row["id"]
        ).one()
        evidence["review_decision_row"] = {
            "id": decision.id,
            "import_row_id": decision.import_row_id,
            "actor_id": decision.actor_id,
            "decision_type": decision.decision_type,
            "action": decision.action.value,
            "chosen_value": decision.chosen_value,
            "seconds_taken": decision.seconds_taken,
            "at": decision.at.isoformat(),
        }

        r = client.post(f"/api/v1/imports/{batch_id}/commit", headers=tech_h)
        assert r.status_code == 200, r.text
        evidence["commit_response"] = r.json()

        novel_row_after = client.get(
            f"/api/v1/imports/{batch_id}/rows", headers=tech_h
        ).json()
        novel_row_after = next(
            r for r in novel_row_after if r["id"] == novel_row["id"]
        )
        asset_id = novel_row_after["final_asset_id"]
        assert asset_id is not None, "the novel row did not commit to an asset"
        evidence["resulting_asset"] = client.get(
            f"/api/v1/assets/{asset_id}", headers=tech_h
        ).json()

        audit = client.get(
            "/api/v1/audit", params={"entity_id": asset_id, "page_size": 5}, headers=admin_h
        )
        assert audit.status_code == 200, audit.text
        evidence["audit_row"] = audit.json()["items"][0]

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
        print(f"wrote {OUT}")
        return 0

    finally:
        batch_ids = [b.id for b in db.query(ImportBatch.id).filter(
            ImportBatch.uploaded_by.in_(made_user_ids)
        )]
        if batch_ids:
            row_ids = [r.id for r in db.query(ImportRow.id).filter(
                ImportRow.batch_id.in_(batch_ids)
            )]
            if row_ids:
                db.execute(
                    delete(ReviewDecision).where(ReviewDecision.import_row_id.in_(row_ids))
                )
            db.execute(delete(ImportRow).where(ImportRow.batch_id.in_(batch_ids)))
            db.execute(delete(ImportBatch).where(ImportBatch.id.in_(batch_ids)))
        asset_ids = [a.id for a in db.query(Asset.id).filter(
            Asset.created_by.in_(made_user_ids)
        )]
        if asset_ids:
            db.execute(delete(AuditLog).where(AuditLog.entity_id.in_(asset_ids)))
            db.execute(delete(Asset).where(Asset.id.in_(asset_ids)))
        db.execute(delete(AuditLog).where(AuditLog.actor_id.in_(made_user_ids)))
        db.execute(delete(User).where(User.id.in_(made_user_ids)))
        db.commit()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
