"""§7.8, §9. The bulk import wizard and the review queue.

Upload -> map columns -> background processing splits rows auto/review -> a human resolves
the review rows one at a time -> commit writes accepted rows to `assets`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.db import get_db
from fcesapi.models import (
    Asset,
    AuditLog,
    DedupCall,
    ImportBatch,
    ImportRow,
    ImportRoute,
    ImportStatus,
    ReviewAction,
    ReviewDecision,
    User,
    UserRole,
)
from fcesapi.schemas import (
    ImportMappingRequest,
    ImportRowOut,
    ImportUploadOut,
    ResolveRowRequest,
)
from fcesapi.security import get_current_user, require_role
from fcesapi.services import importer

router = APIRouter(prefix="/imports", tags=["imports"])

#: Raw rows are held on the batch between upload and mapping-confirmation rather than
#: persisted before a mapping exists -- `import_rows.normalised` has nothing to be
#: normalised from until `column_mapping` is known. Keyed by batch id; cleared on mapping.
_pending_rows: dict[int, list[dict]] = {}


def _read_rows(upload: UploadFile) -> list[dict]:
    if upload.filename and upload.filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(upload.file)
    else:
        df = pd.read_csv(upload.file)
    df = df.where(pd.notna(df), None)
    return df.to_dict("records")


@router.post("", response_model=ImportUploadOut, status_code=status.HTTP_201_CREATED)
def upload_import(
    file: UploadFile,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
) -> ImportUploadOut:
    rows = _read_rows(file)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_file", "message": "the uploaded file has no rows"},
        )
    batch = ImportBatch(
        filename=file.filename or "upload",
        uploaded_by=user.id,
        precision_target=0.95,
        column_mapping={},
        status=ImportStatus.pending,
    )
    db.add(batch)
    db.flush()
    _pending_rows[batch.id] = rows
    db.commit()
    return ImportUploadOut(
        batch_id=batch.id,
        detected_columns=list(rows[0].keys()),
        sample_rows=rows[:5],
    )


@router.post("/{batch_id}/mapping")
def set_mapping(
    batch_id: int,
    body: ImportMappingRequest,
    background: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
):
    batch = db.get(ImportBatch, batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_not_found", "message": f"no import batch {batch_id}"},
        )
    rows = _pending_rows.pop(batch_id, None)
    if rows is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "already_mapped", "message": "this batch has already been mapped"},
        )
    batch.column_mapping = body.column_mapping
    batch.precision_target = body.precision_target
    batch.status = ImportStatus.processing
    db.commit()

    background.add_task(_process_and_close, batch_id, rows)
    return {"batch_id": batch_id, "status": "processing"}


def _process_and_close(batch_id: int, rows: list[dict]) -> None:
    from fcesapi.db import get_sessionmaker

    db = get_sessionmaker()()
    try:
        batch = db.get(ImportBatch, batch_id)
        try:
            importer.process_batch(db, batch, rows)
        except Exception as exc:  # noqa: BLE001 -- a batch failure must not vanish silently
            batch.status = ImportStatus.failed
            batch.error = str(exc)[:2000]
            db.commit()
    finally:
        db.close()


@router.get("/{batch_id}")
def get_batch(
    batch_id: int, db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    batch = db.get(ImportBatch, batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_not_found", "message": f"no import batch {batch_id}"},
        )
    return {
        "id": batch.id, "filename": batch.filename, "status": batch.status,
        "row_count": batch.row_count, "auto_count": batch.auto_count,
        "review_count": batch.review_count, "error": batch.error,
    }


@router.get("/{batch_id}/rows", response_model=list[ImportRowOut])
def list_rows(
    batch_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
    route: str | None = None,
    resolved: bool | None = None,
):
    stmt = select(ImportRow).where(ImportRow.batch_id == batch_id)
    if route:
        stmt = stmt.where(ImportRow.route == route)
    if resolved is not None:
        stmt = stmt.where(
            ImportRow.resolved_at.isnot(None) if resolved else ImportRow.resolved_at.is_(None)
        )
    rows = db.scalars(stmt.order_by(ImportRow.row_index)).all()
    return [ImportRowOut.model_validate(r) for r in rows]


@router.post("/{batch_id}/rows/{row_id}/resolve")
def resolve_row(
    batch_id: int,
    row_id: int,
    body: ResolveRowRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
):
    """A review-queue decision. Writes the six logging fields the review queue must never
    lose (§ ruling): decision_type/action/chosen_value/seconds_taken here, joined to the
    original dedup_score/class_score on `ImportRow` via `import_row_id`."""
    row = db.get(ImportRow, row_id)
    if row is None or row.batch_id != batch_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_row_not_found", "message": f"no row {row_id} in batch {batch_id}"},
        )
    try:
        action = ReviewAction(body.action)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_action", "message": f"unknown action {body.action!r}"},
        ) from None

    chosen_value = body.chosen_cpv_code or (
        str(body.chosen_asset_id) if body.chosen_asset_id is not None else None
    )
    db.add(
        ReviewDecision(
            import_row_id=row.id,
            actor_id=user.id,
            decision_type=body.decision_type,
            action=action,
            chosen_value=chosen_value,
            seconds_taken=body.seconds_taken,
        )
    )

    if action == ReviewAction.override:
        if body.decision_type == "classification" and body.chosen_cpv_code:
            row.class_cpv_code = body.chosen_cpv_code
        if body.decision_type == "dedup" and body.chosen_asset_id is not None:
            row.dedup_decision = DedupCall.duplicate
            row.dedup_candidate_asset_id = body.chosen_asset_id
    elif action == ReviewAction.accept and body.decision_type == "dedup":
        row.dedup_decision = row.dedup_decision or DedupCall.new

    if body.action != "skip":
        row.resolved_by = user.id
        row.resolved_at = datetime.now(UTC)

    db.commit()
    return {"row_id": row_id, "action": action.value}


@router.post("/{batch_id}/commit")
def commit_batch(
    batch_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
):
    """Writes remaining accepted rows to `assets` (§9.2, E5). One audit row per asset."""
    batch = db.get(ImportBatch, batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "import_not_found", "message": f"no import batch {batch_id}"},
        )
    rows = db.scalars(
        select(ImportRow).where(
            ImportRow.batch_id == batch_id,
            ImportRow.final_asset_id.is_(None),
        )
    ).all()

    n = db.scalar(select(Asset).order_by(Asset.id.desc()).limit(1))
    next_n = (n.id if n else 0) + 1

    created = 0
    for row in rows:
        if row.route == ImportRoute.review and row.resolved_at is None:
            continue  # still awaiting a human decision -- not committed
        if row.dedup_decision == "duplicate":
            continue  # a confirmed duplicate is not written as a new asset
        normalised = row.normalised or {}
        if not normalised.get("name"):
            continue

        asset = Asset(
            asset_tag=f"FCES-{next_n:06d}",
            name=normalised["name"],
            description=normalised.get("description"),
            manufacturer=normalised.get("manufacturer"),
            model=normalised.get("model"),
            serial_number=normalised.get("serial_number"),
            cpv_code=row.class_cpv_code,
            owning_department=normalised.get("owning_department"),
            source_row_id=row.id,
            created_by=user.id,
        )
        db.add(asset)
        db.flush()
        row.final_asset_id = asset.id
        db.add(
            AuditLog(
                actor_id=user.id, entity_type="asset", entity_id=asset.id,
                action="import_commit", before=None,
                after={"source_row_id": row.id, "batch_id": batch_id},
            )
        )
        next_n += 1
        created += 1

    batch.status = ImportStatus.committed
    db.commit()
    return {"batch_id": batch_id, "assets_created": created}
