"""§5c. Attachments: photos, PDFs and links per asset (D5).

`AttachmentKind` already carries `risk_assessment` alongside `photo`/`pdf`/`document`/
`manual`/`certificate`/`link` (models.py) -- health and safety and risk assessment files are
one more kind value, not a new table. Every kind except `link` stores a file under
`storage_root`; `link` stores a URL and no file, which the DB's own CHECK constraint
(`storage_path IS NOT NULL OR url IS NOT NULL`) already enforces either side of.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fcesapi.config import get_settings
from fcesapi.db import get_db
from fcesapi.models import Asset, Attachment, AttachmentKind, User, UserRole
from fcesapi.schemas import AttachmentOut
from fcesapi.security import get_current_user, require_role

router = APIRouter(tags=["attachments"])

#: Which kinds are file-backed vs URL-only, and what MIME a file-backed kind must carry.
#: `link` is the one kind with no file at all. Every other kind currently accepted by the
#: client brief (photo; pdf/document/risk_assessment/manual/certificate as PDF) is checked
#: against a real MIME rather than trusted from the filename extension.
_IMAGE_KINDS = {AttachmentKind.photo}
_PDF_KINDS = {
    AttachmentKind.pdf,
    AttachmentKind.document,
    AttachmentKind.risk_assessment,
    AttachmentKind.manual,
    AttachmentKind.certificate,
}


def _get_asset_or_404(db: Session, asset_id: int) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": f"no asset with id {asset_id}"},
        )
    return asset


def _get_attachment_or_404(db: Session, attachment_id: int) -> Attachment:
    attachment = db.get(Attachment, attachment_id)
    if attachment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "attachment_not_found",
                "message": f"no attachment with id {attachment_id}",
            },
        )
    return attachment


def _storage_dir(asset_id: int) -> Path:
    # Anchored through get_settings().storage_root, never a bare relative Path -- the
    # exact defect test_config.py's whole-package grep exists to catch.
    d = Path(get_settings().storage_root) / "attachments" / str(asset_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


@router.post(
    "/assets/{asset_id}/attachments",
    response_model=AttachmentOut,
    status_code=status.HTTP_201_CREATED,
)
def upload_attachment(
    asset_id: int,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(require_role(UserRole.technician))],
    kind: Annotated[str, Form()],
    title: Annotated[str | None, Form()] = None,
    is_primary: Annotated[bool, Form()] = False,
    file: UploadFile | None = None,
    url: Annotated[str | None, Form()] = None,
) -> AttachmentOut:
    asset = _get_asset_or_404(db, asset_id)
    try:
        kind_enum = AttachmentKind(kind)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "invalid_kind",
                "message": f"kind must be one of {[k.value for k in AttachmentKind]}",
            },
        )

    if kind_enum is AttachmentKind.link:
        if not url:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "url_required", "message": "kind 'link' requires a url"},
            )
        attachment = Attachment(
            asset_id=asset.id, kind=kind_enum, title=title, url=url,
            is_primary=is_primary, uploaded_by=user.id,
        )
    else:
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "file_required", "message": f"kind {kind!r} requires a file"},
            )
        mime = file.content_type or ""
        if kind_enum in _IMAGE_KINDS and not mime.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "wrong_mime",
                    "message": f"kind 'photo' requires an image/* file, got {mime!r}",
                },
            )
        if kind_enum in _PDF_KINDS and mime != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "wrong_mime",
                    "message": f"kind {kind!r} requires application/pdf, got {mime!r}",
                },
            )
        contents = file.file.read()
        stored_name = f"{uuid.uuid4().hex}_{file.filename or 'upload'}"
        dest = _storage_dir(asset.id) / stored_name
        dest.write_bytes(contents)
        attachment = Attachment(
            asset_id=asset.id, kind=kind_enum, title=title,
            filename=file.filename, mime=mime, size_bytes=len(contents),
            storage_path=str(dest), is_primary=is_primary, uploaded_by=user.id,
        )

    db.add(attachment)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "primary_already_set",
                "message": f"asset {asset_id} already has a primary attachment of kind {kind!r}",
            },
        )
    db.refresh(attachment)
    return AttachmentOut.model_validate(attachment)


@router.get("/assets/{asset_id}/attachments", response_model=list[AttachmentOut])
def list_attachments(
    asset_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> list[AttachmentOut]:
    _get_asset_or_404(db, asset_id)
    rows = db.scalars(
        select(Attachment).where(Attachment.asset_id == asset_id).order_by(Attachment.id)
    ).all()
    return [AttachmentOut.model_validate(r) for r in rows]


@router.get("/attachments/{attachment_id}/download")
def download_attachment(
    attachment_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    attachment = _get_attachment_or_404(db, attachment_id)
    if attachment.url is not None:
        return RedirectResponse(attachment.url)
    path = Path(attachment.storage_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "file_missing", "message": "the stored file no longer exists"},
        )
    return FileResponse(
        path, media_type=attachment.mime or "application/octet-stream",
        filename=attachment.filename or path.name,
    )


@router.delete("/attachments/{attachment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_attachment(
    attachment_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_role(UserRole.technician))],
) -> None:
    attachment = _get_attachment_or_404(db, attachment_id)
    if attachment.storage_path:
        Path(attachment.storage_path).unlink(missing_ok=True)
    db.delete(attachment)
    db.commit()
