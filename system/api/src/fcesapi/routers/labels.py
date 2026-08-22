"""§5a. QR codes and printable labels for physical asset tagging (D6).

The QR encodes the asset's persistent item URL, `{base_url}/a/{public_id}` -- reusing the
existing `base_url` setting rather than adding a new one, since it is already exactly what
this needs and was sitting unused. `/a/{public_id}` is mounted directly on this API (no
frontend exists yet this session) so the encoded URL is one this API actually resolves,
not a promise a future frontend has to keep.

`GET /assets/{id}/label.svg` is the single printable label endpoint the project's scope
fence names (QR + Code128 barcode combined, one file) -- both rendered as PNG and embedded
as `<image>` elements inside one wrapping SVG canvas, since qrcode's and python-barcode's own
SVG output are two independent documents with incompatible viewports and cannot simply be
concatenated. `POST /assets/label-sheet` is the multi-asset PDF sheet requested this session,
which is broader than that single-endpoint fence -- flagged in the closing report rather than
built silently, since a written scope decision does not get quietly reopened by one session's
detailed instructions without saying so.
"""

from __future__ import annotations

import base64
import io
from typing import Annotated

import barcode
import qrcode
from barcode.writer import ImageWriter
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.config import get_settings
from fcesapi.db import get_db
from fcesapi.models import Asset, User
from fcesapi.schemas import AssetOut, LabelSheetRequest
from fcesapi.security import get_current_user

router = APIRouter(tags=["labels"])
#: Mounted WITHOUT the /api/v1 prefix (see main.py) -- a physical label's printed/scanned URL
#: stays short, and the persistent item URL this API hands out matches what it actually serves.
public_router = APIRouter(tags=["labels"])


def _get_or_404(db: Session, asset_id: int) -> Asset:
    asset = db.get(Asset, asset_id)
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": f"no asset with id {asset_id}"},
        )
    return asset


def asset_url(asset: Asset) -> str:
    """The persistent item URL a QR code / printed label encodes."""
    return f"{get_settings().base_url.rstrip('/')}/a/{asset.public_id}"


def _qr_png(data: str) -> bytes:
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _barcode_png(asset_tag: str) -> bytes:
    code = barcode.get("code128", asset_tag, writer=ImageWriter())
    buf = io.BytesIO()
    code.write(buf, options={"write_text": True})
    return buf.getvalue()


@public_router.get("/a/{public_id}", response_model=AssetOut)
def resolve_public_asset_url(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> AssetOut:
    """Requires authentication -- anonymous read is a limitation, not a feature (settled)."""
    asset = db.scalar(select(Asset).where(Asset.public_id == public_id))
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "asset_not_found", "message": "no asset with that public_id"},
        )
    return AssetOut.model_validate(asset)


@router.get("/assets/{asset_id}/qr.png")
def asset_qr_code(
    asset_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> Response:
    asset = _get_or_404(db, asset_id)
    return Response(content=_qr_png(asset_url(asset)), media_type="image/png")


@router.get("/assets/{asset_id}/label.svg")
def asset_label(
    asset_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> Response:
    """QR (persistent URL) + Code128 barcode (asset_tag) on one printable label."""
    asset = _get_or_404(db, asset_id)
    qr_b64 = base64.b64encode(_qr_png(asset_url(asset))).decode("ascii")
    bc_b64 = base64.b64encode(_barcode_png(asset.asset_tag)).decode("ascii")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="140" viewBox="0 0 300 140">
<image x="0" y="0" width="120" height="120" href="data:image/png;base64,{qr_b64}"/>
<image x="130" y="20" width="160" height="100" href="data:image/png;base64,{bc_b64}"/>
</svg>'''
    return Response(content=svg, media_type="image/svg+xml")


_COLS, _ROWS = 3, 8
_LABEL_W, _LABEL_H = A4[0] / _COLS, A4[1] / _ROWS


@router.post("/assets/label-sheet")
def label_sheet(
    body: LabelSheetRequest,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
) -> Response:
    """A printable multi-label PDF sheet for a batch of assets, one QR + tag per cell."""
    if not body.asset_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "empty_request", "message": "asset_ids must be non-empty"},
        )
    assets = [_get_or_404(db, i) for i in body.asset_ids]

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i, asset in enumerate(assets):
        cell = i % (_COLS * _ROWS)
        if i > 0 and cell == 0:
            c.showPage()
        col, row = cell % _COLS, cell // _COLS
        x0 = col * _LABEL_W
        y0 = A4[1] - (row + 1) * _LABEL_H

        qr_reader = ImageReader(io.BytesIO(_qr_png(asset_url(asset))))
        qr_size = min(_LABEL_W, _LABEL_H) - 8 * mm
        c.drawImage(qr_reader, x0 + 4 * mm, y0 + 4 * mm, width=qr_size, height=qr_size)
        c.setFont("Helvetica", 7)
        c.drawString(x0 + 4 * mm, y0 + _LABEL_H - 10, asset.asset_tag)
        c.drawString(x0 + 4 * mm, y0 + _LABEL_H - 20, (asset.name or "")[:28])
    c.save()
    return Response(content=buf.getvalue(), media_type="application/pdf")
