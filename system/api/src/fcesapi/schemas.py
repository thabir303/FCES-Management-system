"""Pydantic request/response models (§7). JSON keys stay snake_case -- no translation at
the API boundary, matching the SQL column names exactly."""

from __future__ import annotations

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    detail: ErrorDetail


# --------------------------------------------------------------------------- auth


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: str
    name: str
    role: str
    is_active: bool


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------------------------------------------------------- assets


class AssetCreate(BaseModel):
    name: str
    description: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    cpv_code: str | None = None
    location_id: int | None = None
    owning_department: str | None = None
    service_interval_days: int | None = None
    last_serviced_at: date | None = None


class AssetUpdate(BaseModel):
    """Every field optional: PATCH semantics, not PUT -- only supplied fields are written."""

    name: str | None = None
    description: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    cpv_code: str | None = None
    location_id: int | None = None
    owning_department: str | None = None
    service_interval_days: int | None = None
    last_serviced_at: date | None = None


class AssetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    public_id: UUID
    asset_tag: str
    name: str
    description: str | None
    manufacturer: str | None
    model: str | None
    serial_number: str | None
    cpv_code: str | None
    location_id: int | None
    owning_department: str | None
    service_interval_days: int | None
    last_serviced_at: date | None
    next_due_at: date | None
    created_at: datetime
    updated_at: datetime


class AssetListOut(BaseModel):
    items: list[AssetOut]
    total: int
    page: int
    page_size: int


# --------------------------------------------------------------------------- imports


class ImportBatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    filename: str
    row_count: int | None
    auto_count: int | None
    review_count: int | None
    status: str
    error: str | None
    created_at: datetime


class ImportUploadOut(BaseModel):
    batch_id: int
    detected_columns: list[str]
    sample_rows: list[dict]


class ImportMappingRequest(BaseModel):
    column_mapping: dict[str, str]
    precision_target: float = 0.95


class ImportRowOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    row_index: int
    raw: dict
    normalised: dict | None
    dedup_decision: str | None
    dedup_score: float | None
    dedup_candidate_asset_id: int | None
    class_cpv_code: str | None
    class_score: float | None
    class_alternatives: list | None
    evidence: dict | None
    route: str
    resolved_at: datetime | None
    final_asset_id: int | None


class ResolveRowRequest(BaseModel):
    action: str  # accept | override | reject | skip
    decision_type: str  # dedup | classification
    chosen_cpv_code: str | None = None
    chosen_asset_id: int | None = None
    seconds_taken: int


# --------------------------------------------------------------------------- attachments


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    asset_id: int
    kind: str
    title: str | None
    filename: str | None
    mime: str | None
    size_bytes: int | None
    url: str | None
    is_primary: bool
    created_at: datetime


# --------------------------------------------------------------------------- floor plans


class FloorPlanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    building: str
    floor: str
    name: str | None
    image_w: int
    image_h: int
    created_at: datetime


class PinCreate(BaseModel):
    x_pct: float
    y_pct: float
    label: str | None = None
    asset_id: int | None = None


class PinUpdate(BaseModel):
    """PATCH semantics: only supplied fields move or relink the pin."""

    x_pct: float | None = None
    y_pct: float | None = None
    label: str | None = None
    asset_id: int | None = None


class PinOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    floorplan_id: int | None
    building: str | None
    floor: str | None
    room: str | None
    label: str | None
    x_pct: float | None
    y_pct: float | None


# --------------------------------------------------------------------------- labels


class LabelSheetRequest(BaseModel):
    asset_ids: list[int]


# --------------------------------------------------------------------------- service reminders


class ServiceEventCreate(BaseModel):
    performed_at: date
    provider: str | None = None
    notes: str | None = None


class ServiceEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    asset_id: int
    performed_at: date
    provider: str | None
    notes: str | None
    created_at: datetime


class NotificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    asset_id: int
    kind: str
    due_at: date
    generated_at: datetime
    acknowledged_at: datetime | None


# --------------------------------------------------------------------------- audit log


class AuditLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    actor_id: int | None
    entity_type: str
    entity_id: int
    action: str
    before: dict | None
    after: dict | None
    at: datetime


class AuditLogListOut(BaseModel):
    items: list[AuditLogOut]
    total: int
    page: int
    page_size: int
