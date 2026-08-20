"""ORM models mirroring the Alembic migration exactly (§5).

`asset_status` is deliberately absent -- the scope fence removes the soft-delete lifecycle
state, and neither the client brief nor the paper claims one (§1.1). No model here should
grow a `status`/`deleted_at`/`is_deleted` field without that decision being reopened first.

Generated columns (`assets.next_due_at`, `assets.search_tsv`) are declared with
`server_default=None` and excluded from INSERT/UPDATE -- Postgres computes them, and the
ORM must never attempt to write to a column the database owns.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fcesapi.db import Base

#: Native Postgres enum columns, matched by NAME to the types the migration creates.
#: create_type=False because the migration owns type creation -- the ORM must not try to
#: (re)create a type that CREATE TYPE already made, which would fail on every checkout.
def _enum(python_enum, pg_name: str):
    return PgEnum(python_enum, name=pg_name, create_type=False, values_callable=lambda e: [m.value for m in e])


class UserRole(str, enum.Enum):
    admin = "admin"
    technician = "technician"
    readonly = "readonly"


class AttachmentKind(str, enum.Enum):
    photo = "photo"
    pdf = "pdf"
    document = "document"
    link = "link"
    risk_assessment = "risk_assessment"
    manual = "manual"
    certificate = "certificate"


class ImportStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    ready_for_review = "ready_for_review"
    committed = "committed"
    failed = "failed"


class ImportRoute(str, enum.Enum):
    auto = "auto"
    review = "review"


class DedupCall(str, enum.Enum):
    new = "new"
    duplicate = "duplicate"
    uncertain = "uncertain"


class ReviewAction(str, enum.Enum):
    accept = "accept"
    override = "override"
    reject = "reject"
    skip = "skip"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, "user_role"), nullable=False, default=UserRole.readonly
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FloorPlan(Base):
    __tablename__ = "floorplans"
    __table_args__ = (UniqueConstraint("building", "floor", "name"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    building: Mapped[str] = mapped_column(Text, nullable=False)
    floor: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text, nullable=False)
    image_w: Mapped[int] = mapped_column(Integer, nullable=False)
    image_h: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Location(Base):
    __tablename__ = "locations"
    __table_args__ = (
        CheckConstraint("x_pct BETWEEN 0 AND 100", name="locations_x_pct_check"),
        CheckConstraint("y_pct BETWEEN 0 AND 100", name="locations_y_pct_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    building: Mapped[str | None] = mapped_column(Text)
    floor: Mapped[str | None] = mapped_column(Text)
    room: Mapped[str | None] = mapped_column(Text)
    label: Mapped[str | None] = mapped_column(Text)
    floorplan_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("floorplans.id", ondelete="SET NULL")
    )
    # Percentage coordinates, not pixels -- the plan image can be re-exported at any
    # resolution and pins stay correct.
    x_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))
    y_pct: Mapped[float | None] = mapped_column(Numeric(5, 2))


class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (CheckConstraint("level IN (2,4,8)", name="categories_level_check"),)

    cpv_code: Mapped[str] = mapped_column(Text, primary_key=True)
    cpv_description: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    parent_code: Mapped[str | None] = mapped_column(Text, ForeignKey("categories.cpv_code"))
    # Illustrative designation only -- not a hazard taxonomy. See seed_categories.py.
    hazard_class: Mapped[str | None] = mapped_column(Text)
    default_service_interval_days: Mapped[int | None] = mapped_column(Integer)


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    public_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), unique=True, nullable=False, server_default=func.gen_random_uuid()
    )
    asset_tag: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    manufacturer: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    serial_number: Mapped[str | None] = mapped_column(Text)
    cpv_code: Mapped[str | None] = mapped_column(Text, ForeignKey("categories.cpv_code"))
    location_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("locations.id", ondelete="SET NULL")
    )
    owning_department: Mapped[str | None] = mapped_column(Text)
    service_interval_days: Mapped[int | None] = mapped_column(Integer)
    last_serviced_at: Mapped[date | None] = mapped_column(Date)
    # GENERATED ALWAYS in the migration -- Computed() tells the ORM this column is
    # DB-owned, so it is never included in an INSERT or UPDATE. The expression given here
    # is documentation only: SQLAlchemy does not execute it, since the migration (raw SQL)
    # owns DDL and this class is never used with create_all().
    next_due_at: Mapped[date | None] = mapped_column(
        Date,
        Computed(
            "CASE WHEN service_interval_days IS NOT NULL AND last_serviced_at IS NOT NULL "
            "THEN last_serviced_at + service_interval_days END",
            persisted=True,
        ),
    )
    source_row_id: Mapped[int | None] = mapped_column(BigInteger)
    created_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    search_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(name,'')), 'A') || "
            "setweight(to_tsvector('english', coalesce(manufacturer,'') || ' ' || "
            "coalesce(model,'')), 'B') || "
            "setweight(to_tsvector('english', coalesce(description,'')), 'C')",
            persisted=True,
        ),
    )


class Attachment(Base):
    __tablename__ = "attachments"
    __table_args__ = (
        CheckConstraint(
            "storage_path IS NOT NULL OR url IS NOT NULL", name="attachments_path_or_url"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[AttachmentKind] = mapped_column(_enum(AttachmentKind, "attachment_kind"), nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(Text)
    mime: Mapped[str | None] = mapped_column(Text)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    storage_path: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ServiceEvent(Base):
    __tablename__ = "service_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    performed_at: Mapped[date] = mapped_column(Date, nullable=False)
    performed_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    provider: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    before: Mapped[dict | None] = mapped_column(JSONB)
    after: Mapped[dict | None] = mapped_column(JSONB)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (UniqueConstraint("asset_id", "kind", "due_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asset_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    due_at: Mapped[date] = mapped_column(Date, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))


class ImportBatch(Base):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    row_count: Mapped[int | None] = mapped_column(Integer)
    auto_count: Mapped[int | None] = mapped_column(Integer)
    review_count: Mapped[int | None] = mapped_column(Integer)
    precision_target: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    column_mapping: Mapped[dict] = mapped_column(JSONB, nullable=False)
    pipeline_run_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[ImportStatus] = mapped_column(
        _enum(ImportStatus, "import_status"), nullable=False, default=ImportStatus.pending
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    rows: Mapped[list["ImportRow"]] = relationship(back_populates="batch")


class ImportRow(Base):
    __tablename__ = "import_rows"
    __table_args__ = (UniqueConstraint("batch_id", "row_index"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("import_batches.id", ondelete="CASCADE"), nullable=False
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    raw: Mapped[dict] = mapped_column(JSONB, nullable=False)
    normalised: Mapped[dict | None] = mapped_column(JSONB)
    dedup_decision: Mapped[DedupCall | None] = mapped_column(_enum(DedupCall, "dedup_call"))
    dedup_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    dedup_candidate_asset_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("assets.id")
    )
    class_cpv_code: Mapped[str | None] = mapped_column(Text)
    class_score: Mapped[float | None] = mapped_column(Numeric(5, 4))
    class_alternatives: Mapped[list | None] = mapped_column(JSONB)
    evidence: Mapped[dict | None] = mapped_column(JSONB)
    route: Mapped[ImportRoute] = mapped_column(_enum(ImportRoute, "import_route"), nullable=False)
    resolved_by: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    final_asset_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("assets.id"))

    batch: Mapped["ImportBatch"] = relationship(back_populates="rows")


class ReviewDecision(Base):
    """The six review-queue logging fields the supervisor requires live across this table
    and `ImportRow`, joined by `import_row_id` rather than duplicated onto one table:

    1. why the record entered review    -- derivable from `ImportRow` (dedup_decision,
                                            class_score vs. threshold, route) at the moment
                                            it was routed; not re-stored here, since the
                                            reason is a property of the routing decision,
                                            not of a later resolution.
    2. time spent                       -- seconds_taken
    3. decision made                    -- action
    4. automated suggestion accepted?   -- action == 'accept'
    5. category or match changed?       -- action == 'override' + chosen_value
    6. confidence on the original decision -- ImportRow.dedup_score / class_score, joined
                                               via import_row_id

    A queue that did not record these from the first resolved row has no history to
    recover, so this table exists before any review UI does.
    """

    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    import_row_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("import_rows.id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"))
    decision_type: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[ReviewAction] = mapped_column(_enum(ReviewAction, "review_action"), nullable=False)
    chosen_value: Mapped[str | None] = mapped_column(Text)
    seconds_taken: Mapped[int] = mapped_column(Integer, nullable=False)
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("seconds_taken >= 0", name="review_decisions_seconds_check"),
        CheckConstraint(
            "decision_type IN ('dedup','classification')",
            name="review_decisions_type_check",
        ),
    )
