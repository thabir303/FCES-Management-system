"""0001_initial -- the whole schema in one migration (§5).

Written as raw SQL rather than through Alembic's op.create_table helpers: the schema
includes generated columns (`assets.next_due_at`, `assets.search_tsv`), a partial unique
index (`attachments_one_primary_idx`), and CHECK constraints on expressions -- all things
the op.* helpers express awkwardly or not at all. Raw SQL keeps this migration a direct,
checkable transcription of §5 rather than a reconstruction of it.

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-20
"""
from __future__ import annotations

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.execute("CREATE TYPE user_role AS ENUM ('admin','technician','readonly')")
    op.execute(
        "CREATE TYPE attachment_kind AS ENUM "
        "('photo','pdf','document','link','risk_assessment','manual','certificate')"
    )
    op.execute(
        "CREATE TYPE import_status AS ENUM "
        "('pending','processing','ready_for_review','committed','failed')"
    )
    op.execute("CREATE TYPE import_route AS ENUM ('auto','review')")
    op.execute("CREATE TYPE dedup_call AS ENUM ('new','duplicate','uncertain')")
    op.execute("CREATE TYPE review_action AS ENUM ('accept','override','reject','skip')")

    op.execute("""
        CREATE TABLE users (
          id            BIGSERIAL PRIMARY KEY,
          email         TEXT UNIQUE NOT NULL,
          name          TEXT NOT NULL,
          password_hash TEXT NOT NULL,
          role          user_role NOT NULL DEFAULT 'readonly',
          is_active     BOOLEAN NOT NULL DEFAULT TRUE,
          created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE floorplans (
          id          BIGSERIAL PRIMARY KEY,
          building    TEXT NOT NULL,
          floor       TEXT NOT NULL,
          name        TEXT,
          image_path  TEXT NOT NULL,
          image_w     INT NOT NULL,
          image_h     INT NOT NULL,
          uploaded_by BIGINT REFERENCES users(id),
          created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
          UNIQUE (building, floor, name)
        )
    """)

    op.execute("""
        CREATE TABLE locations (
          id           BIGSERIAL PRIMARY KEY,
          building     TEXT,
          floor        TEXT,
          room         TEXT,
          label        TEXT,
          floorplan_id BIGINT REFERENCES floorplans(id) ON DELETE SET NULL,
          x_pct        NUMERIC(5,2) CHECK (x_pct BETWEEN 0 AND 100),
          y_pct        NUMERIC(5,2) CHECK (y_pct BETWEEN 0 AND 100)
        )
    """)

    op.execute("""
        CREATE TABLE categories (
          cpv_code                      TEXT PRIMARY KEY,
          cpv_description               TEXT NOT NULL,
          level                         SMALLINT NOT NULL CHECK (level IN (2,4,8)),
          parent_code                   TEXT REFERENCES categories(cpv_code),
          hazard_class                  TEXT,
          default_service_interval_days INT
        )
    """)

    op.execute("""
        CREATE TABLE assets (
          id                    BIGSERIAL PRIMARY KEY,
          public_id             UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
          asset_tag             TEXT UNIQUE NOT NULL,
          name                  TEXT NOT NULL,
          description           TEXT,
          manufacturer          TEXT,
          model                 TEXT,
          serial_number         TEXT,
          cpv_code              TEXT REFERENCES categories(cpv_code),
          location_id           BIGINT REFERENCES locations(id) ON DELETE SET NULL,
          owning_department     TEXT,
          service_interval_days INT,
          last_serviced_at      DATE,
          next_due_at           DATE GENERATED ALWAYS AS (
                                  CASE WHEN service_interval_days IS NOT NULL
                                        AND last_serviced_at IS NOT NULL
                                       THEN last_serviced_at + service_interval_days
                                  END) STORED,
          source_row_id         BIGINT,
          created_by            BIGINT REFERENCES users(id),
          created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
          updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
          search_tsv            TSVECTOR GENERATED ALWAYS AS (
            setweight(to_tsvector('english', coalesce(name,'')), 'A') ||
            setweight(to_tsvector('english', coalesce(manufacturer,'') || ' ' || coalesce(model,'')), 'B') ||
            setweight(to_tsvector('english', coalesce(description,'')), 'C')) STORED
        )
    """)
    op.execute("CREATE INDEX assets_search_idx ON assets USING GIN (search_tsv)")
    op.execute("CREATE INDEX assets_due_idx ON assets (next_due_at)")
    op.execute("CREATE INDEX assets_cpv_idx ON assets (cpv_code)")

    op.execute("""
        CREATE TABLE attachments (
          id           BIGSERIAL PRIMARY KEY,
          asset_id     BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          kind         attachment_kind NOT NULL,
          title        TEXT,
          filename     TEXT,
          mime         TEXT,
          size_bytes   BIGINT,
          storage_path TEXT,
          url          TEXT,
          is_primary   BOOLEAN NOT NULL DEFAULT FALSE,
          uploaded_by  BIGINT REFERENCES users(id),
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
          CHECK (storage_path IS NOT NULL OR url IS NOT NULL)
        )
    """)
    op.execute("CREATE INDEX attachments_asset_idx ON attachments (asset_id, kind)")
    op.execute("""
        CREATE UNIQUE INDEX attachments_one_primary_idx
          ON attachments (asset_id, kind) WHERE is_primary
    """)

    op.execute("""
        CREATE TABLE service_events (
          id           BIGSERIAL PRIMARY KEY,
          asset_id     BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          performed_at DATE NOT NULL,
          performed_by BIGINT REFERENCES users(id),
          provider     TEXT,
          notes        TEXT,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE audit_log (
          id          BIGSERIAL PRIMARY KEY,
          actor_id    BIGINT REFERENCES users(id),
          entity_type TEXT NOT NULL,
          entity_id   BIGINT NOT NULL,
          action      TEXT NOT NULL,
          before      JSONB,
          after       JSONB,
          at          TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX audit_entity_idx ON audit_log (entity_type, entity_id, at DESC)")

    op.execute("""
        CREATE TABLE notifications (
          id         BIGSERIAL PRIMARY KEY,
          asset_id   BIGINT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
          kind       TEXT NOT NULL CHECK (kind IN ('due_soon','overdue')),
          due_at     DATE NOT NULL,
          generated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          acknowledged_at TIMESTAMPTZ,
          acknowledged_by BIGINT REFERENCES users(id),
          UNIQUE (asset_id, kind, due_at)
        )
    """)
    op.execute("""
        CREATE INDEX notifications_open_idx ON notifications (kind, due_at)
          WHERE acknowledged_at IS NULL
    """)

    op.execute("""
        CREATE TABLE import_batches (
          id               BIGSERIAL PRIMARY KEY,
          filename         TEXT NOT NULL,
          uploaded_by      BIGINT REFERENCES users(id),
          row_count        INT,
          auto_count       INT,
          review_count     INT,
          precision_target NUMERIC(4,3) NOT NULL,
          column_mapping   JSONB NOT NULL,
          pipeline_run_id  TEXT,
          status           import_status NOT NULL DEFAULT 'pending',
          error            TEXT,
          created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE import_rows (
          id                       BIGSERIAL PRIMARY KEY,
          batch_id                 BIGINT NOT NULL REFERENCES import_batches(id) ON DELETE CASCADE,
          row_index                INT NOT NULL,
          raw                      JSONB NOT NULL,
          normalised               JSONB,
          dedup_decision           dedup_call,
          dedup_score              NUMERIC(5,4),
          dedup_candidate_asset_id BIGINT REFERENCES assets(id),
          class_cpv_code           TEXT,
          class_score              NUMERIC(5,4),
          class_alternatives       JSONB,
          evidence                 JSONB,
          route                    import_route NOT NULL,
          resolved_by              BIGINT REFERENCES users(id),
          resolved_at              TIMESTAMPTZ,
          final_asset_id           BIGINT REFERENCES assets(id),
          UNIQUE (batch_id, row_index)
        )
    """)
    op.execute("""
        CREATE INDEX import_rows_review_idx ON import_rows (batch_id, route)
          WHERE resolved_at IS NULL
    """)

    op.execute("""
        CREATE TABLE review_decisions (
          id             BIGSERIAL PRIMARY KEY,
          import_row_id  BIGINT NOT NULL REFERENCES import_rows(id) ON DELETE CASCADE,
          actor_id       BIGINT REFERENCES users(id),
          decision_type  TEXT NOT NULL CHECK (decision_type IN ('dedup','classification')),
          action         review_action NOT NULL,
          chosen_value   TEXT,
          seconds_taken  INT NOT NULL CHECK (seconds_taken >= 0),
          at             TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS review_decisions")
    op.execute("DROP TABLE IF EXISTS import_rows")
    op.execute("DROP TABLE IF EXISTS import_batches")
    op.execute("DROP TABLE IF EXISTS notifications")
    op.execute("DROP TABLE IF EXISTS audit_log")
    op.execute("DROP TABLE IF EXISTS service_events")
    op.execute("DROP TABLE IF EXISTS attachments")
    op.execute("DROP TABLE IF EXISTS assets")
    op.execute("DROP TABLE IF EXISTS categories")
    op.execute("DROP TABLE IF EXISTS locations")
    op.execute("DROP TABLE IF EXISTS floorplans")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TYPE IF EXISTS review_action")
    op.execute("DROP TYPE IF EXISTS dedup_call")
    op.execute("DROP TYPE IF EXISTS import_route")
    op.execute("DROP TYPE IF EXISTS import_status")
    op.execute("DROP TYPE IF EXISTS attachment_kind")
    op.execute("DROP TYPE IF EXISTS user_role")
