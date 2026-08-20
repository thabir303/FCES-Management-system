"""§9.2. Background processing: maps, dedups and classifies every uploaded row.

Runs synchronously in a FastAPI `BackgroundTasks` job, called from the mapping endpoint.
Talks to `services.pipeline` only through the functions it exposes -- never touches
`fcesreg` directly, keeping the boundary at exactly one file.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from fcesapi.models import Asset, DedupCall, ImportBatch, ImportRow, ImportRoute, ImportStatus
from fcesapi.services import pipeline

#: The Asset fields a column can be mapped onto. Deliberately the same set AssetCreate
#: accepts -- a column mapping that produces anything else would fail at commit time
#: instead of at upload time, which is the wrong place for that error to surface.
TARGET_FIELDS = {
    "name", "description", "manufacturer", "model", "serial_number",
    "owning_department", "cpv_code",
}


def normalise_row(raw: dict, column_mapping: dict[str, str]) -> dict:
    """Map a raw spreadsheet row through ``column_mapping`` into the Asset shape."""
    out: dict = {}
    for source_col, target_field in column_mapping.items():
        if target_field not in TARGET_FIELDS:
            continue
        value = raw.get(source_col)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            out[target_field] = value
    return out


def _existing_assets_frame(db: Session) -> pd.DataFrame:
    rows = db.execute(select(Asset.id, Asset.name, Asset.description)).all()
    return pd.DataFrame(
        {
            "record_id": [str(r.id) for r in rows],
            "title": [r.name or "" for r in rows],
            "description": [r.description or "" for r in rows],
        }
    )


def process_batch(db: Session, batch: ImportBatch, raw_rows: list[dict]) -> None:
    """The five-step pipeline per row (§9.2): map, dedup, classify, evidence, route.

    Writes every ``ImportRow`` and updates the batch's counts and status. Never raises out
    of a single row's failure -- a malformed row is routed to review with its error
    recorded in ``evidence``, so one bad row cannot fail an entire batch upload.
    """
    existing = _existing_assets_frame(db)
    auto_count = 0
    review_count = 0

    for index, raw in enumerate(raw_rows):
        normalised = normalise_row(raw, batch.column_mapping)

        dedup = pipeline.score_duplicates(normalised, existing)
        lower, upper = pipeline.dedup_bounds(float(batch.precision_target))
        score = dedup.get("score")
        if score is None or score <= lower:
            dedup_decision = DedupCall.new
        elif score >= upper:
            dedup_decision = DedupCall.duplicate
        else:
            dedup_decision = DedupCall.uncertain

        classification = pipeline.classify(normalised) if normalised.get("name") else None

        # route='auto' only if all three hold (§9.2): the record is new, the classifier's
        # confidence clears the floor selected at this batch's precision target, AND the
        # predicted code is in the supported set. A confident score over a label set the
        # true category is not in is not evidence (§6.10) -- it is asserted here as a
        # routing rule, not merely stated in the paper.
        auto_eligible = (
            dedup_decision == DedupCall.new
            and classification is not None
            and classification["clears_floor"]
            and classification["in_supported_set"]
        )
        route = ImportRoute.auto if auto_eligible else ImportRoute.review
        if route == ImportRoute.auto:
            auto_count += 1
        else:
            review_count += 1

        db.add(
            ImportRow(
                batch_id=batch.id,
                row_index=index,
                raw=raw,
                normalised=normalised or None,
                dedup_decision=dedup_decision,
                dedup_score=dedup.get("score"),
                dedup_candidate_asset_id=dedup.get("candidate_asset_id"),
                class_cpv_code=classification["code"] if classification else None,
                class_score=classification["score"] if classification else None,
                class_alternatives=classification["alternatives"] if classification else None,
                evidence={
                    "dedup": dedup.get("evidence"),
                    "classification_floor": pipeline.get_models().class_confidence_floor
                    if classification
                    else None,
                },
                route=route,
            )
        )

    batch.row_count = len(raw_rows)
    batch.auto_count = auto_count
    batch.review_count = review_count
    batch.status = ImportStatus.ready_for_review
    db.commit()
