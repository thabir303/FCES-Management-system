---
paths:
  - "system/**"
---

# Rules for the delivered system

**`services/pipeline.py` is the only file in `system/` that may import `fcesreg`.**
`grep -r "import fcesreg" system/` must return exactly that one file. This is what makes the
deployment claim true rather than decorative.

**Configuration comes from a Pydantic `Settings` class, never `os.getenv` inline** (§12.3):
`DATABASE_URL`, `JWT_SECRET`, `GROQ_API_KEY`, `STORAGE_ROOT`, `BASE_URL`. **Postgres is on
5433**, not 5432, and `pgcrypto` is required for `gen_random_uuid()`.

This rule is `system/`-only. `fcesreg` cannot use that `Settings` class without importing from
`system/`, which the boundary forbids, so it reads `GROQ_API_KEY` through one accessor in
`llm.py` — never at import time, never inline at a call site.

**Error envelope** (§12.2): `{"detail": {"code": "asset_not_found", "message": "..."}}`. Never leak
a stack trace. 401 unauthenticated, **403 authenticated-but-wrong-role** (a `readonly` user issuing
a write gets 403, not 401), 404 missing, 409 conflict, 422 validation.

**The scope fence (§1.1) is a fence, not a starting point.** Do not build: a dashboard route, a user
administration interface (users are seeded by script), an audit *browsing* interface (the log is
claimed, a viewer is not), Playwright coverage, a trigram fuzzy search path, soft delete, the
four-value status enum, any delete endpoint beyond a floor-plan pin (§ below), `value_gbp`,
`purchase_date`, or `POST /auth/logout`.

**Amended 2026-08-22, by explicit supervisor instruction that session:** the fence originally read
"a label sheet builder beyond the single printable label endpoint." `GET /assets/{id}/label.svg`
(QR + Code128, one file) still satisfies that as written. `POST /assets/label-sheet` — a multi-asset
PDF sheet — was added the same session on direct instruction, is broader than the original fence,
and was flagged as a scope reopening rather than built silently. This line records that reopening
so the rule matches the repository instead of contradicting it. Two pin endpoints
(`DELETE /floorplans/{id}/pins/{pin_id}`) were added the same session — a floor-plan pin is
metadata about where an asset sits, not the asset-lifecycle deletion §13's "deletion and retention
are out of scope" and "no delete endpoint" refer to; no asset delete endpoint exists.

**Settled decisions (§13).** `/a/{public_id}` requires authentication — anonymous read is a
limitation in the report, not a feature. `hazard_class` is an illustrative designation on divisions
33, 38 and 42 only, marked as such; do not design a hazard vocabulary. Deletion and retention are
out of scope.

**One search mechanism**: `search_tsv` with its GIN index. **Service reminders need a scheduled
job** (§5.8) writing `notifications` rows for due-soon and overdue items, surfaced on the service
view and on login — a view a user must visit is not a reminder. No email, no external service, no
preferences UI. Idempotent within a day.

**Import routing.** `route='auto'` only if all three hold: `dedup_decision='new'`, `class_score`
clears the threshold at `precision_target`, **and** the predicted code is in the supported label
set. A confident score over a label set the true category is not in is not evidence.

**Protection order if short of time**, cutting from the bottom: import wizard, review queue,
`fcesreg`/`system` boundary, experiment runners, provenance machinery.
