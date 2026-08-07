PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
PYENV   := $(HOME)/.pyenv/versions/3.12.8/bin/python3
TEXBIN  := /Library/TeX/texbin

.PHONY: bootstrap venv deps db migrate seed data annotate judge-distractors llm-pilot experiments tables paper dev test smoke clean-db

bootstrap: venv deps db migrate seed
	@echo "bootstrap: done"

venv:
	@test -d .venv || $(PYENV) -m venv .venv
	@$(PIP) install -q -U pip

deps: venv
	$(PIP) install -q -e "./research[dev]"
	$(PIP) install -q -e "./system/api[dev]"

db:
	docker compose up -d db
	@echo "waiting for postgres on 5433..."
	@until docker compose exec -T db pg_isready -U fces -d fces >/dev/null 2>&1; do sleep 1; done
	@echo "postgres ready"

# migrate and seed depend on Phase D. Until D1/D4 land they announce themselves
# and skip, rather than failing the bootstrap or pretending to have run.
migrate:
	@if ls system/api/alembic/versions/*.py >/dev/null 2>&1; then \
		cd system/api && ../../.venv/bin/alembic upgrade head; \
	else \
		echo "migrate: SKIPPED — no alembic revision yet (build D1)"; \
	fi

seed:
	@if [ -f system/api/scripts/seed_categories.py ]; then \
		$(PY) system/api/scripts/seed_users.py && \
		$(PY) system/api/scripts/seed_categories.py; \
	else \
		echo "seed: SKIPPED — seed scripts not built yet (D2, D4)"; \
	fi

data:
	$(PY) -m fcesreg.ingest_contractsfinder --raw data/raw --years 2022 2023 2024 2025 \
		--out data/processed/corpus_b_contractsfinder.parquet
	$(PY) -m fcesreg.ingest_abtbuy --dest data/raw/abtbuy \
		--out data/processed/corpus_a_abtbuy.parquet
	$(PY) research/scripts/build_taxonomy.py
	$(PY) research/scripts/freeze_splits.py

# The timed annotation exercise. Produces BOTH the label-noise labels and the
# mean_seconds_per_item figure that run_operating_point.py consumes (§6.15, §13.3).
# Run this before `make experiments`.
annotate:
	$(PY) annotation/annotate.py

# Every mined distractor judged by hand. C6 and the transfer runner consume this set,
# so they wait on it being complete.
judge-distractors:
	$(PY) annotation/judge_distractors.py

# C5 acceptance: a live pilot, then proof the identical re-run is free. Needs GROQ_API_KEY.
# Not part of `experiments` — an acceptance check is not a measurement of the method.
llm-pilot:
	$(PY) research/scripts/run_llm_pilot.py --config research/configs/llm.yaml

experiments:
	$(PY) research/scripts/run_profile.py         --config research/configs/profile.yaml
	$(PY) research/scripts/audit_real_errors.py   --config research/configs/audit.yaml
	$(PY) research/scripts/run_degrade_check.py   --config research/configs/degrade.yaml
	$(PY) research/scripts/run_blocking.py        --config research/configs/blocking.yaml
	$(PY) research/scripts/run_label_noise.py     --config research/configs/label_noise.yaml
	$(PY) research/scripts/run_dedup.py           --config research/configs/dedup_abtbuy.yaml
	$(PY) research/scripts/run_dedup.py           --config research/configs/dedup_cf_sweep.yaml
	$(PY) research/scripts/run_classify.py        --config research/configs/classify.yaml
	$(PY) research/scripts/run_costs.py           --config research/configs/costs.yaml
	$(PY) research/scripts/run_operating_point.py --config research/configs/operating_point.yaml

tables:
	$(PY) research/scripts/make_tables.py

# A clean two-pass build is a precondition for every paper commit, in the same way
# make_tables refuses a dirty git tree. No bibtex: the bibliography is inline.
paper:
	@PATH="$(TEXBIN):$$PATH"; \
	 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && \
	 pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && \
	 echo "paper: clean build, $$(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $$2}') pages" || \
	 { echo "paper: BUILD FAILED — see main.log"; exit 1; }

dev:
	cd system/api && ../../.venv/bin/uvicorn fcesapi.main:app --reload --port 8000 & \
	cd system/web && npm run dev

test:
	$(PYTEST) research/tests -q
	@if [ -d system/api/tests ]; then $(PYTEST) system/api/tests -q; \
	 else echo "api tests: none yet"; fi

smoke:
	$(PY) research/scripts/run_dedup.py --config research/configs/smoke.yaml

clean-db:
	docker compose down -v
	rm -rf .pgdata
