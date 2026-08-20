"""Application configuration (§12.3).

**Every setting comes from here, never from `os.getenv` inline.** A call site that reads
the environment directly is invisible to this file and untestable without setting real
process environment variables; a `Settings` object is overridable in a test fixture with
one line. This is the boundary's other half — `fcesreg` is not allowed to import this class
(the `system/` -> `fcesreg` direction is the only one that exists), so it reads
`GROQ_API_KEY` through its own accessor in `llm.py` rather than through this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_root() -> Path:
    """Walk up from this file looking for the repo-root marker.

    **A bare relative default is wrong from anywhere but one specific working directory.**
    `storage_root` defaults to a plain ``"storage"`` string once; measured concretely: the
    fitted-classifier joblib cache written from a manual run at the repo root landed at
    ``storage/models/classifier.joblib`` there, and the same default under pytest (run from
    `system/api/`) landed at `system/api/storage/models/classifier.joblib` instead --
    a silent cache miss, refitting from scratch rather than an error. `fcesreg.paths`
    exists to prevent exactly this defect on the research side; this is its counterpart
    here, anchored to `docker-compose.yml` rather than `data/processed/`.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docker-compose.yml").exists():
            return parent
    raise RuntimeError(f"could not locate repo root (docker-compose.yml) from {here}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres on 5433, not 5432 -- avoids clashing with a local install (docker-compose.yml).
    database_url: str = Field(
        default="postgresql+psycopg://fces:fces@localhost:5433/fces",
        alias="DATABASE_URL",
    )
    jwt_secret: str = Field(default="dev-secret-change-me", alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60 * 12

    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    # One-shot seed credentials -- read here rather than via os.getenv in the seed script,
    # for the same reason every other setting is: untestable and invisible otherwise.
    # Empty by default so a seed run without one set skips that user rather than seeding a
    # guessable password.
    seed_admin_password: str = Field(default="", alias="SEED_ADMIN_PASSWORD")
    seed_technician_password: str = Field(default="", alias="SEED_TECHNICIAN_PASSWORD")
    seed_readonly_password: str = Field(default="", alias="SEED_READONLY_PASSWORD")

    # Anchored to the repo root, not the working directory -- see _repo_root().
    storage_root: str = Field(
        default_factory=lambda: str(_repo_root() / "storage"), alias="STORAGE_ROOT"
    )
    base_url: str = Field(default="http://localhost:3000", alias="BASE_URL")


@lru_cache
def get_settings() -> Settings:
    """Cached: the environment does not change mid-process, and every dependency that
    needs a setting should see the same object rather than re-parsing `.env` each time."""
    return Settings()
