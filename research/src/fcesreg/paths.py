"""Repository-anchored paths.

**No path inside this package may resolve against the current working directory.** A bare
relative path like ``Path("data/lexicon/abbreviations.yaml")`` is correct from the
repository root and wrong from anywhere else, so ``cd research && pytest`` — the most
natural thing a developer does — fails with 14 errors that read as broken code rather than
as a configuration difference.

The root is discovered by **walking upward from ``__file__`` for a marker file**, rather
than by counting parent directories. Counting (``Path(__file__).parents[3]``) is shorter
but silently wrong the moment the package moves within the tree; the marker survives that,
and it also survives the package being installed somewhere else entirely, because in that
case the walk fails loudly and names the override rather than guessing.

``FCES_ROOT`` overrides the search, which is what a non-editable install or a relocated
data directory needs.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "ENV_VAR",
    "ROOT_MARKERS",
    "RootNotFound",
    "repo_root",
    "data_path",
    "results_path",
    "annotation_path",
]

ENV_VAR = "FCES_ROOT"

#: Files that exist only at the repository root. ``PROJECT_PLAN.md`` is the specification
#: and ``main.tex`` is the paper; neither appears anywhere else in the tree.
ROOT_MARKERS = ("PROJECT_PLAN.md", "main.tex", ".git")


class RootNotFound(RuntimeError):
    """The repository root could not be located, and no override was given."""


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The repository root, discovered once and cached.

    Order: ``FCES_ROOT`` if set, then an upward walk from this file for a marker.
    """
    override = os.environ.get(ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise RootNotFound(f"{ENV_VAR}={override!r} is not a directory")
        return root

    for candidate in Path(__file__).resolve().parents:
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate

    raise RootNotFound(
        f"could not locate the repository root above {__file__}. Looked for "
        f"{list(ROOT_MARKERS)}. Set {ENV_VAR} to the repository root — this happens when "
        "fcesreg is installed outside the tree it reads its data from."
    )


def data_path(*parts: str) -> Path:
    """A path under ``<root>/data``."""
    return repo_root().joinpath("data", *parts)


def results_path(*parts: str) -> Path:
    """A path under ``<root>/results``."""
    return repo_root().joinpath("results", *parts)


def annotation_path(*parts: str) -> Path:
    """A path under ``<root>/annotation``."""
    return repo_root().joinpath("annotation", *parts)
