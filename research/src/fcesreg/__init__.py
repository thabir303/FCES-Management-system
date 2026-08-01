"""fcesreg — the FCES register migration pipeline.

Pure Python over pandas DataFrames and numpy arrays. This package must never import
from ``system/``: it takes DataFrames and returns DataFrames, and the API layer adapts
between HTTP/DB and those DataFrames in exactly one file. No database access, and no
network access outside ``ingest_*`` and ``llm``.
"""

__version__ = "0.1.0"
