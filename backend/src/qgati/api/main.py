"""FastAPI application entrypoint.

Run with: uv run uvicorn qgati.api.main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Q-Gati API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
