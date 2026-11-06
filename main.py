"""FastAPI application entrypoint for the financial research backend."""

from __future__ import annotations

from fastapi import FastAPI

from ai.api.research import router as research_router

app = FastAPI(
    title="Financial Research Agentic RAG",
    version="0.3.0",
    description="Phase 3 single-agent financial research API",
)
app.include_router(research_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    # Render/local: bind 0.0.0.0 and honor PORT when present.
    import os

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
