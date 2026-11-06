# Phase 3 Acceptance Report

Date: 2026-11-06

## Scope completed

Phase 3 Steps 1–17 implemented for Advanced RAG + single Financial Research Agent:

| Criterion | Status |
| --- | --- |
| Typed config + Phase 3 deps | Pass |
| Pydantic research schemas | Pass |
| OpenRouter chat factory | Pass |
| Reuse OpenRouter embeddings | Pass |
| Namespace allowlist registry | Pass |
| Metadata filter builder | Pass |
| Query understanding + rewriter | Pass (LLM with heuristic fallback) |
| Dense retrieval + RRF | Pass |
| Hybrid BM25 | Gated — index metric is cosine; dense+RRF used; no silent migrate |
| Local BGE reranker | Pass (falls back to retrieval scores if model load fails) |
| Context expansion + citation registry | Pass |
| Decimal calculation engine | Pass |
| Context builder / prioritizer / compressor / token budget | Pass |
| Prompts + answer generation | Pass |
| LangChain internal tools + registry | Pass |
| Bounded single-agent loop (no LangGraph) | Pass |
| FastAPI SSE `POST /api/v1/research/stream` | Pass |

## Explicit non-goals (deferred)

- Phase 4 Mem0 / Guardrails AI packages
- Phase 5 LangSmith evaluation suites
- Silent Pinecone index recreation for hybrid (`dotproduct`) migration

## Live retrieval limitation

Phase 2 Pinecone ingest was incomplete at Phase 3 implementation time (embedding cache resume / OpenRouter free-tier 429s). Unit and mocked API tests pass without a full corpus. End-to-end retrieval QA remains gated on finishing upserts into `financial-research`.

## Hybrid index note

`HybridRetriever` inspects the live index metric:

- `dotproduct` → dense + sparse hybrid queries enabled
- otherwise (current Phase 2 cosine) → dense-only + multi-query RRF; hybrid remains disabled pending explicit migration approval

## How to run

```bash
uv sync
uv run pytest tests/ai/unit/test_phase3_schemas.py tests/ai/unit/test_phase3_rag_helpers.py tests/ai/unit/test_query_understanding_heuristic.py tests/ai/integration/test_research_api.py -q
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Compare NVIDIA and Microsoft revenue growth from 2021 to 2025.\",\"selected_companies\":[\"NVDA\",\"MSFT\"]}"
```
