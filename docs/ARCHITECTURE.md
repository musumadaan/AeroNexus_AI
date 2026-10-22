# Architecture

## Platform view (AeroNexus)

```mermaid
flowchart LR
  subgraph Data["Data (with provenance)"]
    D1[Drawings · revisions] --- D2[C-MAPSS sensor histories] --- D3[NTSB / ASRS / SDR text] --- D4[Logbooks · invoices · ERP/CRM]
  end
  Data --> X[Extraction cascade + validation]
  Data --> M[Models: RUL · health · safety triage · price]
  X --> C[Deterministic calc engine<br/>versioned rules · receipts]
  M --> C
  X --> R[Revision-aware RAG<br/>contradictions · citations · abstention]
  C --> K[Case: durable event log<br/>verified · needs_review · blocked]
  R --> K
  K -->|approver| A[Approved → executed<br/>ECO · work order · notices]
  A --> AU[(Audit log · replay)]
  subgraph Assure
    G[32-check regression gate] --- J[Calibrated judge] --- RB[RBAC + tenant tests]
  end
```

The detailed drawing-extraction path is below.

```mermaid
flowchart LR
  subgraph Ingest
    U[PDF / scan / logbook / invoice] --> T[Text layer · OCR]
    T --> S[PII redaction · injection screen]
  end
  subgraph Perceive
    S --> L[Layout detector<br/>YOLO or heuristic]
  end
  subgraph Extract["Extract (cascade, cheapest first)"]
    L --> R[Regex rules] --> N[NER<br/>gazetteer → fine-tuned]
    N -->|only unresolved fields| V[VLM/LLM small → large]
    V --> G1[Verbatim guard<br/>value must exist on sheet]
    G1 --> X[Cross-checks<br/>title block vs BOM]
  end
  subgraph Reason["Ground & reason"]
    X --> K[Knowledge-graph rules]
    X --> P[Price model + conformal 90% interval]
    X --> H[Hybrid RAG<br/>BM25 + dense + RRF → rerank → MMR]
    H --> C[Grounded generation<br/>citations per sentence]
    C --> VF[Citation verifier<br/>numbers + support / NLI → prune or abstain]
  end
  subgraph Act["Act (agents)"]
    K & P & VF --> TR[Tool registry · risk tiers]
    TR -->|READ / DRAFT| ERPR[(ERP / CRM reads)]
    TR -->|WRITE / EXTERNAL| Q[[Approval queue<br/>idempotency keys]]
  end
  subgraph Human
    Q --> HU{Named approver}
    HU -->|approve| EX[ERP quote · CRM opp · email/voice RFI w/ AI disclosure]
    HU -->|corrections| AL[(Active-learning data)]
  end
  AL -.retrain in Colab.-> N & V
  subgraph Assure
    GS[Golden sets] --> EV[Eval runner] --> GT{Regression gate in CI}
    J[LLM judge calibrated on human labels<br/>kappa · CI · position audit · correction] --> EV
    OB[Tracing p50/p95 · cache · circuit breaker · PSI drift · audit log]
  end
```

## Request path for one drawing (latency budget)

| Stage | Typical cost | Why it's ordered this way |
|---|---|---|
| Layout + regex + NER | ~2 ms CPU (measured p95 on the golden set) | Deterministic, auditable, free. Resolves all but 1–2 of the 13 fields on most sheets. |
| LLM fallback (small model) | hundreds of ms for a local 3B model (estimate; measure with your provider) | Only for fields the rules can't read (title, odd layouts). One call per sheet, all missing fields batched. |
| Escalation to large model | seconds (estimate) | Only when the small model's confidence < 0.60 or its JSON fails validation. |
| Retrieval + grounded rationale | ~5 ms retrieval; LLM time as above | Cached per (model, normalized prompt). |
| Verification | < 1 ms lexical (measured); NLI adds a cross-encoder pass per sentence (not measured here) | Cheap enough to run on every answer. |

Levers, in the order they pay off: **cascade** (most fields never touch an LLM) → **batching** (one LLM call per
sheet) → **cache** (repeat drawings and FAQ chat are free) → **small-first routing** → **pixel caps for VLMs**
(`max_pixels` in notebook 02 bounds tokens per drawing) → **async workers** for uploads (uvicorn workers today; a queue
such as Celery/RQ when volume needs it).

## Trade-offs we chose deliberately

- **Wrong is worse than missing.** Rules refuse half-readable numbers ("12O") instead of guessing, so the field goes to
  review. The gate enforces ≥ 98% precision and ≥ 99% auto-accept precision even if recall suffers.
- **Rules before models.** Less impressive in a demo, far more reliable and cheaper in production; the LLM is a
  fallback with a verbatim guard, not the main reader.
- **Lexical verifier by default, NLI optional.** The lexical check (numbers must match + content recall) is strict and
  instant but misses paraphrase; NLI catches paraphrase at the cost of a model download and a forward pass per sentence. Both share one interface.
- **In-memory stores in the repo.** Swappable behind `services()` for Postgres + pgvector, Redis cache and a real
  ERPNext (`ERPNextAdapter`) without touching modules.
- **The judge is not trusted by default.** It must clear the human–human agreement bar (kappa, with CI) before it may
  gate releases; until then humans stay in the loop and the harness reports *why*.

## Failure behaviour

| Failure | What happens |
|---|---|
| LLM provider down / slow | Retries with backoff → circuit opens after 5 failures → system runs rules-only, LLM fields go to review, chatbot falls back to extractive answers. UI shows "paused, rules only". |
| Model returns invalid JSON | One repair round-trip, then the field is left for review. |
| Model invents a value | Discarded (not on the sheet), counter `extract.llm.hallucination_blocked`. |
| Document contains instructions to the AI | LLM fallback disabled for that document, no actions drafted, flagged as a security issue. |
| Retrieved passage contains instructions | Passage quarantined before it reaches a prompt. |
| Duplicate approval click | Idempotency key — executes once. |
| Traffic spike | Token-bucket rate limit (429) protects the model budget. |
