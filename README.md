# AeroNexus AI

**An aerospace engineering intelligence platform that turns drawings, sensor data, maintenance records, safety reports and
supplier documents into evidence-backed engineering insights and traceable business actions.**

AeroNexus combines technical-drawing intelligence, predictive maintenance, aviation safety analytics, engineering change
management and workflow automation. It uses multimodal extraction, machine learning trained on **open public aviation
datasets**, retrieval-augmented generation, a deterministic calculation engine and human-in-the-loop review. Every number
can be re-computed from a receipt, and every case can be replayed from its event log. Nothing is released without a named
approver.

| Workflow | Data it learns from | What it does |
|---|---|---|
| **Engineering change orders** (flagship) | Synthetic revision pairs + ERP/CRM seed | Compares two drawing revisions field by field and finds affected assemblies, orders, work orders, material POs, suppliers and customers. Computes cost and lead time with versioned formulas and produces an auditable ECR/ECO. |
| **Predictive maintenance** | **NASA C-MAPSS** (public, all 4 subsets) | Remaining useful life with a conformal 90% range, plus an engine health score. A versioned rule proposes inspection work orders for a technician to approve. |
| **Safety analytics** | **NTSB accident database** (public, 17,485 probable-cause statements) | Severity triage with abstention, similar-event retrieval and cause themes and trends. ASRS and FAA SDR exports can also be dropped in. |
| Drawing to quote | Synthetic drawings; MechVQA in Colab | Extraction cascade, finish/material checks, price with conformal interval, ERP/CRM drafts. |
| Directive tracker | Synthetic logbooks; MaintNet in Colab | Matches logbook evidence to ADs and works out when each is next due (effective-dated rules). |
| Invoice matching | Synthetic invoices; FATURA/DocILE in Colab | Three-way match with the price tolerance in force on the invoice date. |

## Results (latest run — real public data separated from synthetic)

| Suite | Data | Result |
|---|---|---|
| RUL, C-MAPSS FD001 / FD002 / FD003 / FD004 | **Public, real benchmark** (simulated engines) | Test RMSE **11.6 / 12.9 / 13.5 / 13.5** cycles; 90% ranges cover 89% / 90% / 82% / 92%; health AUROC 0.94–0.995 |
| Safety triage, NTSB (train 2008–15, test 2017–22) | **Public, real** | AUROC **0.92**, PR-AUC 0.66 at 12% prevalence, calibration error 0.017; masking outcome words changes AUROC by 0.004 |
| Change orders | Synthetic | Field-change F1 1.0, severity 100%; **100% of receipts re-compute exactly**, 100% of cases identical on replay |
| Drawing extraction | Synthetic (an upper bound) | F1 0.99; auto-accepted values right 99.8% of the time |
| Access control | Tests | 100% of role and tenant-isolation checks |

All **32 release gates** pass (`python -m aeronexus.evals.runner`). Synthetic-data scores are labelled as such in the UI and
in the report, because real scans and real fleets will be harder. The LLM-judge harness correctly **refuses** the
deliberately flawed demo judge as a gate: its agreement with humans is kappa 0.57, against 0.68 between two humans.

## Run it

```bash
cd backend
pip install -r requirements.txt
python scripts/generate_data.py --train 600     # synthetic golden sets + ERP/CRM seed (deterministic)
python scripts/fetch_public_data.py             # NASA C-MAPSS + NTSB, SHA-256 verified
python scripts/train_public_models.py           # RUL (4 subsets, ~3–4 min CPU) + safety model (~10 s)
AERONEXUS_STATE_DB=:memory: pytest -q           # 138 tests
AERONEXUS_STATE_DB=:memory: python -m aeronexus.evals.runner   # 32-check regression gate
uvicorn aeronexus.api:app --reload --port 8000

cd ../frontend && npm install && npm run dev    # http://localhost:3000
```

Or `docker compose up --build` (fetches, trains and gates during the build).

**Roles:** use the switcher at the bottom left: viewer, engineer, approver or admin. Only approvers can sign off, only admins
can run evaluations, and agents can never approve their own actions. In production, replace the demo headers with
OIDC (`aeronexus/security.py`).

## Free LLM: Llama on Groq (no OpenAI or Claude spend)

```bash
cd backend
cp .env.example .env            # then paste your free key from https://console.groq.com/keys into GROQ_API_KEY
python scripts/check_groq.py    # one tiny call per model: confirms the key, shows latency and remaining quota
uvicorn aeronexus.api:app --reload --port 8000
```

| Role | Default Groq model | Why |
|---|---|---|
| First pass (extraction fallback, grounded answers, chatbot) | `llama-3.1-8b-instant` | Fast and cheap on quota; most work never reaches an LLM anyway (rules → NER first) |
| Escalation (low confidence or invalid JSON) | `llama-3.3-70b-versatile` | Only when the 8B model is unsure |
| LLM judge | `openai/gpt-oss-20b` (open weights, also free on Groq) | A different model family, so the judge isn't grading its own family's answers |

How the system stays inside the free tier:
- **Pacing.** Requests are spaced to about 28 per minute (`AERONEXUS_GROQ_RPM`).
- **Rate limits.** A 429 waits for Groq's `retry-after` and doesn't trip the circuit breaker. A wait longer than 30 s degrades
  to rules, extractive answers and human review instead of hanging.
- **Caching and batching.** Responses are cached, and each drawing makes at most one batched LLM call.
- **No key, no problem.** Without a key the app still runs; LLM-only fields go to review.

Tests and CI always use the offline mock, even if `.env` selects Groq, so they never spend your quota. To evaluate the real
Llama models, run `python -m aeronexus.evals.runner` with your `.env` in place. The 32 gates then measure Llama itself, and
some thresholds may need re-calibrating with your own labels.

The predictive-maintenance and safety models are classic ML trained on public data. They don't use an LLM.

## Train in Colab, then test in VS Code

| Notebook | Runtime | What it trains |
|---|---|---|
| `05_cmapss_rul_tpu` | **TPU** (Keras 3 / JAX) | 1D-CNN + GRU RUL model on all C-MAPSS subsets, compared with the shipped baseline |
| `06_safety_text` | **TPU** (PyTorch/XLA) or GPU | DistilRoBERTa severity triage on the same NTSB temporal split; optional ASRS categories |
| `01_ner_tpu` | **TPU** or GPU | Drawing and logbook NER (+ MaintNet weak labels) |
| `02_vlm_lora` | GPU (L4/A100) | Qwen2.5-VL LoRA on MechVQA drawings |
| `03_detector` | GPU | YOLO layout detector (synthetic + Roboflow drawings) |
| `04_price_model` | CPU | Price model with a conformal interval |
| `07_opensky_trajectories` | CPU | Approach-anomaly detection (OpenSky: research use only) |

Set `REPO` in `notebooks/build_notebooks.py` to your fork, run it, then open the notebooks in Colab. A new model ships only
if the regression gate still passes. I executed notebook 04 and the windowing code of 05 against the real data. The others
need Colab hardware and were only syntax-checked, so expect to pin library versions.

## What's inside

```
backend/aeronexus/
  calc.py          units · gauge tables · versioned effective-dated rules · formulas with reproducible receipts
  cases.py         durable SQLite event log · status machine · resumable steps with retry · replay
  security.py      RBAC (viewer/engineer/approver/admin/agent) · tenant context
  public_data.py   dataset registry: source of record, license, terms, pinned SHA-256
  pdm/cmapss.py    regime normalisation · window features · GBT vs MLP · conformal RUL · IsolationForest health
  safety/ntsb.py   masked outcome words · temporal split · calibrated TF-IDF model · NMF themes · rule tags
  modules/         change_order (flagship) · pdm · safety · quote · pricing · mro · ap_match
  extraction/      regex rules · NER · layout · cascade with hallucination guard and cross-checks
  rag.py           hybrid BM25 + dense + RRF + MMR · revision-aware · contradiction detection · quarantine
  grounding.py     per-sentence citation verification · generation.py coverage-based abstention
  agents.py        tools by risk tier (ERP, CRM, CMMS, email, voice) · approvals · idempotency · audit
  evals/           metrics · judge harness · runner + thresholds.json (32 gates)
frontend/src/      React + TypeScript, red/white "redline" design; work queue, change-order case view, fleet, safety
notebooks/         Colab training (TPU/GPU) · docs/ architecture, datasets, roadmap, research, responsible AI, demo
```

Docs: [Architecture](docs/ARCHITECTURE.md) · [Datasets & licenses](docs/DATASETS.md) · [Roadmap vs vision](docs/ROADMAP.md) ·
[Research basis](docs/RESEARCH.md) · [Responsible AI](docs/RESPONSIBLE_AI.md) · [Demo script](docs/DEMO_SCRIPT.md)

## Before you publish the name

Two similarly named organisations turned up in a quick web search: a Northeastern University student club called
"Aero Nexus", and a Dubai company called "Aero Nex FZCO". Neither is an aerospace AI product. Before committing publicly,
check:
- the CIPO and USPTO trademark databases for "AeroNexus" in classes 9 and 42;
- the domain;
- the GitHub organisation name.
