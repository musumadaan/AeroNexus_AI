# Vision vs. what is built

The platform vision (your two briefs) is broader than any one repo. This table is the honest map: **built** means it runs
here and is tested; **partial** means the mechanism exists but not at enterprise scope; **planned** means it is designed
but not implemented.

| Capability | Status | Where / next step |
|---|---|---|
| Two-revision change impact → auditable ECR/ECO (the recommended starting point) | **Built** | `modules/change_order.py`, `/change`. Covers where-used, SO/WO/material POs, inventory, supplier lead time, cost receipts, report export, approval → execution |
| Deterministic calculation engine (units, tolerances, effective-dated rules, reproducible receipts) | **Built** | `calc.py`; the gate requires 100% receipt reproducibility |
| Explicit statuses (verified / needs review / blocked / approved / executed), durable state, replay, resumable retries | **Built** | `cases.py` (SQLite event log; Postgres DSN in prod) |
| RBAC, tenant isolation, agents can't approve | **Built** | `security.py`; demo headers → OIDC/JWT next |
| Predictive maintenance (RUL, health index) on NASA C-MAPSS | **Built** | `pdm/cmapss.py`; deep TPU model in notebook 05 |
| Aviation safety analytics on NTSB (triage, similar events, trends) | **Built** | `safety/ntsb.py`; ASRS/SDR exports supported; transformer in notebook 06 |
| Revision-aware RAG, contradiction detection, citation verification, abstention | **Built** | `rag.py`, `grounding.py`, `generation.py` |
| Knowledge graph (materials, finishes, suppliers, parts → assemblies) | **Partial** | In-memory adjacency; move to Neo4j/Postgres with work orders, incidents and ADs as nodes |
| Multi-page PDF, OCR, layout, GD&T | **Partial** | Text-layer/Tesseract path + YOLO notebook; GD&T feature-control-frame parsing is planned |
| CAD/PLM/PDM integration (digital thread) | **Planned** | Adapter interface exists (`ERPNextAdapter` pattern); add PLM part/revision sync |
| SAP, SharePoint, M365, Power BI, webhooks | **Planned** | Same adapter pattern; idempotent tools and dry-run already in place |
| Supplier identity resolution, tax/currency validation, contract-price checks | **Partial** | FX and price rules in `calc.py`; supplier dedup is planned |
| Flight-trajectory analytics (OpenSky) | **Partial** | Notebook 07 (research use only, by license) |
| Object storage, background workers, IaC, backups | **Planned** | Docker + CI today; add S3/MinIO, a worker queue and Terraform |
| Digital twin, what-if simulation, supply-chain risk | **Planned** | Build on the receipt-based calc engine so simulations stay reproducible |
