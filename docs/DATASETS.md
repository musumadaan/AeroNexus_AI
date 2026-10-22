# Datasets and licenses

Every dataset is registered in `backend/aeronexus/public_data.py` with its **source of record**, **license**, **terms** and
(for auto-fetched files) a pinned **SHA-256**. A checksum mismatch aborts the fetch rather than training on changed data.
"Verified" below means I checked the fact against the source page while building this (October 2026).

## Trained on in this repo (real public data)

| Dataset | Size used | License / terms | Used for | How it gets here |
|---|---|---|---|---|
| **NASA C-MAPSS** turbofan run-to-failure, FD001–FD004 | 709 training and 707 test engines | US Government work (public domain in the US); cite Saxena et al. 2008. Simulated engines, not a real fleet. | RUL with conformal ranges, health score, work-order rule | `fetch_public_data.py` (GitHub mirror of the NASA PCoE files, checksummed). Source of record: data.nasa.gov |
| **NTSB aviation accident database** (extract 1948–2022, 88,890 events) | 17,485 events with probable-cause text (2008–2022) | US Government work (public domain in the US). The extract is a community redistribution, so refresh from NTSB CAROL for current data. | Severity triage, similar-event retrieval, cause themes and trends | `fetch_public_data.py` (checksummed) |

## Supported by loaders and notebooks (download or export yourself)

| Dataset | License / terms (verified) | Used for |
|---|---|---|
| **NASA ASRS** narratives | Sanitized, de-identified; **NASA does not verify or validate reports**; CSV export ≤10,000 records per download | Retrieval (labelled "unverified report"), notebook 06 categories |
| **FAA Service Difficulty Reports** | Yearly CSVs, 1995–2026; operator and repair-station submissions, not verified failure rates | Component trend retrieval |
| **OpenSky Network** ADS-B | Limited, terminable license: non-profit research and education, government, or *internal* commercial testing only; no redistribution; cite OpenSky | Notebook 07 only; never stored in the repo |
| **MechVQA** (Hugging Face `XiaofengAlg/MechVQA`) | Apache-2.0; 3.3K drawings, about 21K QA pairs | Drawing VLM LoRA (notebook 02) |
| **MaintNet** aviation logbooks | Open research library (check repository terms); 6,169 aviation entries | NER robustness (notebook 01) |
| **FATURA** / **DocILE** invoices | FATURA freely accessible (10K invoices); DocILE **non-commercial** | Invoice layout models |
| Roboflow engineering-drawing sets | Per project | Layout detector (notebook 03) |

## Synthetic corpus (license-free, deterministic)

`scripts/generate_data.py` produces:
- aerospace drawings: 2024-T3, 7075-T6, 6061-T6, TI-6AL-4V, SS 304 and ASTM A36 parts, with MIL/AMS finishes;
- revision pairs;
- ERP data: suppliers, inventory, material POs, work orders, where-used;
- CRM accounts;
- logbooks with synthetic ADs;
- invoices;
- judge labels.

It deliberately includes the hard cases: OCR damage, numbers OCR turns into other valid numbers, conflicting values,
unreadable title-block values, prompt injection and process-incompatible finishes. **Synthetic results are an upper
bound and are always labelled synthetic in the UI and the report.**

## Good practice built in
- Splits by source unit: C-MAPSS by engine id, NTSB by year. Never by row, because that leaks.
- The golden sets are frozen and never trained on; the training corpus is generated separately.
- Every report records provenance per suite (`report["provenance"]`).
- Model cards: each notebook writes `model_card.json` (data, metrics, intended use, limitations).
