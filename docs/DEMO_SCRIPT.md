# Demo video script (about 7 minutes)

Set up first:
- Run `make fetch train` once, then `make api` and `make web`.
- Record at 1440×900.
- In the bottom-left switcher, set your name and the **Approver** role.

1. **Overview (40 s).** "AeroNexus turns aerospace documents and data into traceable decisions." Read the headline numbers
   (real-data ones first) and click each workflow to light up its stages.
2. **Change orders, the flagship (2 min).**
   - Compare CH000 (a thickness change). The case opens as *Verified*, severity *Critical* because the part is in production.
   - Walk through the affected work order, sales order, material PO and customer, then the cost total and lead time.
   - Click the thickness row: the revision clouds mark the value on both sheets.
   - Show the receipts. Approve with a note: the case becomes *Executed* and the ECO is released.
   - Click **Re-check from event log**: "identical, calculations re-computed exactly". Download the .md report.
3. **Predictive maintenance (1 min).** FD001: RMSE 11.6 cycles on NASA's public data, 89% of engines inside the 90% range.
   Click the top engine to show its sensor trend and the rule reasons, then **Propose inspection work order**. Switch the
   role to Viewer: you can no longer approve.
4. **Safety analytics (1 min).** Score the stall narrative: *high priority*, with similar NTSB events. Explain the abstain
   band and the masking of outcome words. Show the trends and the cause themes.
5. **Work queue + approvals (30 s).** Filter by status. Approve the maintenance work order as Approver.
6. **Evaluation (1 min).** Real-data tables (C-MAPSS, NTSB) next to synthetic ones, each labelled. Point out the receipt and
   replay gates and the judge being refused as a gate.
7. **Data and rules (20 s).** Licenses and checksums; the price-tolerance rule v1 → v2 effective July 2026. Mention that the
   same invoice is judged by the rule in force on its date.
8. **Engineering (20 s).** Run `pytest -q` (138 tests) and `python -m aeronexus.evals.runner` (32 gates), and show Colab
   notebook 05 running on a TPU.
