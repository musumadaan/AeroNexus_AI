# Responsible AI notes (model card style)

**Intended use.** Decision support for estimators, maintenance controllers and AP clerks. The system drafts; people decide.

**Not intended for.** Unreviewed quoting, airworthiness determinations, removing engines from service, accident
investigation findings, or releasing payments. Predictive maintenance and safety triage are decision support: a certifying
technician or safety analyst decides.

**Public-data caveats.** C-MAPSS engines are simulated, so real-fleet RUL error will be higher. NTSB probable causes are
written after an investigation, so live triage on early narratives will be weaker than the test AUROC. ASRS reports are
unverified, and SDRs are not failure rates. The UI labels every result with its data provenance.

**Access control.** Viewer, engineer, approver and admin roles; agents run as a principal that can never approve; cases
are tenant-isolated (tested). Production: replace demo headers with OIDC/JWT.

**Reproducibility.** Every cost, lead-time, severity, due-date and tolerance decision is produced by a versioned formula and
an effective-dated rule. Its receipt can be re-computed later, and replay of the event log must equal the live state.
Both are release gates.

**Human oversight.** Every write to ERP/CRM and every outbound message is an approval request with a named approver,
an idempotency key and an audit-log entry. Agents cannot approve their own actions (enforced in code and tested).

**Transparency.** Every extracted value shows where on the drawing it came from and which layer produced it. Every
generated sentence cites a source and is verified; unsupported sentences are removed or the system abstains.
Customer emails and calls disclose that an AI assistant prepared them.

**Privacy.** PII (emails, phone numbers, card-like numbers) is redacted before logging. Customer drawings are never sent
to a third-party API unless `AERONEXUS_LLM_PROVIDER` is set to one; the default runs locally.

**Security.** Prompt-injection screening on documents and retrieved passages; strict tool-argument validation;
GraphQL depth limit; rate limiting; CORS allow-list; non-root container.

**Known limitations.**
- The golden set is synthetic; expect lower accuracy on real scans until you add real labelled drawings.
- OCR can turn one valid number into another (8 → 3). Single-source fields like thickness can then be auto-accepted
  wrongly; the eval reports this explicitly (2 of ~620 auto-accepted values on the golden set). Mitigations: approval
  before any quote is sent, redundancy cross-checks where a value is printed twice, and the VLM second read.
- The default lexical verifier can reject faithful paraphrases (false negatives → more abstentions, not false claims).
- The mock judge is deliberately flawed to exercise the harness; a real judge must be calibrated on your own labels.

**Fairness.** No decisions are made about people. Vendor anomaly scores compare a vendor only with its own history.
