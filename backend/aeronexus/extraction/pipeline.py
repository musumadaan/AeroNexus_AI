"""Extraction cascade: layout → regex → NER → LLM (only for leftovers) → verify → route.

Every field comes out with: value, calibrated confidence, which layer produced it, the source
token ids (for UI highlighting) and a routing status:
  auto_accepted  (conf ≥ auto_accept)       → flows straight into the ERP draft
  needs_review   (escalate ≤ conf < auto)   → human reviewer confirms
  missing        (nothing trustworthy found) → reviewer fills it or an RFI goes to the customer
LLM outputs must be *verbatim-supported* by the sheet text, otherwise they are discarded as
hallucinations — the model may read, but it may not invent.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config import settings
from ..llm import LLMError, LLMRequest, MockLLM, get_router
from ..observability import tracer
from .document import Sheet, get_detector
from .ner import get_ner
from .regex_rules import looks_like_injection, run_rules

QUOTE_FIELDS = [
    "drawing_number", "part_number", "title", "revision", "material", "thickness", "coating",
    "quantity", "diameter", "length", "tolerance", "standard", "scale",
]

# Which region a field normally lives in (layout prior). Values found elsewhere are down-weighted.
REGION_PRIOR: dict[str, set[str]] = {
    "drawing_number": {"title_block"}, "part_number": {"title_block", "bom_table"}, "title": {"title_block"},
    "revision": {"title_block", "revision_table"}, "scale": {"title_block"}, "material": {"title_block", "bom_table", "notes"},
    "thickness": {"title_block", "bom_table", "notes", "view"}, "coating": {"notes", "title_block"},
    "quantity": {"bom_table", "title_block"}, "diameter": {"view"}, "length": {"view", "bom_table"},
    "tolerance": {"title_block", "notes", "view"}, "standard": {"notes", "title_block"},
}
NER_FIELD = {"MATERIAL": "material", "COATING": "coating"}
FIELD_LABELS = {"thickness": {"THK", "THICK", "THICKNESS"}, "quantity": {"QTY", "QUANTITY"}, "revision": {"REV", "REVISION"},
                "scale": {"SCALE"}}


@dataclass
class FieldResult:
    name: str
    value: str | None
    confidence: float
    source: str  # regex | ner | llm | none
    rule: str = ""
    token_ids: list[str] = field(default_factory=list)
    status: str = "missing"
    alternatives: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class ExtractionResult:
    sheet_id: str
    fields: dict[str, FieldResult]
    regions: list[dict[str, Any]]
    warnings: list[str]
    llm_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet_id": self.sheet_id, "regions": self.regions, "warnings": self.warnings, "llm_calls": self.llm_calls,
            "fields": {k: asdict(v) for k, v in self.fields.items()},
        }


def _status(conf: float) -> str:
    if conf >= settings.auto_accept_conf:
        return "auto_accepted"
    if conf >= 0.4:
        return "needs_review"
    return "missing"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class ExtractionPipeline:
    def __init__(self, fields: list[str] | None = None, use_llm: bool = True) -> None:
        self.fields = fields or QUOTE_FIELDS
        self.use_llm = use_llm
        self.detector = get_detector()
        self.ner = get_ner()

    def run(self, sheet: Sheet) -> ExtractionResult:
        warnings: list[str] = []
        with tracer.span("extract.layout"):
            sheet.regions = self.detector(sheet)
        region_of = {tid: r.label for r in sheet.regions for tid in r.token_ids}
        text, offsets = sheet.text_with_offsets()
        if looks_like_injection(text):
            warnings.append("Possible prompt-injection text on sheet; LLM fallback disabled for this document.")

        candidates: dict[str, list[FieldResult]] = {f: [] for f in self.fields}

        with tracer.span("extract.regex"):
            for m in run_rules(text, self.fields):
                tids = sheet.tokens_for_span(offsets, m.start, m.end)
                in_prior = any(region_of.get(t) in REGION_PRIOR.get(m.field, set()) for t in tids)
                conf = m.confidence * (1.0 if in_prior else 0.85)
                candidates[m.field].append(FieldResult(m.field, m.value, conf, "regex", m.rule, tids))

        with tracer.span("extract.ner"):
            for ent in self.ner(text):
                fname = NER_FIELD.get(ent.label)
                if fname in candidates:
                    tids = sheet.tokens_for_span(offsets, ent.start, ent.end)
                    candidates[fname].append(FieldResult(fname, ent.text.upper(), ent.confidence * 0.92, "ner", ent.source, tids))

        results: dict[str, FieldResult] = {}
        for fname, cands in candidates.items():
            results[fname] = self._resolve(fname, cands)

        # Label-without-value check: the title block prints "THK"/"QTY"/… but OCR destroyed the value,
        # so any value we found elsewhere on the sheet (e.g. a local note) must not be auto-accepted.
        title = next((r for r in sheet.regions if r.label == "title_block"), None)
        if title:
            title_ids = set(title.token_ids)
            title_text = {sheet.token(t).text.upper().rstrip(":") for t in title.token_ids}
            for fname, labels in FIELD_LABELS.items():
                r = results.get(fname)
                if r and r.value and title_text & labels and not (set(r.token_ids) & title_ids):
                    r.confidence = round(r.confidence * 0.6, 3)
                    r.note = "title-block value unreadable; value taken from elsewhere on the sheet"
                    tracer.incr("extract.crosscheck.label_without_value")

        # Redundancy cross-check: the same fact printed twice (title block QTY vs BOM row) must agree.
        # Cheap, deterministic, and it catches OCR turning one valid number into another.
        q = results.get("quantity")
        bom = next((r for r in sheet.regions if r.label == "bom_table"), None)
        if q and q.value and bom:
            bom_vals = {sheet.token(t).text for t in bom.token_ids}
            if q.value in bom_vals:
                q.confidence = round(min(0.99, q.confidence + 0.02), 3)
            else:
                q.confidence = round(q.confidence * 0.7, 3)
                q.note = "title-block QTY disagrees with BOM table"
                tracer.incr("extract.crosscheck.qty_mismatch")

        llm_calls = 0
        missing = [f for f, r in results.items() if r.value is None]
        if missing and self.use_llm and not warnings:
            llm_calls = 1
            for fname, r in self._llm_fallback(sheet, missing).items():
                results[fname] = r

        for r in results.values():
            r.status = _status(r.confidence) if r.value is not None else "missing"
        tracer.incr("extract.fields.auto", sum(r.status == "auto_accepted" for r in results.values()))
        tracer.incr("extract.fields.review", sum(r.status == "needs_review" for r in results.values()))
        return ExtractionResult(sheet.id, results, sheet.to_dict()["regions"], warnings, llm_calls)

    @staticmethod
    def _resolve(fname: str, cands: list[FieldResult]) -> FieldResult:
        if not cands:
            return FieldResult(fname, None, 0.0, "none")
        # agreement voting: same normalized value from several rules/layers boosts confidence,
        # disagreement between strong candidates lowers it (ambiguity → human review).
        groups: dict[str, list[FieldResult]] = {}
        for c in sorted(cands, key=lambda c: -len(_norm(c.value or ""))):
            key = _norm(c.value or "")
            # "GALVANIZED" (NER) inside "HOT-DIP GALVANIZED PER ASTM A123" (regex) = agreement, not conflict
            parent = next((k for k in groups if key and key in k), None)
            groups.setdefault(parent or key, []).append(c)
        ranked = sorted(groups.values(), key=lambda g: (any(c.source == "regex" for c in g), max(c.confidence for c in g), len(g)),
                        reverse=True)
        best_group = ranked[0]
        best = max(best_group, key=lambda c: (c.source == "regex", c.confidence))
        conf = max(c.confidence for c in best_group)
        if len({c.source for c in best_group}) > 1 or len(best_group) > 1:
            conf = min(0.99, conf + 0.03)
        alts = [g[0].value for g in ranked[1:] if g[0].value]
        rival_regex = [c for g in ranked[1:] for c in g if c.source == "regex" and c.confidence > 0.8]
        if rival_regex and fname not in {"tolerance", "standard", "diameter"}:
            conf *= 0.8  # two strong, different rule-based values: a human must pick
            best.note = "conflicting values on sheet"
        tids = sorted({t for c in best_group for t in c.token_ids})
        return FieldResult(fname, best.value, round(conf, 3), best.source, best.rule, tids, alternatives=alts[:3], note=best.note)

    def _llm_fallback(self, sheet: Sheet, missing: list[str]) -> dict[str, FieldResult]:
        title_ids = next((r.token_ids for r in sheet.regions if r.label == "title_block"), [])
        context, offsets = sheet.text_with_offsets(title_ids or None)
        full_text, full_offsets = sheet.text_with_offsets()
        req = LLMRequest(
            task="extract_fields",
            prompt=(
                "Extract the following fields from this engineering drawing text. Copy values EXACTLY as written; "
                f"use null if absent. Fields: {missing}.\nReturn JSON: {{\"fields\": {{...}}, \"confidence\": 0-1}}\n\n"
                f"TITLE BLOCK:\n{context}\n\nFULL SHEET:\n{full_text[:4000]}"
            ),
            payload={"missing": missing, "text": full_text},
            json_schema={"fields": "dict", "confidence": "float"},
        )
        out: dict[str, FieldResult] = {}
        try:
            with tracer.span("extract.llm_fallback", n_fields=len(missing)):
                resp = get_router().complete(req)
        except LLMError:
            tracer.incr("extract.llm_fallback.failed")
            return out
        vals = (resp.parsed or {}).get("fields", {}) or {}
        conf = resp.confidence or 0.5
        for fname in missing:
            v = vals.get(fname)
            if not v:
                continue
            idx = full_text.lower().find(str(v).lower())
            if idx < 0:  # hallucination guard: value must exist verbatim on the sheet
                tracer.incr("extract.llm.hallucination_blocked")
                continue
            tids = sheet.tokens_for_span(full_offsets, idx, idx + len(str(v)))
            out[fname] = FieldResult(fname, str(v), round(min(conf, 0.85), 3), "llm", resp.model, tids,
                                     note="LLM fallback; verified verbatim on sheet")
        return out


# ---- deterministic mock behaviour for offline runs -------------------------------------------
@MockLLM.register("extract_fields")
def _mock_extract(req: LLMRequest, model: str) -> str:
    text: str = req.payload.get("text", "")
    fields: dict[str, Any] = {}
    for f in req.payload.get("missing", []):
        if f == "title":
            m = re.search(r"TITLE\s*:?\s*(.+?)(?=\s+(?:DWG|DRAWING|P/N|PART|REV|SCALE|MATERIAL|MATL|QTY|DRAWN)\b|$)", text, re.I)
            if m:
                fields["title"] = m.group(1).strip()
    # the "small" model is deliberately less confident so the cascade is exercised in tests
    conf = 0.55 if "3b" in model or "small" in model else 0.82
    return json.dumps({"fields": fields, "confidence": conf})
