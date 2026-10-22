"""Document model + layout/vision layer.

A `Sheet` is a page as a list of OCR tokens with bounding boxes (the same shape PaddleOCR /
docTR / Tesseract / a PDF text layer give you). Keeping boxes end-to-end is what lets the UI
highlight exactly where every extracted value came from (provenance = trust).

Layout detection finds regions (title block, BOM table, notes, views). Default is a fast
geometric heuristic; if a YOLO model trained in notebooks/03_detector.ipynb is exported to
backend/models/detector.pt and `ultralytics` is installed, it is used instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import MODELS


@dataclass
class Token:
    id: str
    text: str
    bbox: tuple[float, float, float, float]  # x, y, w, h in sheet units (0..1000)
    page: int = 0


@dataclass
class Region:
    label: str  # title_block | bom_table | notes | view | revision_table
    bbox: tuple[float, float, float, float]
    token_ids: list[str]
    confidence: float


@dataclass
class Sheet:
    id: str
    tokens: list[Token]
    width: float = 1000
    height: float = 707  # A-series landscape aspect
    meta: dict[str, Any] = field(default_factory=dict)
    regions: list[Region] = field(default_factory=list)

    # text with a char-offset → token index so regex/NER spans map back to boxes
    def text_with_offsets(self, token_ids: list[str] | None = None) -> tuple[str, list[tuple[int, int, str]]]:
        parts, offsets, pos = [], [], 0
        allowed = set(token_ids) if token_ids is not None else None
        for t in self.tokens:
            if allowed is not None and t.id not in allowed:
                continue
            parts.append(t.text)
            offsets.append((pos, pos + len(t.text), t.id))
            pos += len(t.text) + 1
        return " ".join(parts), offsets

    def tokens_for_span(self, offsets: list[tuple[int, int, str]], start: int, end: int) -> list[str]:
        return [tid for s, e, tid in offsets if s < end and e > start]

    def token(self, tid: str) -> Token:
        return next(t for t in self.tokens if t.id == tid)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sheet":
        toks = [Token(t["id"], t["text"], tuple(t["bbox"]), t.get("page", 0)) for t in d["tokens"]]
        return cls(id=d["id"], tokens=toks, width=d.get("width", 1000), height=d.get("height", 707), meta=d.get("meta", {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "width": self.width, "height": self.height, "meta": self.meta,
            "tokens": [{"id": t.id, "text": t.text, "bbox": list(t.bbox), "page": t.page} for t in self.tokens],
            "regions": [{"label": r.label, "bbox": list(r.bbox), "token_ids": r.token_ids, "confidence": r.confidence}
                        for r in self.regions],
        }


def _union(tokens: list[Token]) -> tuple[float, float, float, float]:
    x0 = min(t.bbox[0] for t in tokens)
    y0 = min(t.bbox[1] for t in tokens)
    x1 = max(t.bbox[0] + t.bbox[2] for t in tokens)
    y1 = max(t.bbox[1] + t.bbox[3] for t in tokens)
    return (x0, y0, x1 - x0, y1 - y0)


class HeuristicLayoutDetector:
    """Title block = bottom-right block; BOM = rows under an ITEM/QTY header; notes = under NOTES."""

    def __call__(self, sheet: Sheet) -> list[Region]:
        W, H = sheet.width, sheet.height
        regions: list[Region] = []
        title = [t for t in sheet.tokens if t.bbox[0] > 0.55 * W and t.bbox[1] > 0.72 * H]
        if title:
            regions.append(Region("title_block", _union(title), [t.id for t in title], 0.9))
        header = next((t for t in sheet.tokens if t.text.upper() in {"ITEM", "BOM", "PARTS LIST"}), None)
        if header:
            hx, hy = header.bbox[0], header.bbox[1]
            rows = [t for t in sheet.tokens if t.bbox[1] >= hy - 2 and t.bbox[1] < hy + 0.3 * H
                    and t.bbox[0] >= hx - 5 and t.id not in {i for r in regions for i in r.token_ids}]
            if rows:
                regions.append(Region("bom_table", _union(rows), [t.id for t in rows], 0.85))
        notes_hdr = next((t for t in sheet.tokens if t.text.upper().startswith("NOTES")), None)
        if notes_hdr:
            used = {i for r in regions for i in r.token_ids}
            notes = [t for t in sheet.tokens if t.bbox[1] >= notes_hdr.bbox[1] and abs(t.bbox[0] - notes_hdr.bbox[0]) < 0.4 * W
                     and t.id not in used]
            if notes:
                regions.append(Region("notes", _union(notes), [t.id for t in notes], 0.8))
        used = {i for r in regions for i in r.token_ids}
        view = [t for t in sheet.tokens if t.id not in used]
        if view:
            regions.append(Region("view", _union(view), [t.id for t in view], 0.7))
        return regions


class YoloLayoutDetector:  # pragma: no cover - needs trained weights
    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))

    def __call__(self, sheet: Sheet) -> list[Region]:
        img = sheet.meta.get("image_path")
        if not img:
            return HeuristicLayoutDetector()(sheet)
        res = self.model(img, verbose=False)[0]
        out = []
        for box, cls, conf in zip(res.boxes.xywh.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            x, y, w, h = box
            rb = (x - w / 2, y - h / 2, w, h)
            ids = [t.id for t in sheet.tokens if rb[0] <= t.bbox[0] <= rb[0] + w and rb[1] <= t.bbox[1] <= rb[1] + h]
            out.append(Region(self.model.names[int(cls)], rb, ids, float(conf)))
        return out


def get_detector():
    w = MODELS / "detector.pt"
    if w.exists():
        try:
            return YoloLayoutDetector(w)
        except Exception:
            pass
    return HeuristicLayoutDetector()


def ocr_pdf(path: str) -> list[Sheet]:  # pragma: no cover - optional dependency path
    """Real-document entry point: PDF text layer via pdfplumber; falls back to Tesseract OCR."""
    import pdfplumber

    sheets = []
    with pdfplumber.open(path) as pdf:
        for p, page in enumerate(pdf.pages):
            s = 1000 / page.width  # normalise to a 1000-unit-wide sheet
            words = page.extract_words() or []
            if not words:  # scanned page: OCR it
                import pytesseract

                img = page.to_image(resolution=200).original
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                px = page.width / img.width  # image px → PDF points
                words = [{"text": t, "x0": data["left"][i] * px, "top": data["top"][i] * px,
                          "x1": (data["left"][i] + data["width"][i]) * px,
                          "bottom": (data["top"][i] + data["height"][i]) * px}
                         for i, t in enumerate(data["text"]) if t.strip()]
            toks = [Token(f"p{p}t{i}", w["text"], (w["x0"] * s, w["top"] * s, (w["x1"] - w["x0"]) * s,
                                                    (w["bottom"] - w["top"]) * s), p) for i, w in enumerate(words)]
            sy = s
            sheets.append(Sheet(id=f"{Path(path).stem}-p{p}", tokens=toks, width=1000, height=page.height * sy))
    return sheets


def load_sheets(path: Path) -> list[Sheet]:
    return [Sheet.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
