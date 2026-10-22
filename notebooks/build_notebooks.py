"""Builds the Colab notebooks from plain Python cell lists (easier to review/diff than raw .ipynb JSON).

Run: python notebooks/build_notebooks.py   → notebooks/*.ipynb
Cells starting with '# %% [markdown]' become markdown cells.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/<you>/aeronexus-ai.git"  # ← change to your fork


def nb(cells: list[str], accelerator: str | None) -> dict:
    out = []
    for c in cells:
        c = c.strip("\n")
        if c.startswith("# %% [markdown]"):
            src = c.replace("# %% [markdown]\n", "", 1)
            out.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": c.splitlines(keepends=True)})
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if accelerator:
        meta["accelerator"] = accelerator
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": out}


SETUP = f'''
# Get the repo, regenerate the synthetic corpus (deterministic, ~5 s) and fetch the public datasets (checksummed)
!git clone -q {REPO} /content/aeronexus-ai || true
%cd /content/aeronexus-ai/backend
!pip -q install -r requirements.txt
!python scripts/generate_data.py --train 4000
!python scripts/fetch_public_data.py
'''

# ---------------------------------------------------------------------------------------------
NER = [
    """# %% [markdown]
# 01 · Fine-tune the NER model (TPU or GPU)

Trains a token classifier for drawing entities (MATERIAL, COATING, PART_NUMBER, STANDARD, THICKNESS, DIMENSION)
and logbook entities. It replaces the gazetteer in layer 2 of the extraction cascade.

**Runtime:** *Runtime → Change runtime type → TPU* (v5e/v6e). The cell below installs PyTorch/XLA so the Hugging Face
Trainer runs on the TPU. If the TPU image refuses the install, switch to a **T4 GPU** runtime — every other cell is
unchanged. A DistilRoBERTa NER model trains in a few minutes on either.

**Data:** the repo's BIO-tagged synthetic drawings (license-free) + optionally the MaintNet aviation logbook set
(COLING 2020, open source) for abbreviation-heavy maintenance text.""",
    SETUP,
    """# TPU support for PyTorch (skip on GPU). Versions must match the preinstalled torch.
import importlib, subprocess, sys
try:
    import jax; tpu = any(d.platform == "tpu" for d in jax.devices())
except Exception:
    tpu = False
if tpu:
    import torch
    v = torch.__version__.split("+")[0]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torch_xla[tpu]=={v}",
                    "-f", "https://storage.googleapis.com/libtpu-releases/index.html"], check=False)
!pip -q install transformers datasets accelerate seqeval evaluate
print("TPU runtime:", tpu)""",
    """import json, random
from datasets import Dataset, DatasetDict

rows = [json.loads(l) for l in open("data/train/ner_bio.jsonl")]
random.Random(0).shuffle(rows)
labels = sorted({t for r in rows for t in r["tags"]}, key=lambda t: (t != "O", t))
label2id = {l: i for i, l in enumerate(labels)}; id2label = {i: l for l, i in label2id.items()}
n = len(rows)
ds = DatasetDict({
    "train": Dataset.from_list(rows[: int(.8 * n)]),
    "validation": Dataset.from_list(rows[int(.8 * n): int(.9 * n)]),
    "test": Dataset.from_list(rows[int(.9 * n):]),
})
print(ds, labels)""",
    """# (optional) add MaintNet aviation logbook text for domain robustness.
# Download the aviation CSV from the MaintNet repository (see docs/DATASETS.md), upload it, then weak-label it with
# the repo's gazetteer so the model learns abbreviations/misspellings it has never seen in drawings.
import os
if os.path.exists("/content/maintnet_aviation.csv"):
    import pandas as pd, sys
    sys.path.insert(0, ".")
    from aeronexus.extraction.ner import GazetteerNER
    g = GazetteerNER()
    df = pd.read_csv("/content/maintnet_aviation.csv")
    text_col = next(c for c in df.columns if df[c].dtype == object)
    extra = []
    for txt in df[text_col].dropna().astype(str).head(5000):
        words = txt.split(); tags = ["O"] * len(words); pos = 0
        for i, w in enumerate(words):
            start = txt.find(w, pos); pos = start + len(w)
            for e in g(txt):
                if e.start <= start < e.end and e.label in {"COMPONENT", "ACTION"}:
                    tags[i] = ("B-" if start == e.start else "I-") + e.label
        extra.append({"id": f"mn{len(extra)}", "tokens": words, "tags": tags})
    for l in {t for r in extra for t in r["tags"]} - set(label2id):
        label2id[l] = len(label2id); id2label[label2id[l]] = l
    from datasets import concatenate_datasets
    ds["train"] = concatenate_datasets([ds["train"], Dataset.from_list(extra)])
    print("added MaintNet rows:", len(extra))""",
    """from transformers import AutoTokenizer
MODEL = "distilroberta-base"
tok = AutoTokenizer.from_pretrained(MODEL, add_prefix_space=True)

def encode(batch):
    enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=256)
    out = []
    for i, tags in enumerate(batch["tags"]):
        prev, lab = None, []
        for wid in enc.word_ids(i):
            lab.append(-100 if wid is None or wid == prev else label2id[tags[wid]])  # label first sub-token only
            prev = wid
        out.append(lab)
    enc["labels"] = out
    return enc

tokenized = ds.map(encode, batched=True, remove_columns=ds["train"].column_names)""",
    """import numpy as np, evaluate
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
seqeval = evaluate.load("seqeval")
model = AutoModelForTokenClassification.from_pretrained(MODEL, num_labels=len(label2id), id2label=id2label, label2id=label2id)

def metrics(p):
    preds, labs = np.argmax(p.predictions, -1), p.label_ids
    P = [[id2label[x] for x, y in zip(pr, la) if y != -100] for pr, la in zip(preds, labs)]
    L = [[id2label[y] for x, y in zip(pr, la) if y != -100] for pr, la in zip(preds, labs)]
    r = seqeval.compute(predictions=P, references=L)
    return {"precision": r["overall_precision"], "recall": r["overall_recall"], "f1": r["overall_f1"]}

args = TrainingArguments("ner-out", learning_rate=5e-5, per_device_train_batch_size=32, per_device_eval_batch_size=64,
                         num_train_epochs=3, weight_decay=0.01, eval_strategy="epoch", save_strategy="no",
                         logging_steps=50, report_to=[], seed=7)
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                  data_collator=DataCollatorForTokenClassification(tok), compute_metrics=metrics)
trainer.train()
test_metrics = trainer.evaluate(tokenized["test"]); print(test_metrics)""",
    """# Export in the exact layout the backend loads (backend/models/ner) + a model card
out = "/content/export/ner"
trainer.save_model(out); tok.save_pretrained(out)
json.dump({"base_model": MODEL, "labels": labels, "test": test_metrics,
           "data": "AeroNexus synthetic drawings (+MaintNet if provided)", "intended_use": "entity tagging in drawings/logbooks",
           "limitations": "synthetic layouts; verify on your own labelled drawings before relying on it"},
          open(f"{out}/model_card.json", "w"), indent=1)
!cd /content/export && zip -qr ner.zip ner && ls -lh ner.zip
from google.colab import files; files.download("/content/export/ner.zip")""",
    """# %% [markdown]
**Next (VS Code):** unzip into `backend/models/ner/`, then run with the trained model:

```bash
cd backend
AERONEXUS_USE_HF_NER=1 python -m aeronexus.evals.runner   # gate must still pass
AERONEXUS_USE_HF_NER=1 pytest -q
```""",
]

# ---------------------------------------------------------------------------------------------
VLM = [
    """# %% [markdown]
# 02 · LoRA fine-tune a vision-language model on engineering drawings

Fine-tunes **Qwen2.5-VL-3B-Instruct** with LoRA (PEFT) on **MechVQA** (Apache-2.0, 3.3K drawings / 21K QA pairs) so the
extraction fallback reads mechanical drawings better. MechVQA's authors report a domain-tuned model beating the strongest
closed baseline by 7.57 points (see docs/RESEARCH.md, CLM-DRAW-2).

**Runtime: GPU (L4 or A100).** Honest note: LoRA for vision-language models on TPU via PyTorch/XLA is not well supported
in Colab today; use a GPU runtime for this notebook (QLoRA 4-bit fits a 3B VLM on a 24 GB L4). Run notebook 01 on TPU.

Output: a LoRA adapter you serve with vLLM (`--enable-lora`), which exposes an OpenAI-compatible API the backend already
speaks (`AERONEXUS_LLM_PROVIDER=openai`).""",
    SETUP,
    """!pip -q install -U transformers peft trl datasets accelerate bitsandbytes qwen-vl-utils
import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU — switch runtime")""",
    """from datasets import load_dataset
raw = load_dataset("XiaofengAlg/MechVQA")
print(raw)                                  # inspect splits/columns — the mapping below adapts to common layouts
split_tr = "train" if "train" in raw else list(raw)[0]
cols = raw[split_tr].column_names; print(cols)""",
    """from PIL import Image
import io, json

def pick(ex, *names):
    for n in names:
        if n in ex and ex[n] is not None:
            return ex[n]
    raise KeyError(f"none of {names} in {list(ex)}")

def to_chat(ex):
    img = pick(ex, "image", "images", "img")
    if isinstance(img, list): img = img[0]
    if isinstance(img, dict) and "bytes" in img: img = Image.open(io.BytesIO(img["bytes"]))
    if isinstance(img, str): img = Image.open(img)
    q = pick(ex, "question", "query", "prompt")
    a = pick(ex, "answer", "response", "label", "gt_answer")
    if isinstance(a, list): a = a[0]
    return {"messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(q)}]},
        {"role": "assistant", "content": [{"type": "text", "text": str(a)}]}], "images": [img.convert("RGB")]}

train = raw[split_tr].shuffle(seed=0).select(range(min(6000, len(raw[split_tr])))).map(to_chat, remove_columns=cols)
evals = (raw["validation"] if "validation" in raw else raw[split_tr].select(range(200))).map(to_chat, remove_columns=cols)
print(train[0]["messages"])""",
    """from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig
BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(BASE, min_pixels=256*28*28, max_pixels=1024*28*28)  # cap tokens per drawing
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])""",
    """from trl import SFTConfig, SFTTrainer
cfg = SFTConfig(output_dir="vlm-lora", num_train_epochs=1, per_device_train_batch_size=2, gradient_accumulation_steps=8,
                learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=20,
                save_strategy="no", gradient_checkpointing=True, report_to=[], max_length=2048)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=train, eval_dataset=evals, peft_config=lora,
                     processing_class=processor)
trainer.train()""",
    """# Quick before/after check on held-out items: exact match after normalisation (+ judge later in the repo harness)
import re
def norm(s): return re.sub(r"[^a-z0-9.]", "", str(s).lower())
def answer(ex):
    msgs = ex["messages"][:1]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = processor(text=[text], images=ex["images"], return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=64, do_sample=False)
    return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
sample = evals.select(range(min(100, len(evals))))
em = sum(norm(answer(e)) == norm(e["messages"][1]["content"][0]["text"]) for e in sample) / len(sample)
print("held-out exact match:", em)""",
    """out = "/content/export/vlm-lora"
trainer.model.save_pretrained(out); processor.save_pretrained(out)
json.dump({"base_model": BASE, "dataset": "MechVQA (Apache-2.0)", "held_out_exact_match": em, "lora": lora.to_dict()},
          open(f"{out}/model_card.json", "w"), indent=1, default=str)
!cd /content/export && zip -qr vlm-lora.zip vlm-lora && ls -lh vlm-lora.zip
from google.colab import files; files.download("/content/export/vlm-lora.zip")""",
    """# %% [markdown]
**Serve it (GPU box or cloud VM) and point the backend at it:**

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --enable-lora --lora-modules drawings=./vlm-lora --max-model-len 8192
# backend
AERONEXUS_LLM_PROVIDER=openai OPENAI_BASE_URL=http://<host>:8000/v1 AERONEXUS_SMALL_MODEL=drawings \\
AERONEXUS_LARGE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct python -m aeronexus.evals.runner
```
The regression gate decides whether the fine-tune ships — not the training loss.""",
]

# ---------------------------------------------------------------------------------------------
DET = [
    """# %% [markdown]
# 03 · Train the layout detector (title block, BOM table, notes, views)

YOLO (Ultralytics) detector for drawing regions. Two data sources:
1. **Synthetic** — rendered from the repo's generator, labels come free from the layout (no annotation cost).
2. **Real** — public Roboflow Universe engineering-drawing sets (title block / dimension / tolerance classes; free
   account + API key). Mixing real scans fixes the synthetic-only domain gap.

**Runtime:** GPU (T4 is enough). Export `detector.pt` into `backend/models/` — the backend switches to it automatically.""",
    SETUP,
    "!pip -q install ultralytics pillow roboflow",
    """# Render synthetic sheets to PNG + YOLO labels from the heuristic layout (which is exact on synthetic sheets)
import json, os, random, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from aeronexus.extraction.document import Sheet, HeuristicLayoutDetector
CLASSES = ["title_block", "bom_table", "notes", "view"]
det = HeuristicLayoutDetector()
root = "/content/yolo";
for sp in ["train", "val"]:
    os.makedirs(f"{root}/images/{sp}", exist_ok=True); os.makedirs(f"{root}/labels/{sp}", exist_ok=True)
rows = [json.loads(l) for l in open("data/train/drawings.jsonl")][:1500]
for i, d in enumerate(rows):
    s = Sheet.from_dict(d); W, H = 1280, int(1280 * s.height / s.width); k = W / s.width
    img = Image.new("L", (W, H), 255); dr = ImageDraw.Draw(img)
    dr.rectangle([8*k, 8*k, (s.width-8)*k, (s.height-8)*k], outline=0, width=2)
    for t in s.tokens:
        dr.text((t.bbox[0]*k, t.bbox[1]*k), t.text, fill=random.randint(0, 60))
    dr.rectangle([110*k, 190*k, 450*k, 390*k], outline=0, width=2)  # part outline
    sp = "val" if i % 10 == 0 else "train"
    img.save(f"{root}/images/{sp}/{s.id}.png")
    with open(f"{root}/labels/{sp}/{s.id}.txt", "w") as f:
        for r in det(s):
            x, y, w, h = r.bbox; pad = 8
            cx, cy = (x - pad + (w + 2*pad)/2)/s.width, (y - pad + (h + 2*pad)/2)/s.height
            f.write(f"{CLASSES.index(r.label)} {cx:.5f} {cy:.5f} {(w+2*pad)/s.width:.5f} {(h+2*pad)/s.height:.5f}\\n")
open(f"{root}/data.yaml", "w").write(f"path: {root}\\ntrain: images/train\\nval: images/val\\nnames: {CLASSES}\\n")
print("rendered", len(rows))""",
    """# (optional) add a real scanned-drawing set from Roboflow Universe (needs a free API key)
import os
if os.environ.get("ROBOFLOW_API_KEY"):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    # e.g. workspace "trial-yolo", project "eng_drawing-idyau" — check the dataset's license on its page first
    real = rf.workspace("trial-yolo").project("eng_drawing-idyau").version(1).download("yolov8")
    print("downloaded to", real.location, "— merge its title_block class into data.yaml or fine-tune in a 2nd stage")""",
    """from ultralytics import YOLO
model = YOLO("yolo11n.pt")
res = model.train(data="/content/yolo/data.yaml", epochs=40, imgsz=1280, batch=16, seed=0, patience=10)
metrics = model.val(); print(metrics.box.map50, metrics.box.map)""",
    """import shutil
best = model.trainer.best; shutil.copy(best, "/content/detector.pt")
from google.colab import files; files.download("/content/detector.pt")""",
]

# ---------------------------------------------------------------------------------------------
PRICE = [
    """# %% [markdown]
# 04 · Price model with conformal intervals (CPU is fine; TPU optional via JAX/Keras)

Gradient boosting on log price + **split-conformal** calibration → a distribution-free 90% interval. Exports
`price_model.joblib` with the keys the backend expects (`model`, `q`, `metrics`). Swap the synthetic CSV for your
real quote history (same columns) to make it yours.""",
    SETUP,
    "!pip -q install lightgbm joblib",
    """import numpy as np, pandas as pd, math, json, joblib, sys
sys.path.insert(0, ".")
from aeronexus.modules.pricing import featurize, CLASSES, NUMERIC
df = pd.read_csv("data/train/historical_quotes.csv")
X = np.array([featurize(r) for r in df.to_dict("records")]); y = np.log(df["unit_price"].values)
rng = np.random.default_rng(7); idx = rng.permutation(len(df))
tr, cal, te = np.split(idx, [int(.6*len(df)), int(.8*len(df))])""",
    """import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
cands = {
  "lightgbm": lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.9, verbose=-1),
  "sk_gbr": GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.08, subsample=0.9, random_state=7),
}
scores = {}
for name, m in cands.items():
    m.fit(X[tr], y[tr]); p = m.predict(X[cal])
    scores[name] = float(np.mean(np.abs(np.exp(p) - np.exp(y[cal])) / np.exp(y[cal])))
best = min(scores, key=scores.get); model = cands[best]; print(scores, "→", best)""",
    """alpha = 0.10
res = np.abs(model.predict(X[cal]) - y[cal])
q = float(np.sort(res)[min(len(res)-1, math.ceil((len(cal)+1)*(1-alpha)) - 1)])
pt = model.predict(X[te])
metrics = {"model": best, "test_mape": float(np.mean(np.abs(np.exp(pt)-np.exp(y[te]))/np.exp(y[te]))),
           "test_coverage": float(np.mean(np.abs(pt - y[te]) <= q)), "target_coverage": 1 - alpha,
           "n_train": len(tr), "n_cal": len(cal), "n_test": len(te)}
print(json.dumps(metrics, indent=1))
assert metrics["test_coverage"] >= 0.85, "intervals under-cover: recalibrate before shipping"
joblib.dump({"model": model, "q": q, "metrics": metrics}, "/content/price_model.joblib")
from google.colab import files; files.download("/content/price_model.joblib")""",
]

# ---------------------------------------------------------------------------------------------
RUL = [
    """# %% [markdown]
# 05 · Deep remaining-useful-life model on NASA C-MAPSS (TPU, JAX/Keras)

The backend ships a gradient-boosted RUL model trained on CPU (`scripts/train_public_models.py`). This notebook trains a
**1D-CNN + GRU sequence model on Cloud TPU** with Keras 3 on the JAX backend — the path TPUs support best — and compares
it with that baseline on the official test sets. Same preprocessing (per-regime normalisation, RUL cap 125, windows of
30 cycles) via the repo's own code, so the comparison is fair.

**Runtime:** *TPU*. Works unchanged on GPU/CPU (slower). C-MAPSS is a NASA public-domain dataset (simulated engines).""",
    SETUP,
    """!pip -q install -U keras jax
import os; os.environ["KERAS_BACKEND"] = "jax"
import jax, keras, numpy as np, json, sys
sys.path.insert(0, ".")
print(jax.devices())
from aeronexus.pdm.cmapss import load, load_rul, add_target, Normalizer, CAP, rmse, phm_score""",
    """W = 30
def windows(z, sensors, with_y):
    X, y, units = [], [], []
    for u, g in z.groupby("unit"):
        a = g[sensors].to_numpy("float32")
        if len(a) < W:
            a = np.vstack([np.repeat(a[:1], W - len(a), 0), a])      # left-pad short histories
        if with_y:
            r = g["rul"].to_numpy("float32")
            r = np.r_[np.repeat(r[:1], max(0, W - len(r))), r]
            for i in range(W, len(a) + 1):
                X.append(a[i - W:i]); y.append(r[i - 1]); units.append(u)
        else:
            X.append(a[-W:]); units.append(u)
    return np.stack(X), (np.array(y) if with_y else None), np.array(units)

def model(n_feat):
    inp = keras.Input((W, n_feat))
    x = keras.layers.Conv1D(64, 5, padding="causal", activation="relu")(inp)
    x = keras.layers.Conv1D(64, 5, padding="causal", activation="relu")(x)
    x = keras.layers.GRU(64)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m""",
    """results = {}
for subset in ["FD001", "FD002", "FD003", "FD004"]:
    tr = add_target(load(subset, "train")); norm = Normalizer.fit(tr)
    z = norm.transform(tr); z["rul"] = tr["rul"].values
    X, y, units = windows(z, norm.sensors, True)
    rng = np.random.default_rng(0); val_units = set(rng.choice(np.unique(units), size=max(10, len(np.unique(units)) // 5), replace=False))
    vm = np.isin(units, list(val_units))
    m = model(X.shape[-1])
    m.fit(X[~vm], y[~vm], validation_data=(X[vm], y[vm]), epochs=40, batch_size=512, verbose=0,
          callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
    q = float(np.quantile(np.abs(m.predict(X[vm], verbose=0).ravel() - y[vm]), 0.9))   # conformal half-width
    te = load(subset, "test"); Xt, _, _ = windows(norm.transform(te), norm.sensors, False)
    pred = np.clip(m.predict(Xt, verbose=0).ravel(), 0, CAP); true = np.minimum(load_rul(subset), CAP)
    results[subset] = {"rmse": round(rmse(pred, true), 3), "phm": round(phm_score(pred, true), 1),
                       "coverage": round(float(np.mean(np.abs(pred - true) <= q)), 3)}
    m.save(f"/content/rul_{subset}.keras")
    print(subset, results[subset])""",
    """baseline = json.load(open("data/reports/pdm_cmapss.json")) if os.path.exists("data/reports/pdm_cmapss.json") else {}
for s, r in results.items():
    b = baseline.get(s, {}).get("metrics", {})
    print(f"{s}: deep RMSE {r['rmse']}  vs  gradient-boosted {b.get('test_rmse_capped', 'n/a')} (run scripts/train_public_models.py for the baseline)")
json.dump(results, open("/content/rul_deep_results.json", "w"), indent=1)""",
    """# %% [markdown]
**Ship it only if it wins the gate.** Copy `rul_deep_results.json` next to `data/reports/` and compare in the evaluation
page; if the deep model beats the baseline on every subset *and* keeps interval coverage ≥ 0.8, add a Keras loader in
`aeronexus/modules/pdm.py` (`keras.saving.load_model`) — the rest of the workflow (rule, case, approval) is unchanged.""",
]

SAFETY = [
    """# %% [markdown]
# 06 · Transformer safety-text model (NTSB, optional ASRS) — TPU or GPU

Fine-tunes DistilRoBERTa for the same task as the shipped TF-IDF model — fatal-outcome triage from probable-cause text with
outcome words masked — on the same **temporal split** (train 2008–2015, validate 2016, test 2017–2022), so the numbers are
directly comparable. Optional: if you export ASRS reports (CSV, ≤10,000 per download) into `data/public/asrs/`, the last
cell trains an anomaly-category classifier on them. ASRS narratives are self-reported and **not verified by NASA**.""",
    SETUP,
    """import importlib, subprocess, sys
try:
    import jax; tpu = any(d.platform == "tpu" for d in jax.devices())
except Exception:
    tpu = False
if tpu:
    import torch; v = torch.__version__.split("+")[0]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"torch_xla[tpu]=={v}", "-f",
                    "https://storage.googleapis.com/libtpu-releases/index.html"], check=False)
!pip -q install transformers datasets accelerate evaluate
sys.path.insert(0, ".")
from aeronexus.safety.ntsb import load_events, split
df = load_events(); tr, va, te = split(df); print(len(tr), len(va), len(te), "TPU:", tpu)""",
    """import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, average_precision_score
M = "distilroberta-base"; tok = AutoTokenizer.from_pretrained(M)
enc = lambda d: Dataset.from_dict({"text": d.text_masked.tolist(), "label": d.fatal.tolist()}).map(
    lambda b: tok(b["text"], truncation=True, max_length=128, padding="max_length"), batched=True)
dtr, dva, dte = enc(tr), enc(va), enc(te)
pos_w = (1 - tr.fatal.mean()) / tr.fatal.mean()
model = AutoModelForSequenceClassification.from_pretrained(M, num_labels=2)
args = TrainingArguments("safety-out", per_device_train_batch_size=32, per_device_eval_batch_size=64, num_train_epochs=2,
                         learning_rate=3e-5, eval_strategy="epoch", save_strategy="no", report_to=[], seed=7)
trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva)
trainer.train()
logits = trainer.predict(dte).predictions; p = np.exp(logits)[:, 1] / np.exp(logits).sum(1)
print("test AUROC", roc_auc_score(te.fatal, p), "PR-AUC", average_precision_score(te.fatal, p))
print("compare with the shipped TF-IDF model in data/reports/safety_ntsb.json")""",
    """# Optional: ASRS anomaly categories from your own export (column names vary by export; adjust if needed)
import glob, pandas as pd
files = glob.glob("data/public/asrs/*.csv")
if files:
    a = pd.concat(pd.read_csv(f, low_memory=False, skiprows=[1]) for f in files)
    text = next(c for c in a.columns if "Narrative" in c); lab = next(c for c in a.columns if "Anomaly" in c)
    a = a[[text, lab]].dropna(); a["y"] = a[lab].str.split(";").str[0].str.split(" ").str[0]
    print(a["y"].value_counts().head(10))
    # → reuse the Trainer above with num_labels = number of top categories
else:
    print("No ASRS export found — see https://asrs.arc.nasa.gov/search/database.html")""",
]

TRAJ = [
    """# %% [markdown]
# 07 · Flight-trajectory analytics (OpenSky data — research use only)

Detects unusual approaches (energy/descent-rate anomalies, go-arounds) from ADS-B trajectories with the open-source
`traffic` library. **License:** OpenSky Network data may be used for non-profit research/education, government purposes or
*internal* commercial testing only, may not be redistributed, and must be cited. This notebook therefore never writes
trajectory data into the repo — only aggregate metrics.""",
    "!pip -q install traffic scikit-learn",
    """from traffic.data.samples import quickstart    # small sample bundled with the library for tutorials
import numpy as np
from sklearn.ensemble import IsolationForest
t = quickstart
feats, ids = [], []
for f in t:
    f = f.resample("5s")
    d = f.data
    if len(d) < 30 or "vertical_rate" not in d:
        continue
    last = d.tail(60)       # final ~5 minutes
    feats.append([last.vertical_rate.mean(), last.vertical_rate.std(), last.groundspeed.mean(), last.groundspeed.std(),
                  (last.altitude.diff() > 0).mean()])          # share of climbing samples ≈ go-around signal
    ids.append(f.flight_id or f.callsign)
X = np.nan_to_num(np.array(feats)); iso = IsolationForest(random_state=0).fit(X)
score = -iso.score_samples(X)
for i in np.argsort(-score)[:5]:
    print(ids[i], round(float(score[i]), 3))""",
    """# %% [markdown]
For real analysis request historical data through OpenSky's research access (Trino) under their terms, keep it inside
your institution, and cite: Schäfer et al., "Bringing up OpenSky: A large-scale ADS-B sensor network for research", IPSN 2014.""",
]


def main() -> None:
    for name, cells, acc in [("01_ner_tpu", NER, "TPU"), ("02_vlm_lora", VLM, "GPU"), ("03_detector", DET, "GPU"),
                             ("04_price_model", PRICE, None), ("05_cmapss_rul_tpu", RUL, "TPU"),
                             ("06_safety_text", SAFETY, "TPU"), ("07_opensky_trajectories", TRAJ, None)]:
        (HERE / f"{name}.ipynb").write_text(json.dumps(nb(cells, acc), indent=1))
        print("wrote", name)


if __name__ == "__main__":
    main()
