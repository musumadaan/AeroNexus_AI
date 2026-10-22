"""Download the auto-fetchable public datasets and verify their SHA-256 checksums.

Usage: python scripts/fetch_public_data.py [--only cmapss,ntsb]
Files land in backend/data/public/<key>/ (git-ignored). A checksum mismatch aborts that dataset.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aeronexus.public_data import REGISTRY, sha256  # noqa: E402


def fetch(key: str) -> bool:
    ds = REGISTRY[key]
    if not ds.auto_fetch:
        print(f"[{key}] manual download — see {ds.source_of_record}")
        return True
    ds.folder.mkdir(parents=True, exist_ok=True)
    ok = True
    for fname, digest in ds.files.items():
        dest = ds.folder / fname
        if dest.exists() and sha256(dest) == digest:
            continue
        print(f"[{key}] downloading {fname}")
        urllib.request.urlretrieve(ds.mirror + fname, dest)
        if sha256(dest) != digest:
            print(f"[{key}] CHECKSUM MISMATCH for {fname} — refusing to use it")
            dest.unlink()
            ok = False
    print(f"[{key}] {'ok' if ok else 'FAILED'}  ({ds.license})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    keys = [k for k in (ap.parse_args().only.split(",") if ap.parse_args().only else REGISTRY)]
    return 0 if all(fetch(k) for k in keys) else 1


if __name__ == "__main__":
    sys.exit(main())
