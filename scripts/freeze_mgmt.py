"""Freeze MGMT labels — single source of truth, document-before-use.

Takes an open crosswalk (e.g. open-radiomics 2023↔MGMT) and writes
data/manifests/mgmt_labels_v1.csv (case_id,mgmt) plus .sha256.
Validates 0/1 labels and prints per-partition balance from the frozen
split so you can paste it into the notebook/data card.

    python scripts/freeze_mgmt.py --source path/to/crosswalk.csv
    python scripts/freeze_mgmt.py --source path/to/crosswalk.csv \
        --url https://example.com/resource --license "CC BY 4.0"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze MGMT manifest from a crosswalk.")
    ap.add_argument("--source", required=True, help="path to source CSV with case_id,mgmt")
    ap.add_argument("--out", default="data/manifests/mgmt_labels_v1.csv")
    ap.add_argument("--split", default="data/manifests/split_v1.json")
    ap.add_argument("--url", default="", help="source URL to record in data card")
    ap.add_argument("--license", default="", help="source license/terms")
    ap.add_argument("--notes", default="", help="free-form provenance note")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    df = pd.read_csv(src, dtype=str)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    if "case_id" not in lower or "mgmt" not in lower:
        raise SystemExit(f"source must have columns case_id,mgmt — got {list(df.columns)}")
    df = df.rename(columns={lower["case_id"]: "case_id", lower["mgmt"]: "mgmt"})
    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["mgmt"] = pd.to_numeric(df["mgmt"], errors="coerce")
    if df["mgmt"].isna().any():
        bad = df[df["mgmt"].isna()].head()
        raise SystemExit(f"non-numeric mgmt values:\n{bad}")
    if not set(df["mgmt"].unique()) <= {0, 1}:
        raise SystemExit(f"mgmt must be 0/1, got {sorted(df['mgmt'].unique())}")
    df["mgmt"] = df["mgmt"].astype(int)
    df = df.drop_duplicates("case_id").sort_values("case_id")
    if df["case_id"].duplicated().any():
        raise SystemExit("duplicate case_id after dedup")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["case_id", "mgmt"]].to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    sha_path = out.with_suffix(".sha256")
    sha_path.write_text(sha + "\n")
    # keep .csv.sha256 for backwards compat if out is .csv
    if out.suffix == ".csv":
        Path(str(out) + ".sha256").write_text(sha + "\n")
    print(f"wrote {out} ({len(df)} rows)  sha256:{sha[:12]}...")

    # per-partition audit against frozen split
    split = json.loads(Path(args.split).read_text())
    sha_f = Path(args.split).with_suffix(".sha256")
    sha_short = sha_f.read_text().strip()[:12] + "..." if sha_f.exists() else "no sha"
    print(f"split {args.split} — hash {sha_short}")
    mgmt_set = set(df.case_id)
    total = 0
    for part in ("train", "val", "calibration", "test"):
        ids = set(split.get(part, [])) & mgmt_set
        sub = df[df.case_id.isin(ids)]
        total += len(sub)
        bal = round(float(sub.mgmt.mean()), 3) if len(sub) else float("nan")
        n_split = len(split.get(part, []))
        print(f"  {part:12s} {len(sub):4d}  mgmt+ {bal:.3f}  ({len(sub)}/{n_split} of split)")
    dev_with = total - len(set(split.get("test", [])) & mgmt_set)
    print(f"  dev (train+val+cal) {dev_with} cases with MGMT")
    print(f"  overall {len(df)} cases, mgmt+ {df.mgmt.mean():.3f} ({int(df.mgmt.sum())}/{len(df)})")
    if args.url or args.license:
        print(f"source: {args.url or '—'}  license: {args.license or '—'}")
        print("→ paste the URL/license into docs/data_card.md and cite the resource")
    if args.notes:
        print(f"notes: {args.notes}")


if __name__ == "__main__":
    main()
