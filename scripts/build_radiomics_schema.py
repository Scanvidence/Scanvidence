"""QA + schema freeze — run after cache/radiomics_gt_dev.csv lands.

Drop-list computed on train only (never val/cal). Freezes kept feature ids
for every downstream model.

    python scripts/build_radiomics_schema.py \
        --in cache/radiomics_gt_dev.csv --out cache/radiomics_schema_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build radiomics QA schema from train partition.")
    ap.add_argument("--in", dest="inp", default="cache/radiomics_gt_dev.csv")
    ap.add_argument("--out", dest="out", default="cache/radiomics_schema_v1.json")
    ap.add_argument("--na-thr", type=float, default=0.05)
    ap.add_argument("--abs-max", type=float, default=1e8)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    feat = [c for c in df.columns if "__original_" in c]
    tr = df[df.partition == "train"]
    if tr.empty:
        raise SystemExit("no train rows in radiomics csv — check split/cohort")

    drop: list[str] = []
    for c in feat:
        s = tr[c]
        if (
            s.isna().mean() > args.na_thr
            or float(s.std()) < 1e-12
            or float(s.abs().max()) > args.abs_max
        ):
            drop.append(c)

    keep = [c for c in feat if c not in drop]
    schema = {
        "schema_version": 1,
        "feature_ids": keep,
        "dropped": drop,
        "n_kept": len(keep),
        "n_raw": len(feat),
        "na_thr": args.na_thr,
        "abs_max": args.abs_max,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(schema, indent=2))
    print(f"kept {len(keep)} of {len(feat)} raw features (expect ~250: 3 blocks x ~88)")
    if drop:
        print("dropped:", ", ".join(drop[:10]) + (" ..." if len(drop) > 10 else ""))


if __name__ == "__main__":
    main()
