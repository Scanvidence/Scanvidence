"""Stage-1 radiomics cache (GT masks primary; --mask-dir for predicted-mask sensitivity).

Extracts three stable blocks (FLAIR+WT, T1c+TC, T1c+ET) over the 577-case
dev cohort (train/val/calibration only — locked test never touched).
Uses SimpleITK native geometry and PyRadiomics; no wavelets, tight
minimumROISize, fixed binWidth for stability at low n.

Run (128 GB box, parallel):
    python scripts/extract_radiomics.py --data-root "...TrainingData" --n-jobs -2
    python scripts/extract_radiomics.py --data-root "..." --mask-dir cache/pred_masks --out cache
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed

CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm")  # no wavelets at n~577
SETTINGS: dict[str, object] = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "nearestNeighbour",
    "minimumROISize": 10,
    "enableCExtensions": True,
}
BLOCKS: tuple[tuple[str, str, object], ...] = (
    ("flair_wt", "t2f", lambda L: L >= 1),
    ("t1c_tc", "t1c", lambda L: (L == 1) | (L == 3)),
    ("t1c_et", "t1c", lambda L: L == 3),
)


def discover(root: str | Path) -> dict[str, dict[str, Path]]:
    keys = ("seg", "t1n", "t1c", "t2w", "t2f")
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in Path(root).rglob("*"):
        if not f.is_file():
            continue
        low = f.name.lower()
        for k in keys:
            for sep in ("-", "_"):
                for ext in (".nii.gz", ".nii"):
                    suf = f"{sep}{k}{ext}"
                    if low.endswith(suf):
                        groups[f.name[: -len(suf)]][k] = f
    return {c: fm for c, fm in groups.items() if set(keys) <= set(fm)}


def _mock_feature_names() -> list[str]:
    # 3 blocks × ~88 stable original_* names (firstorder+shape+glcm+glrlm+glszm)
    # Shape is block-independent; keep realistic names for schema freeze QA.
    names: list[str] = []
    for block, _, _ in BLOCKS:
        for c in CLASSES:
            for i in range(18 if c == "firstorder" else 16 if c == "shape" else 22):
                names.append(f"{block}__original_{c}_{c[:2]}{i:02d}")
                if len([n for n in names if n.startswith(block)]) >= 88:
                    break
            if len([n for n in names if n.startswith(block)]) >= 88:
                # trim to exactly 88 per block
                pass
        # enforce 88 per block
        block_names = [n for n in names if n.startswith(block)]
        # keep first 88 for this block, drop overflow from the loop above
        if len(block_names) > 88:
            names = [n for n in names if not n.startswith(block)] + block_names[:88]
        elif len(block_names) < 88:
            for i in range(len(block_names), 88):
                names.append(f"{block}__original_{CLASSES[i % len(CLASSES)]}_pad{i:02d}")
    # ensure exactly 264
    return sorted(set(names))[:264]


def extract_one(cid: str, fm: dict[str, Path], seg_path: Path) -> dict[str, object]:
    try:
        import radiomics  # noqa: F401
        from radiomics import featureextractor
    except ImportError as exc:
        raise SystemExit(
            "pyradiomics not installed — `pip install pyradiomics` or "
            "`pip install -e .[radiomics]` (needs conda-forge / Py 3.9-3.11)"
        ) from exc
    seg = sitk.ReadImage(str(seg_path))  # native geometry, real spacing
    arr = sitk.GetArrayFromImage(seg).astype(np.int16)
    vox_cm3 = float(np.prod(seg.GetSpacing())) / 1000.0
    out: dict[str, object] = {
        "case": cid,
        "wt_volume_cm3": round(float((arr >= 1).sum()) * vox_cm3, 2),
    }
    log: dict[str, str] = {}
    for block, mod, make in BLOCKS:
        mask = make(arr).astype(np.uint8)  # type: ignore[operator]
        if int(mask.sum()) < int(SETTINGS["minimumROISize"]):  # type: ignore[arg-type]
            log[block] = "roi-too-small"
            continue
        img = sitk.ReadImage(str(fm[mod]))
        mimg = sitk.GetImageFromArray(mask)
        mimg.CopyInformation(seg)
        ext = featureextractor.RadiomicsFeatureExtractor()
        ext.settings.update(SETTINGS)
        ext.disableAllFeatures()
        for c in CLASSES:
            ext.enableFeatureClassByName(c)
        try:
            res = ext.execute(img, mimg)
        except Exception as exc:  # noqa: BLE001
            log[block] = f"error:{type(exc).__name__}"
            continue
        for k, v in res.items():
            if str(k).startswith("original_"):
                out[f"{block}__{k}"] = float(v)
        log[block] = "ok"
    out["_log"] = json.dumps(log)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache radiomics features for dev cohort.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="data/manifests/split_v1.json")
    ap.add_argument("--mgmt", default="data/manifests/mgmt_labels_v1.csv")
    ap.add_argument(
        "--mask-dir",
        default=None,
        help="predicted-mask NIfTIs named {case}.nii.gz (sensitivity run)",
    )
    ap.add_argument("--cohort", choices=["mgmt", "all"], default="mgmt")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--n-jobs", type=int, default=-2)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="mock extraction (no pyradiomics) — synthesizes feature columns for pipeline QA",
    )
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text())
    dev: dict[str, str] = {c: p for p in ("train", "val", "calibration") for c in split[p]}
    groups = discover(args.data_root)

    mgmt_path = Path(args.mgmt)
    if args.cohort == "mgmt":
        if not mgmt_path.exists():
            raise SystemExit(
                f"MGMT manifest not found: {mgmt_path} — create it first (case_id,mgmt)."
            )
        mgmt = pd.read_csv(mgmt_path, dtype=str)
        if "case_id" not in mgmt.columns or "mgmt" not in mgmt.columns:
            raise SystemExit("mgmt csv must have columns case_id,mgmt")
        mgmt["mgmt"] = mgmt["mgmt"].astype(int)
        lab = dict(zip(mgmt.case_id, mgmt.mgmt, strict=True))
        cohort = [c for c in dev if c in lab and c in groups]
    else:
        lab = {}  # type: ignore[assignment]
        cohort = [c for c in dev if c in groups]

    print(f"cohort: {len(cohort)} cases (dev partitions only)")

    if args.dry_run:
        rng = np.random.default_rng(17)
        feat_names = _mock_feature_names()
        rows: list[dict[str, object]] = []
        for cid in cohort:
            seg = sitk.ReadImage(str(groups[cid]["seg"]))
            arr = sitk.GetArrayFromImage(seg).astype(np.int16)
            vox_cm3 = float(np.prod(seg.GetSpacing())) / 1000.0
            row: dict[str, object] = {
                "case": cid,
                "wt_volume_cm3": round(float((arr >= 1).sum()) * vox_cm3, 2),
            }
            for fn in feat_names:
                row[fn] = float(rng.normal(0, 1))
            # inject 3% NaN and a constant column to exercise QA drop-list
            for fn in rng.choice(feat_names, size=8, replace=False):
                if rng.random() < 0.5:
                    row[fn] = float("nan")
            row["_log"] = json.dumps({b[0]: "ok (dry-run)" for b in BLOCKS})
            rows.append(row)
        print(f"dry-run: synthesized {len(feat_names)} feature columns")
    else:
        mask_dir = Path(args.mask_dir) if args.mask_dir else None
        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(extract_one)(
                c, groups[c], (mask_dir / f"{c}.nii.gz") if mask_dir else groups[c]["seg"]
            )
            for c in cohort
        )

    df = pd.DataFrame(rows)
    df.insert(1, "partition", [dev[c] for c in df["case"]])
    df.insert(2, "mgmt", [lab.get(c, np.nan) for c in df["case"]])
    feat = sorted(c for c in df.columns if "__original_" in c)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "pred" if args.mask_dir else "gt"
    try:
        import radiomics as _rad

        rad_ver = _rad.__version__
    except Exception:
        rad_ver = "unknown"
    df.to_csv(out / f"radiomics_{tag}_dev.csv", index=False)
    meta = {
        "versions": {
            "radiomics": rad_ver,
            "simpleitk": sitk.__version__,
            "numpy": np.__version__,
        },
        "settings": SETTINGS,
        "classes": list(CLASSES),
        "blocks": [b[0] for b in BLOCKS],
        "n_cases": len(df),
        "n_features": len(feat),
        "cohort": args.cohort,
        "mask_source": str(args.mask_dir) if args.mask_dir else "ground_truth",
    }
    (out / f"radiomics_{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / f'radiomics_{tag}_dev.csv'}: {len(df)} cases x {len(feat)} features")


if __name__ == "__main__":
    main()
