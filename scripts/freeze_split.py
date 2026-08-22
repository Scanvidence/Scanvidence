"""Freeze the official patient-level 70/10/5/15 split for the B0 track.

Discovers every case under ``--data-root`` that carries the four BraTS
sequences plus a label map, shuffles the case IDs with a fixed seed, and
writes ``data/manifests/split_v1.json`` next to a SHA-256 of the blob.
The manifest is the single record of which cases sit in train,
validation, calibration, and test; the test partition is only ever
touched at the very end of the study.

The fractions follow the work plan (G0-G3): 0.70 train, 0.10 validation,
0.05 calibration, 0.15 test. Incomplete cases are dropped with no
downstream effect because nothing re-splits after this script runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

_KEYS = ("seg", "t1n", "t1c", "t2w", "t2f")
_OUT_DIR = Path("data/manifests")


def discover(root: Path) -> dict[str, dict[str, Path]]:
    """Map every case ID to its five NIfTI files.

    A case is only admitted when all four modalities and the label map
    exist for it. Suffix matching handles both ``-`` and ``_`` separators
    and both ``.nii.gz`` and ``.nii`` encodings, so the manifest is
    independent of how a mirror unpacked the dataset.
    """
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        low = f.name.lower()
        for key in _KEYS:
            for sep in ("-", "_"):
                for ext in (".nii.gz", ".nii"):
                    suffix = f"{sep}{key}{ext}"
                    if low.endswith(suffix):
                        groups[f.name[: -len(suffix)]][key] = f
    return {case: files for case, files in groups.items() if set(_KEYS) <= set(files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    ids = sorted(discover(Path(args.data_root)))
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test, n_cal, n_val = round(0.15 * n), round(0.05 * n), round(0.10 * n)

    split = {
        "seed": args.seed,
        "fractions": [0.7, 0.1, 0.05, 0.15],
        "test": ids[:n_test],
        "calibration": ids[n_test : n_test + n_cal],
        "val": ids[n_test + n_cal : n_test + n_cal + n_val],
        "train": ids[n_test + n_cal + n_val :],
    }

    partitions = (split["train"], split["val"], split["calibration"], split["test"])
    if sum(len(p) for p in partitions) != n:
        raise RuntimeError("partitioned case count does not match the discovered cohort")
    for i, left in enumerate(partitions):
        for right in partitions[i + 1 :]:
            overlap = set(left) & set(right)
            if overlap:
                raise RuntimeError(
                    f"patient-level leakage across partitions: {sorted(overlap)[:5]}"
                )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(split, sort_keys=True)
    (_OUT_DIR / "split_v1.json").write_text(blob)
    (_OUT_DIR / "split_v1.sha256").write_text(hashlib.sha256(blob.encode()).hexdigest())

    print({k: len(split[k]) for k in ("train", "val", "calibration", "test")})
    print("hash:", (_OUT_DIR / "split_v1.sha256").read_text())


if __name__ == "__main__":
    main()
