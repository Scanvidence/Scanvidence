"""Reference B0 numbers on the frozen validation partition.

Loads the official B0 checkpoint and reports whole-volume ET/TC/WT Dice
over every case in ``data/manifests/split_v1.json``'s validation list.
Inference is a sliding 96-cubed window across the full volume at a fixed
stride, so the numbers do not depend on any training-time patch
sampling.

Artifacts land in ``runs/eval-{model}-frozen-val/``: ``per_case.csv``
with one row per case and ``summary.json`` with regional mean/std plus
the manifest hash these numbers belong to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scanvidence.models.backbone import SegResNetB0

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def norm(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization over the non-zero foreground."""
    v = np.asarray(v, dtype=np.float32)
    m = v > 0
    out = np.zeros_like(v)
    if m.any():
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-8)
    return out


def sliding_window(
    model: SegResNetB0,
    vol: np.ndarray,
    patch: int = 96,
    stride: int = 48,
    device: str = _DEVICE,
) -> np.ndarray:
    """Segment a full volume by averaging overlapping patch predictions.

    Every voxel receives one prediction per window that covers it; the
    final logits are the per-voxel mean, cropped back to the original
    extent. Weighted means of the same patches would be a different
    number, so the averaging rule is fixed here and nowhere else.
    """
    orig = vol.shape[1:]
    pad = tuple((patch - s % patch) % patch for s in orig)
    if any(pad):
        vol = np.pad(vol, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])))
    _, D, H, W = vol.shape

    def starts(n: int) -> list[int]:
        s = list(range(0, n - patch + 1, stride))
        if s[-1] != n - patch:
            s.append(n - patch)
        return s

    acc = np.zeros((4, D, H, W), np.float64)
    cnt = np.zeros((D, H, W), np.float64)
    with torch.no_grad():
        for i in starts(D):
            for j in starts(H):
                for k in starts(W):
                    x = torch.from_numpy(vol[:, i : i + patch, j : j + patch, k : k + patch])
                    o = model(x[None].to(device))
                    if isinstance(o, (tuple, list)):
                        o = o[0]
                    acc[:, i : i + patch, j : j + patch, k : k + patch] += o[0].cpu().numpy()
                    cnt[i : i + patch, j : j + patch, k : k + patch] += 1
    return (acc / cnt[None])[(slice(4),) + tuple(slice(0, o) for o in orig)]


def regions(L: np.ndarray) -> dict[str, np.ndarray]:
    """BraTS regional masks from the label map (ET/TC/WT)."""
    return {"ET": L == 3, "TC": (L == 1) | (L == 3), "WT": (L >= 1) & (L <= 3)}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sorensen-Dice between two binary masks (1.0 when both empty)."""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ckpt", default="runs/full-b0-seed17/best.pt")
    parser.add_argument(
        "--model",
        default="b0",
        choices=["b0", "b1"],
        help="which architecture to build (b0=SegResNetB0, b1=B1Segmentor)",
    )
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    split = json.loads(Path("data/manifests/split_v1.json").read_text())
    if args.model == "b0":
        model = SegResNetB0(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0)
    else:
        from scanvidence.models.cnn_shared import B1Segmentor

        model = B1Segmentor(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(
        state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    )
    model.eval().to(_DEVICE)

    rows = []
    val_cases = split["val"][: args.max_cases or len(split["val"])]
    for cid in val_cases:
        case_dir = next(Path(args.data_root).rglob(f"{cid}-t1n.nii*")).parent
        vol = np.stack(
            [norm(nib.load(str(case_dir / f"{cid}-{m}.nii.gz")).get_fdata()) for m in _MODALITIES]
        ).astype(np.float32)
        seg = nib.load(str(case_dir / f"{cid}-seg.nii.gz")).get_fdata().astype(np.int16)
        pred = (
            sliding_window(model, vol, stride=args.stride, device=_DEVICE)
            .argmax(0)
            .astype(np.int16)
        )
        gr, pr = regions(seg), regions(pred)
        row = {
            "case": cid,
            **{f"dice_{r}": round(dice(pr[r], gr[r]), 4) for r in ("ET", "TC", "WT")},
        }
        rows.append(row)
        print(row)

    out = Path(f"runs/eval-{args.model}-frozen-val")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        r: {
            "mean": round(float(np.mean([x[f"dice_{r}"] for x in rows])), 4),
            "std": round(float(np.std([x[f"dice_{r}"] for x in rows])), 4),
        }
        for r in ("ET", "TC", "WT")
    }
    summary["n_cases"] = len(rows)
    summary["split_hash"] = Path("data/manifests/split_v1.sha256").read_text()
    (out / "per_case.csv").write_text(
        "case,dice_ET,dice_TC,dice_WT\n"
        + "".join(f"{x['case']},{x['dice_ET']},{x['dice_TC']},{x['dice_WT']}\n" for x in rows)
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
