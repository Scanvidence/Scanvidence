"""Per-step training cost of B0 on the actual workstation.

Times one 96-cubed forward/backward with the same optimizer and AMP
setup the full run uses, after a short warmup, and records the peak CUDA
allocation. Both workstations run this script in AMP and FP32; the JSON
in ``runs/`` is the feasibility artifact: a step that fits with headroom
at the batch-1 design point is a prerequisite for the full run, and the
measured seconds feed the runtime estimate.

A 96-cubed patch is cropped around the first foreground voxel of a
discovered case so the profiled step looks like a real training step.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scanvidence.models.backbone import SegResNetB0

_PATCH = 96


def norm(v: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization over the non-zero foreground."""
    v = np.asarray(v, dtype=np.float32)
    m = v > 0
    out = np.zeros_like(v)
    if m.any():
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("profile_b0 times CUDA training steps; no GPU found on this host")

    amp = not args.no_amp
    dev = "cuda"

    seg_file = next(Path(args.data_root).rglob("*-seg.nii.gz"))
    base = seg_file.parent
    cid = seg_file.name.split("-seg")[0]
    vol = np.stack(
        [
            norm(nib.load(str(base / f"{cid}-{m}.nii.gz")).get_fdata())
            for m in ("t1n", "t1c", "t2w", "t2f")
        ]
    ).astype(np.float32)
    seg = nib.load(str(seg_file)).get_fdata().astype(np.int16)
    c = np.argwhere(seg > 0)[0]
    st = [max(0, min(int(c[a]) - _PATCH // 2, vol.shape[1 + a] - _PATCH)) for a in range(3)]
    x = torch.from_numpy(vol[(slice(None), *[slice(s, s + _PATCH) for s in st])])[None].to(dev)
    y = torch.from_numpy(seg[tuple(slice(s, s + _PATCH) for s in st)])[None].long().to(dev)

    model = SegResNetB0(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)

    def train_step() -> None:
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=amp):
            o = model(x)
            if isinstance(o, (tuple, list)):
                o = o[0]
            p = o.softmax(1)
            fg = p[:, 1:]
            gt = (y[:, None] == torch.arange(4, device=dev).view(1, 4, 1, 1, 1)).float()[:, 1:]
            dice_l = 1 - (2 * (fg * gt).sum() / (fg.sum() + gt.sum() + 1e-5)).mean()
            loss = (dice_l + 0.5 * torch.nn.functional.cross_entropy(o, y)) / 2
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.measured):
        train_step()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) / args.measured * 1000
    alloc = torch.cuda.max_memory_allocated() / 2**30
    resv = torch.cuda.max_memory_reserved() / 2**30
    visible = torch.cuda.get_device_properties(0).total_memory / 2**30

    rep = {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "visible_gb": round(visible, 2),
        "precision": "amp" if amp else "fp32",
        "allocated_gb": round(alloc, 2),
        "reserved_gb": round(resv, 2),
        "step_ms": round(step_ms, 1),
        "pass_90pct": resv <= 0.9 * visible,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    out = Path("runs")
    out.mkdir(exist_ok=True)
    (out / f"profile-b0-{rep['host']}-{'amp' if amp else 'fp32'}.json").write_text(
        json.dumps(rep, indent=2)
    )
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
