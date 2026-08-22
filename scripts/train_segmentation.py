"""Unified B0/B1 segmentation training on the frozen split.

One harness for both baselines so the only difference between a B0
retrain (``--model b0``) and B1 (``--model b1``) is the architecture.
Same seed, same patient-level split, same patch sampler (70% tumor-
centered), same loss (Dice + 0.5*CE), same optimizer/scheduler/budget,
and a production-step profile (allocated + reserved) written before
training starts. The checkpoint contains ``split_hash`` so any frozen-
val evaluation can prove which split it was trained on.

Run order (from task spec):

    python scripts/train_segmentation.py --model b0 \\
        --data-root \".../TrainingData\" --split data/manifests/split_v1.json \\
        --epochs 150 --seed 17 --out-dir runs/full-b0-seed17-frozen

    python scripts/train_segmentation.py --model b1 \\
        --data-root \".../TrainingData\" --split data/manifests/split_v1.json \\
        --epochs 150 --seed 17 --out-dir runs/b1-seed17
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

MODS = ("t1n", "t1c", "t2w", "t2f")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (checkpoint rng stores legacy numpy state)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discover(root: str | Path) -> dict[str, dict[str, Path]]:
    """Group every ``<case>-<mod>.nii[.gz]`` under its case ID."""
    root = Path(root)
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        low = f.name.lower()
        for key in (*MODS, "seg"):
            for sep in ("-", "_"):
                for ext in (".nii.gz", ".nii"):
                    suf = f"{sep}{key}{ext}"
                    if low.endswith(suf):
                        groups[f.name[: -len(suf)]][key] = f
    return groups


def norm(v: np.ndarray) -> np.ndarray:
    """Z-score on the non-zero foreground (matches training contract)."""
    v = np.asarray(v, dtype=np.float32)
    m = v > 0
    out = np.zeros_like(v)
    if m.any():
        out[m] = (v[m] - v[m].mean()) / (v[m].std() + 1e-8)
    return out


def load_case(groups: dict[str, dict[str, Path]], cid: str) -> tuple[np.ndarray, np.ndarray]:
    fm = groups[cid]
    vol = np.stack([norm(nib.load(str(fm[m])).get_fdata()) for m in MODS]).astype(np.float32)
    seg = nib.load(str(fm["seg"])).get_fdata().astype(np.int64)
    return vol, seg


def crop(
    vol: np.ndarray, seg: np.ndarray, center: np.ndarray | tuple | list, p: int = 96
) -> tuple[np.ndarray, np.ndarray]:
    pad = [(0, max(0, p - s)) for s in vol.shape[1:]]
    if any(q for _, q in pad):
        vol = np.pad(vol, ((0, 0), *pad))
        seg = np.pad(seg, pad)
    st = [max(0, min(int(c) - p // 2, vol.shape[1 + a] - p)) for a, c in enumerate(center)]
    sl = (slice(None), *[slice(s, s + p) for s in st])
    return vol[sl], seg[sl]


def regions(L: np.ndarray) -> dict[str, np.ndarray]:
    return {"ET": L == 3, "TC": (L == 1) | (L == 3), "WT": (L >= 1) & (L <= 3)}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    if not a.any() or not b.any():
        return 0.0
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ce = torch.nn.functional.cross_entropy(logits, target)
    p = logits.softmax(1)
    d = 0.0
    for c in (1, 2, 3):
        pc = p[:, c]
        gc = (target == c).float()
        d = d + (2 * (pc * gc).sum() + 1e-5) / (pc.sum() + gc.sum() + 1e-5)
    return (1 - d / 3) + 0.5 * ce


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified B0/B1 training on frozen split.")
    ap.add_argument("--model", choices=["b0", "b1"], required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="data/manifests/split_v1.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--val-every", type=int, default=1)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--profile-steps", type=int, default=10)
    args = ap.parse_args()
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    split = json.loads(Path(args.split).read_text())
    sha_path = Path(str(args.split) + ".sha256")
    split_hash = sha_path.read_text().strip() if sha_path.exists() else ""
    groups = discover(args.data_root)
    train_ids, val_ids = split["train"], split["val"]
    missing = [c for c in train_ids + val_ids if c not in groups]
    if missing:
        raise SystemExit(f"split IDs missing from data root (first 5): {missing[:5]}")

    if args.model == "b0":
        from scanvidence.models.backbone import SegResNetB0 as Model
    else:
        from scanvidence.models.cnn_shared import B1Segmentor as Model
    model = Model(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_max = max(1, len(train_ids) // args.accum * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and dev == "cuda")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_epoch, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best_val_dice"]
        np.random.set_state(ck["rng"]["numpy"])  # noqa: NPY002  (legacy checkpoint state)
        torch.random.set_rng_state(ck["rng"]["torch"])

    def one_step(vol: np.ndarray, seg: np.ndarray) -> float:
        x = torch.from_numpy(vol)[None].to(dev)
        y = torch.from_numpy(seg)[None].to(dev)
        with torch.amp.autocast("cuda", enabled=args.amp and dev == "cuda"):
            o = model(x)
            if isinstance(o, (tuple, list)):
                o = o[0]
            loss = loss_fn(o, y) / args.accum
        scaler.scale(loss).backward()
        return float(loss) * args.accum

    # --- production-step profile (allocated + reserved), written before training ---
    vol0, seg0 = load_case(groups, train_ids[0])
    fg = np.argwhere(seg0 > 0)
    center0 = fg.mean(0) if len(fg) else np.array(vol0.shape[1:]) / 2
    vc0, sc0 = crop(vol0, seg0, center0)
    for _ in range(5):
        one_step(vc0, sc0)
    opt.zero_grad(set_to_none=True)
    if dev == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.profile_steps):
        one_step(vc0, sc0)
    if dev == "cuda":
        torch.cuda.synchronize()
    prof = {
        "model": args.model,
        "precision": "amp" if args.amp else "fp32",
        "step_ms": round((time.perf_counter() - t0) / args.profile_steps * 1000, 1),
        "allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if dev == "cuda"
        else 0.0,
        "reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if dev == "cuda" else 0.0,
        "params": n_params,
    }
    (out / f"profile-{args.model}.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps(prof, indent=2))
    opt.zero_grad(set_to_none=True)

    history: list[dict] = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        order = train_ids[:]
        rng.shuffle(order)
        losses: list[float] = []
        for i, cid in enumerate(order):
            vol, seg = load_case(groups, cid)
            idx = np.argwhere(seg > 0)
            if len(idx) and rng.random() < 0.7:
                center = idx[rng.integers(len(idx))]
            else:
                center = [rng.integers(0, s) for s in vol.shape[1:]]
            losses.append(one_step(*crop(vol, seg, center)))
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        if (epoch + 1) % args.val_every != 0:
            continue
        model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for cid in val_ids:
                vol, seg = load_case(groups, cid)
                fg_idx = np.argwhere(seg > 0)
                c = fg_idx.mean(0) if len(fg_idx) else np.array(vol.shape[1:]) / 2
                vc, sc = crop(vol, seg, c)
                o = model(torch.from_numpy(vc)[None].to(dev))
                if isinstance(o, (tuple, list)):
                    o = o[0]
                pr = regions(o.argmax(1)[0].cpu().numpy())
                gr = regions(sc)
                scores.append(float(np.mean([dice(pr[r], gr[r]) for r in ("ET", "TC", "WT")])))
        val_score = float(np.mean(scores)) if scores else 0.0
        tag = ""
        if val_score > best:
            best = val_score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best_val_dice": best,
                    "params": n_params,
                    "config": vars(args),
                    "split_hash": split_hash,
                    # Legacy numpy state keeps checkpoints resumable across numpy versions.
                    "rng": {"numpy": np.random.get_state(), "torch": torch.random.get_rng_state()},  # noqa: NPY002
                },
                out / "best.pt",
            )
            tag = " *best"
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(losses)), 4),
                "val_dice": round(val_score, 4),
            }
        )
        print(
            f"[epoch {epoch + 1}/{args.epochs}] train_loss {history[-1]['train_loss']:.4f} "
            f"val_dice {val_score:.4f} best {best:.4f}{tag}",
            flush=True,
        )
        (out / "history.json").write_text(json.dumps(history, indent=2))

    (out / "run.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "params": n_params,
                "seed": args.seed,
                "split_hash": split_hash,
                "precision": "amp" if args.amp else "fp32",
                "best_val_dice": best,
                "profile": prof,
                "history": history,
            },
            indent=2,
        )
    )
    print(f"done: best val Dice {best:.4f}, params {n_params}, checkpoint {out / 'best.pt'}")


if __name__ == "__main__":
    main()
