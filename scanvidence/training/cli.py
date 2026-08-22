"""Command-line entry point for B0 segmentation training.

Runs the Milestone 2 / 3 gating ladder on the training machine:

    python -m scanvidence.training overfit --data-root <BraTS-GLI root> ...
    python -m scanvidence.training pilot  --data-root <root>  --max-cases 100
    python -m scanvidence.training train  --data-root <root>

Every run performs a patient-level split (no leakage) at a fixed seed,
hashes the partitions (``split_hashes`` land in ``run.json`` so the split
can be frozen later), drops incomplete cases with an explicit exclusion
count, and refuses to train if the label gate fires. The documented
training contract — AdamW, AMP, batch 1, gradient accumulation, clipping
after unscaling — is configured here and executed by
``training.SegmentationTrainer``.

Windows notes
-------------
- Works with ``python -m scanvidence.training`` from an Anaconda prompt;
  the entry point lives behind ``if __name__ == "__main__"`` so
  multi-worker DataLoaders are safe under the spawn start method.
- The T1000 (Turing) has no tensor cores — AMP will not speed up forward
  passes, only reduce memory; pass ``--no-amp`` if you prefer determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from scanvidence.data import assert_no_leakage, patient_level_split
from scanvidence.data.datasets import BraTSDataset
from scanvidence.models.backbone import SegResNetB0
from scanvidence.training.BraTSPatchDataset import BraTSPatchDataset
from scanvidence.training.Loss import SegmentationLoss
from scanvidence.training.SegmentationTrainer import SegmentationTrainer

ENV_DATA_ROOT = "BRATS_DATA_ROOT"
DEFAULT_SEED = 17


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scanvidence.training",
        description="B0 segmentation training (overfit check -> pilot -> full).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--data-root", default=None, help=f"BraTS case root. Defaults to env {ENV_DATA_ROOT}."
    )
    parent.add_argument("--track", default="GLI", help="BraTS track filter (GLI).")
    parent.add_argument("--patch", type=int, default=96, help="Cubic patch edge (96).")
    parent.add_argument("--accum", type=int, default=8, help="Gradient accumulation steps.")
    parent.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        default=True,
        help="Disable AMP (T1000 has no tensor cores; AMP only saves memory).",
    )
    parent.add_argument("--max-grad-norm", type=float, default=1.0)
    parent.add_argument("--lr", type=float, default=1e-4)
    parent.add_argument("--weight-decay", type=float, default=1e-5)
    parent.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parent.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    parent.add_argument("--out-dir", default="runs/b0", help="Checkpoints + run.json.")
    parent.add_argument("--val-frac", type=float, default=0.2, help="Patient-level val share.")
    parent.add_argument("--widths", default="16,32,64,128")
    parent.add_argument("--num-classes", type=int, default=4)
    parent.add_argument("--dropout", type=float, default=0.0)
    parent.add_argument("--foreground-prob", type=float, default=0.5)
    parent.add_argument("--no-aug", dest="augment", action="store_false", default=True)
    parent.add_argument("--log-every", type=int, default=20)
    parent.add_argument(
        "--profile-steps",
        type=int,
        default=0,
        help="Profile first N training steps (needs --out-dir).",
    )
    parent.add_argument(
        "--remap-legacy-four",
        action="store_true",
        help="Explicit opt-in: remap legacy label 4 -> 3.",
    )
    parent.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false", default=True
    )
    parent.add_argument(
        "--max-mem-fraction",
        type=float,
        default=0.0,
        help="Cap CUDA VRAM to this fraction of the GPU (e.g. 0.8 = 80%%). "
        "0 disables the cap. B0 at batch 1 uses ~1.5 GB, but the cap keeps "
        "peaks from ever threatening an OOM (and keeps the fan sane).",
    )

    overfit = sub.add_parser("overfit", parents=[parent], help="2-case overfit gate.")
    overfit.add_argument("--epochs", type=int, default=60)
    overfit.add_argument(
        "--overfit-dice",
        type=float,
        default=0.90,
        help="Required mean ET/TC/WT train Dice to pass.",
    )

    pilot = sub.add_parser("pilot", parents=[parent], help="Small controlled run.")
    pilot.add_argument("--epochs", type=int, default=30)
    pilot.add_argument("--max-cases", type=int, default=100)
    pilot.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to resume from (must contain state_dict, optimizer, scaler, rng)"
        ),
    )

    train = sub.add_parser("train", parents=[parent], help="Main B0 training run.")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument(
        "--max-cases", type=int, default=0, help="0 = use every usable training case."
    )
    train.add_argument(
        "--resume",
        type=str,
        default=None,
        nargs="?",
        const="AUTO",
        help=(
            "Resume from checkpoint. Pass an explicit path or use --resume "
            "with no value to load <out-dir>/best.pt."
        ),
    )
    return parser


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def set_rng_states(state: dict[str, Any]) -> None:
    """Restore Python, NumPy, and torch RNG states."""

    def _as_cpu_bytetensor(value: Any) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002  (checkpoint stores legacy state)

    torch_cpu = _as_cpu_bytetensor(state["torch_cpu"])
    torch.set_rng_state(torch_cpu)

    cuda_state = state.get("torch_cuda")
    if torch.cuda.is_available() and cuda_state is not None:
        if not isinstance(cuda_state, (list, tuple)):
            cuda_state = [cuda_state]
        cuda_state = [_as_cpu_bytetensor(v) for v in cuda_state]
        torch.cuda.set_rng_state_all(cuda_state)


def _partition_hashes(partition: list[dict]) -> str:
    ids = sorted(str(r["patient_id"]) for r in partition)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _usable_records(records: list[dict]) -> tuple[list[dict], int]:
    needed = {"t1n", "t1c", "t2w", "t2f"}
    usable = [
        r for r in records if needed <= set(r["available_sequences"]) and r["seg_path"] is not None
    ]
    return usable, len(records) - len(usable)


def _make_loader(
    records: list[dict],
    *,
    patch: int,
    seed: int,
    workers: int,
    shuffle: bool,
    augment: bool,
    foreground_prob: float,
    remap_legacy_four: bool,
    use_cuda: bool,
) -> DataLoader:
    dataset = BraTSPatchDataset(
        records,
        patch_size=patch,
        foreground_prob=foreground_prob,
        seed=seed,
        augment=augment,
        remap_legacy_four=remap_legacy_four,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
        generator=generator,
    )


def _run_common(
    args: argparse.Namespace, records: list[dict]
) -> tuple[dict[str, Any], list[dict], list[dict]]:
    """Shared: split, hash, log; returns (run metadata, train, val)."""
    train, val, test = patient_level_split(
        records, ratios=(1.0 - args.val_frac, args.val_frac, 0.0), seed=args.seed
    )
    assert_no_leakage(train, val, test)
    return (
        {
            "cmd": " ".join(sys.argv[1:]),
            "seed": args.seed,
            "split": {
                "ratios": [round(1.0 - args.val_frac, 4), args.val_frac, 0.0],
                "train_cases": len(train),
                "val_cases": len(val),
                "train_hash": _partition_hashes(train),
                "val_hash": _partition_hashes(val),
                "commit_me_when_frozen": True,
            },
        },
        train,
        val,
    )


def _build_model_and_trainer(
    args: argparse.Namespace, run_meta: dict
) -> tuple[SegResNetB0, SegmentationTrainer]:
    widths = tuple(int(w) for w in args.widths.split(","))
    model = SegResNetB0(
        in_channels=4,
        num_classes=args.num_classes,
        widths=widths,
        dropout=args.dropout,
    )
    run_meta["architecture"] = {
        "name": "SegResNetB0",
        "widths": list(widths),
        "in_channels": 4,
        "num_classes": args.num_classes,
        "dropout": args.dropout,
        "parameter_count": model.parameter_count,
    }
    trainer = SegmentationTrainer(
        model,
        config={
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "grad_accum_steps": args.accum,
            "max_grad_norm": args.max_grad_norm,
            "patch_size": args.patch,
            "foreground_prob": args.foreground_prob,
            "run_meta": run_meta,
        },
    )
    if args.command == "overfit":
        # Overfit gate must keep gradients on minority classes to verify memorization.
        trainer.loss_fn = SegmentationLoss(
            dice_weight=1.0,
            ce_weight=0.5,
            smooth=1e-5,
            ignore_empty_classes=False,
        )
    return model, trainer


def _hardware_info() -> dict[str, str]:
    info: dict[str, str] = {
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        info["gpu_name"] = gpu_name
        props = torch.cuda.get_device_properties(0)
        info["gpu_mem_bytes"] = str(props.total_memory)
        info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        # NVIDIA T1000 (TU117, cc 7.5) ships without tensor cores.
        info["gpu_has_tensor_cores"] = "false" if "t1000" in gpu_name.lower() else "unknown"
    return info


def _run(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root or os.environ.get(ENV_DATA_ROOT, ""))
    if not data_root.is_dir():
        print(
            f"Data root not found: {data_root!r}. Pass --data-root or set "
            f"{ENV_DATA_ROOT} (on the Windows box, e.g. "
            f"D:\\datasets\\BraTS-GLI-2023).",
            file=sys.stderr,
        )
        return 2

    records = BraTSDataset(str(data_root), track=args.track).discover()
    if not records:
        print(f"No BraTS-{args.track} cases discovered under {data_root}.", file=sys.stderr)
        return 2

    usable, excluded = _usable_records(records)
    print(
        f"discovered {len(records)} cases; usable (4 modalities + mask): {len(usable)} "
        f"; excluded: {excluded}"
    )
    if not usable:
        print("No usable cases; refusing to train on an empty cohort.", file=sys.stderr)
        return 2

    run_meta, train, val = _run_common(args, usable)
    if args.command == "overfit":
        train = sorted(train, key=lambda r: r["patient_id"])[:2]
        val = []
        args.epochs = max(args.epochs, 300)
        args.lr = 3e-4
        print(f"overfit gate: training on {len(train)} cases only.")
    elif args.max_cases:
        train = train[: args.max_cases]
        print(
            f"{args.command}: training on {len(train)} cases of {len(train) + len(val)} "
            f"(val {len(val)})."
        )

    use_cuda = torch.cuda.is_available()
    _set_seed(args.seed, args.deterministic)

    if args.command == "overfit":
        # Overfit contract: train and gate observe the same, tumor-centered patches.
        train_loader = _make_loader(
            train,
            patch=args.patch,
            seed=args.seed,
            workers=args.workers,
            shuffle=False,
            augment=False,
            foreground_prob=1.0,
            remap_legacy_four=args.remap_legacy_four,
            use_cuda=use_cuda,
        )
        val_loader = _make_loader(
            train,
            patch=args.patch,
            seed=args.seed,
            workers=args.workers,
            shuffle=False,
            augment=False,
            foreground_prob=1.0,
            remap_legacy_four=args.remap_legacy_four,
            use_cuda=use_cuda,
        )
    else:
        train_loader = _make_loader(
            train,
            patch=args.patch,
            seed=args.seed,
            workers=args.workers,
            shuffle=True,
            augment=args.augment,
            foreground_prob=args.foreground_prob,
            remap_legacy_four=args.remap_legacy_four,
            use_cuda=use_cuda,
        )
        val_loader = _make_loader(
            val,
            patch=args.patch,
            seed=args.seed,
            workers=args.workers,
            shuffle=False,
            augment=False,
            foreground_prob=0.3,
            remap_legacy_four=args.remap_legacy_four,
            use_cuda=use_cuda,
        )

    run_meta["hardware"] = _hardware_info()
    model, trainer = _build_model_and_trainer(args, run_meta)
    optimizer = trainer.optimizer
    scaler = trainer.scaler
    scheduler = None
    out_dir = Path(args.out_dir)
    resume_path = getattr(args, "resume", None)
    if resume_path == "AUTO":
        resume_path = str(out_dir / "best.pt")
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=trainer.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if optimizer is None:
            raise RuntimeError("trainer optimizer is not initialized")
        optimizer_state = ckpt.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if ckpt.get("scaler") is not None and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if "scheduler" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if isinstance(ckpt.get("rng"), dict):
            set_rng_states(ckpt["rng"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        start_epoch = 0

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        out_dir=out_dir,
        log_every=args.log_every,
        profile_steps=args.profile_steps,
    )

    if args.command == "overfit":
        achieved = history.best_val_dice
        ok = achieved >= args.overfit_dice
        print(
            f"overfit gate: val ET/TC/WT Dice {achieved:.4f} "
            f"(required >= {args.overfit_dice:.2f}) -> {'PASS' if ok else 'FAIL'}"
        )
        return 0 if ok else 1

    print(
        f"done: best val Dice {history.best_val_dice:.4f} (epoch {history.best_epoch}), "
        f"params {model.parameter_count}, mean step {np.mean(history.step_time_s) * 1e3:.1f} ms, "
        f"checkpoint {out_dir / 'best.pt'}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training CLI; returns the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        return _run(args)
    except (ValueError, RuntimeError) as exc:
        print(f"training aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
