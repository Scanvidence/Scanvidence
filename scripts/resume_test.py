"""Checkpoint/resume equivalence for the B0 training loop.

A short stretch of training is checkpointed with the same keys the
training CLI writes — ``state_dict``, ``optimizer``, ``scaler``,
``scheduler``, ``rng`` — and a second, identically built model resumes
from it exactly the way ``python -m scanvidence.training`` does. If the
checkpoint captures everything the loop needs, step N on the reloaded
model matches step N on the original to tolerance. That is the guarantee
the full run's ``--resume`` path depends on.

The saved ``runs/resume_test.pt`` is loadable by the real resume path,
so it doubles as a fixture for the training CLI. Outputs: that
checkpoint plus ``runs/resume_test_result.json``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from scanvidence.models.backbone import SegResNetB0
from scanvidence.training.cli import set_rng_states

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build() -> tuple[
    SegResNetB0, torch.optim.Optimizer, torch.amp.GradScaler, torch.optim.lr_scheduler.LRScheduler
]:
    model = SegResNetB0(in_channels=4, num_classes=4, widths=(16, 32, 64, 128), dropout=0.0).to(
        _DEVICE
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=_DEVICE == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    return model, opt, scaler, sched


def step(
    model: SegResNetB0,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    sched: torch.optim.lr_scheduler.LRScheduler,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """One scaled forward/backward in the training loop's step order."""
    opt.zero_grad()
    with torch.amp.autocast("cuda", enabled=_DEVICE == "cuda"):
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = torch.nn.functional.cross_entropy(out, y)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()
    sched.step()
    return float(loss.detach())


def capture_rng() -> dict:
    """RNG snapshot in the trainer's checkpoint format (see set_rng_states)."""
    return {
        "python": random.getstate(),
        # Legacy state keeps the checkpoint loadable by the training resume path.
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def main() -> None:
    torch.manual_seed(17)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    x = torch.randn(1, 4, 96, 96, 96, device=_DEVICE)
    y = torch.randint(0, 4, (1, 96, 96, 96), device=_DEVICE)

    A = build()
    for _ in range(3):
        step(*A, x, y)
    ckpt = {
        "state_dict": A[0].state_dict(),
        "optimizer": A[1].state_dict(),
        "scaler": A[2].state_dict(),
        "scheduler": A[3].state_dict(),
        "rng": capture_rng(),
    }
    torch.save(ckpt, "runs/resume_test.pt")
    # Reload from disk so the resumed model gets an independent copy of the
    # pre-step-4 state; the in-memory dict aliases A's live tensors, which
    # step 4 below mutates in place.
    saved = torch.load("runs/resume_test.pt", map_location="cpu", weights_only=False)
    loss_a = step(*A, x, y)  # original's step 4

    B = build()
    B[0].load_state_dict(saved["state_dict"])
    B[1].load_state_dict(saved["optimizer"])
    B[2].load_state_dict(saved["scaler"])
    B[3].load_state_dict(saved["scheduler"])
    set_rng_states(saved["rng"])
    loss_b = step(*B, x, y)  # reloaded continuation's step 4

    ok = abs(loss_a - loss_b) < 1e-4 and all(
        torch.allclose(pa, pb, atol=1e-5)
        for pa, pb in zip(A[0].parameters(), B[0].parameters(), strict=True)
    )
    print(f"loss_orig {loss_a:.6f} vs loss_resumed {loss_b:.6f} -> {'PASS' if ok else 'FAIL'}")
    Path("runs/resume_test_result.json").write_text(
        json.dumps({"loss_orig": loss_a, "loss_resumed": loss_b, "pass": ok})
    )


if __name__ == "__main__":
    main()
