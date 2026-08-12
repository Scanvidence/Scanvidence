# B0 — compact SegResNet-style CNN

This directory contains the **B0 reference architecture** and the training
pipeline required by the Scanvidence work plan.

## Architecture (B0)

- **SegResNet-style** residual encoder–decoder with pre‑activation blocks.
- Encoder widths: ``(16, 32, 64, 128)`` – four MaxPool3d stages → 96³ → 6³ bottleneck.
- U‑Net decoder with transposed‑conv upsample + concat skip connections.
- GroupNorm everywhere, Dropout3d optional (default 0.0).
- 1×1×1 head to 4 classes (BraTS‑GLI: 0=background, 1=necrotic core, 2=edema, 3=enhancing tumor).
- ``from_config()`` validates a YAML dict so every run is a complete record.

**Parameter count:** ~1.6 M (batch‑1 on 96³, 4‑channel patches).

## Training pipeline (`scanvidence/training/`)

- `BraTSPatchDataset` — 96³ cubic patches, per‑modality z‑score on nonzero brain voxels,
  foreground‑aware centre sampling, augmentations (flips, 90° rotations, intensity scale/shift)
  applied *identically* to all modalities and the mask.
- `SegmentationLoss` — combined soft Dice + cross‑entropy (empty‑class ignored).
- `regional_dice` / `mean_regional_dice` — ET/TC/WT Dice with the BraTS empty‑region convention.
- `SegmentationTrainer` — AdamW, AMP (memory optimization on T1000; benchmark FP16 vs FP32 for speed), gradient accumulation,
  clip‑after‑unscaling, checkpointing, profiling (`--profile-steps`), deterministic RNG.
- `cli.py` — CLI entry point `python -m scanvidence.training {overfit|pilot|train}` with:
  - `--data-root`, `--track`, `--patch`, `--accum`, `--amp`/`--no-amp`,
  - `--max-mem-fraction` (0‑1, caps CUDA VRAM),
  - `--foreground-prob`, `--widths`, `--epochs`, `--max-cases`,
  - `--log-every`, `--profile-steps`, `--remap-legacy-four`,
  - `--no-deterministic`.
- Overfit gate: validates on the *same* patch set the model trained on (seed‑+‑index deterministic).
- Pilot/train subcommands train & validate, write `run.json` (config, split hashes, measured metrics)
  and `best.pt` checkpoint.

## Tests (`tests/test_training/`)

- `test_BraTSPatchDataset.py` — shape, normalization, foreground sampling,
  label‑4 gate, geometry mismatch, out‑of‑range labels, augmentation consistency,
  determinism.
- `test_Loss.py` — perfect/worst prediction, gradients, weight effect.
- `test_Metrics.py` — empty‑region Dice, regional/Dice maps, mean Dice.
- `test_SegmentationTrainer.py` — fit records history, gradient accumulation flush,
  determinism, predict, checkpoint round‑trip, loss improvement on 2‑case data.
- `test_cli.py` — CLI smoke tests, overfit gate (pass/fail), pilot with `--max-cases`,
  run‑common splits without leakage.

All tests are CPU‑only, marked `slow` where they run a real training loop.

## Quick start (Windows)

```batch
:: 1. Install environment
conda create -n scanvence python=3.11 -y
conda activate scanvence
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

:: 2. Point at your data
setx BRATS_DATA_ROOT "D:\datasets\BraTS-GLI-2023"

:: 3. Run the gating ladder
python -m scanvidence.training overfit --data-root %BRATS_DATA_ROOT%
python -m scanvidence.training pilot  --data-root %BRATS_DATA_ROOT%
python -m scanvidence.training train  --data-root %BRATS_DATA_ROOT%
```

**GPU memory cap** (optional, recommended for peace of mind):

```batch
python -m scanvidence.training overfit --data-root %BRATS_DATA_ROOT% --max-mem-fraction 0.8
```

- `0.8` ⇒ caps CUDA to ~6.4 GB on an 8 GB T1000.
- `0` (default) ⇒ no cap; the model uses ~1.5 GB at batch‑1.

**BraTS‑GLI 2023 mapping** (from `~/Downloads/BraTS2023_2017_GLI_Mapping.xlsx`):
- 1 251 unique 2023 cases (the work‑plan figure).
- Labels: 0=background, 1=necrotic core, 2=edema, **3=enhancing tumor**.
- The loader refuses label 4 unless `--remap‑legacy‑four` is passed.

---
*Follow the steps in the main `scanvidence/README.md` for the full work‑plan context.*