# Contributing

This repo is reviewed the same way the research proposal is written: every
change is small, scoped, and defensible on its own — the same discipline
behind modeling this on projects like pgmpy, numpy, and sktime rather than
a personal script folder.

## Workflow

1. Work starts as an **Enhancement Proposal** (issue template:
   `enhancement_proposal.md`) — what, why, how it's tested.
2. Break it into issues, and issues into sub-issues small enough to land as
   one PR each.
3. Branch per issue: `feat/<issue-number>-short-description` or
   `fix/<issue-number>-short-description`.
4. Open a PR referencing the issue (`Closes #12`). CI must be green.
   One teammate review required — CODEOWNERS auto-requests the right
   person based on the track you touched.
5. No direct pushes to `main`; it's a protected branch.

## Before you push

```bash
pip install -e ".[dev]"
pre-commit install        # once — formatting/linting then run automatically
pytest -m "not slow"      # the fast tier, same as CI
```

If your change touches data splitting, calibration, or evaluation code,
also run the split-leakage test explicitly and make sure it's green:

```bash
pytest tests/test_data/test_splitting.py -v
```

That test is never allowed to be skipped, weakened, or marked `xfail` —
every non-inferiority claim in the proposal (H1a, H1b) depends on the
guarantee it checks.

## Adding a new subpackage or class

Follow the pgmpy pattern:

1. Create a `base.py` with an abstract base class in the subpackage
2. Create a PascalCase `.py` file for each concrete class
3. Re-export key symbols in `__init__.py`
4. Mirror the structure in `tests/test_<subpackage>/`
5. Add NumPy-style docstrings to every public method

## Testing tiers

- **Fast (default, every PR):** logic on tiny synthetic inputs — splitting,
  metric math, QUBO formulation, config loading. No GPU, no real datasets,
  runs in seconds.
- **Slow (`@pytest.mark.slow`, nightly + on demand):** small real-data
  integration checks.
- **GPU (`@pytest.mark.gpu`, manual only):** anything needing an actual
  GPU. CI never runs these — GitHub-hosted runners don't have one. Real
  training happens locally / on the lab GPU / Colab and gets logged to
  W&B, not asserted on in CI.

## Code style

- Ruff for linting and formatting (line length 100) — `ruff check .` and
  `ruff format .`
- Type hints on public functions; mypy runs in CI against `scanvidence/`.
- NumPy-style docstrings.
- PascalCase filenames for class files (`GradCAM.py`, `TumorClassifier.py`),
  snake_case for utility modules (`splitting.py`, `device.py`).

## What never gets committed

- Raw or processed dataset files — BraTS/TCGA/ADNI/OASIS license terms
  forbid redistributing raw scans, on top of the size problem. `.gitignore`
  already excludes `data/raw/` and `data/processed/`.
- Model checkpoints — log them to W&B/MLflow instead.
- API tokens (IBM Quantum, D-Wave, W&B) — copy `.env.example` to `.env`,
  never hardcode a token in source.

## Versioning

Tag a release at the end of each phase in the timeline (`v0.1.0` = end of
Phase 1, `v0.2.0` = end of Phase 2, ...) so any reported result can be
traced back to the exact code that produced it — this is what the
reproducibility checklist's "artifact versioning" commitment means in
practice.
