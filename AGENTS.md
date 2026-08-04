# AGENTS.md

Instructions for AI coding agents (and humans) working in this repo.
Read this first. When in doubt, this file wins over chat-level guesses.

## What this project is

Scanvidence is a research platform for explainable, uncertainty-aware
medical imaging detection: brain tumor and Alzheimer's disease tracks
share one composable pipeline (data → preprocess → model → calibrate →
explain → evaluate), plus an optional quantum-vs-classical (QUBO)
feature-selection ablation. It is the codebase for a final-year research
proposal whose claims are pre-specified: hypotheses H1a–H3, locked
patient-level test partitions, non-inferiority margins, and fixed
decision rules. **The code exists to answer those hypotheses honestly —
not to maximize accuracy numbers.**

Constraints that follow from this:

- **Public benchmark data only** (BraTS, OASIS, ADNI, TCGA, Figshare).
  Never add, download, or commit real patient data, raw scans, or data
  derivatives. `.gitignore` already excludes `data/raw`, `data/processed`,
  `*.nii*`, `*.dcm`, `*.mha`.
- **Research artifacts, not clinical claims.** No README/docs wording may
  imply diagnosis or clinical readiness.
- **Never weaken `tests/test_data/test_splitting.py`.** It is not allowed
  to be skipped, weakened, or marked `xfail`. Every non-inferiority claim
  depends on the patient-level no-leakage guarantee it checks.

## Repo map

| Path | What it is |
|---|---|
| `scanvidence/` | The package. One subpackage per pipeline stage. |
| `scanvidence/base/` | Shared Pydantic types (`ScanInput`, `DetectionResult`, `XAIResult`). |
| `scanvidence/data/` | Loaders, dataset adapters (BraTS, OASIS, ADNI), `splitting.py` (safety-critical). |
| `scanvidence/preprocessing/` | SkullStripper, Normalizer, Registration, composable `Pipeline`. |
| `scanvidence/models/` | Classification + segmentation wrappers; `TumorClassifier`, `AlzheimersClassifier`, `TumorSegmentor` re-exported at package level. |
| `scanvidence/radiomics/` | PyRadiomics extraction (stub; pyradiomics is an optional extra). |
| `scanvidence/xai/` | GradCAM, SHAPExplainer, LIMEExplainer. |
| `scanvidence/calibration/` | TemperatureScaling, MCDropout. |
| `scanvidence/quantum/` | QUBOSelector (stretch objective, optional extra). |
| `scanvidence/evaluation/` | Metrics, bootstrap CIs, DeLong/McNemar tests. |
| `scanvidence/tasks/` | `BrainTumorTask`, `AlzheimersTask` orchestrators. |
| `scanvidence/api/` + `scanvidence/demo/` | FastAPI + Streamlit (research demo only). |
| `tests/` | Mirrors `scanvidence/`. |
| `docs/ARCHITECTURE.md` | Pipeline map, subpackage → hypothesis map, evaluation protocol. |
| `docs/references.bib` | BibTeX bibliography (48+ entries, pgmpy-style keys). |
| `configs/` | Task YAML configs. |

Many classes are **documented stubs** on purpose (team project work).
Don't "help" by deleting them or rewriting them into something the
proposal doesn't call for.

## Commands (run these, not your memory of them)

```bash
pip install -e ".[dev]"        # setup
pytest -m "not slow and not gpu"   # fast tier, same as CI
ruff check .                   # lint
ruff format .                  # format (run BEFORE ruff check to auto-fix)
mypy scanvidence --ignore-missing-imports   # typecheck
```

The full CI fast tier is exactly:

```bash
ruff check . && ruff format --check . && mypy scanvidence --ignore-missing-imports && pytest -m "not slow and not gpu"
```

Everything must pass before you propose a commit.

## Conventions (pgmpy pattern — follow strictly)

- Every subpackage has `base.py` with an abstract base class.
- One concrete class per file, PascalCase filename (`GradCAM.py`).
  Utility modules snake_case (`splitting.py`).
- `__init__.py` re-exports public symbols, with `__all__`.
- NumPy-style docstrings on every public method.
- Type hints on public functions; mypy is run in CI.
- Global config lives in `scanvidence.global_vars` (`config`, `logger`).
- Imports: `from scanvidence.tasks import BrainTumorTask` (package-level
  re-exports), never deep `from scanvidence.tasks.BrainTumorTask import ...`.

## Known gotchas (agents trip on these)

1. **Ruff version drift.** `pyproject.toml` says `ruff>=0.4`; CI installs
   the latest (0.16.x), which formats **markdown code blocks** in README
   and docs, and applies newer rules. After editing any `.md`, run
   `ruff format .` — the version you have locally is what CI checks are
   measured against for anything you format; if your local ruff is old,
   upgrade: `pip install -U ruff`.
2. **pyradiomics is not in core deps.** It has no wheels for Python
   ≥ 3.10, so it lives in the `radiomics` extra. Never add it back to
   `dependencies`. The radiomics module imports it lazily (or not at all
   — it's a stub).
3. **No GPU in CI.** GitHub runners have none. CI tests logic on tiny
   synthetic inputs only. Don't add tests that train models or need real
   data; mark them `@pytest.mark.slow` or `@pytest.mark.gpu`.
4. **Python matrix.** CI tests 3.10–3.14. Keep code compatible with all;
   no 3.13/3.14-only syntax without a very good reason.
5. **The Dockerfile** installs `.[segmentation,tracking]` and runs
   `python -m scanvidence` — keep `scanvidence/__main__.py` importable
   without optional extras.

## Rules for changes

- Small, single-purpose commits; one commit per logical unit. This repo's
  history is deliberately granular.
- Don't rename/re-export symbols without updating every reference —
  README examples and `docs/ARCHITECTURE.md` are part of the API surface.
- Don't change evaluation decision rules, margins, or the splitting
  guarantee to make results look better — that's the one thing the
  proposal explicitly forbids.
- Don't add dependencies casually. Heavy/fragile ones (quantum, nnU-Net,
  pyradiomics) belong in extras, isolated from core installs.
- No secrets: no API tokens, no `.env` contents, no W&B/MLflow keys.
- Don't commit checkpoints, datasets, notebooks with outputs
  (`nbstripout` handles notebooks on pre-commit).

## Before you finish

1. Run the fast tier (commands above) — all green.
2. If you touched splitting/calibration/evaluation:
   `pytest tests/test_data/test_splitting.py -v` explicitly.
3. If you touched README/docs markdown: `ruff format .` and re-check.
4. Keep the change small enough that a human teammate can review it in
   one sitting (that's the repo's review standard — one PR per issue).
