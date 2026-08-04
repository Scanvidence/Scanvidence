# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Tag a
version at the end of each phase (see `CONTRIBUTING.md`) so every reported
result maps to an exact commit.

## [Unreleased]

### Changed
- **BREAKING**: Renamed package from `tumor-xai` / `tumorxai` to `scanvidence`
  (aligns with the Scanvidence GitHub organization).
- README: documented the planned compute split — training on the college
  NVIDIA A1000 8GB GPU (Docker or pinned venv), development on personal
  Mac/Windows machines (native venv, no GPU) — with the VRAM implications
  for nnU-Net patches and heavy 3D comparators, and a Windows (WSL2) /
  Linux Docker quickstart.
- Canonical contributor instructions moved to `DEVELOPMENT.md`; `AGENTS.md`
  now points there (same rules, single source of truth).
- Package `__init__.py` re-exports for concrete classes so the documented
  imports work: `calibration` (TemperatureScaling, MCDropout), `evaluation`
  (ClassificationMetrics, StatisticalTests), `quantum` (QUBOSelector),
  `preprocessing` (Pipeline, Normalizer, SkullStripper, Registration),
  `xai` (GradCAM, SHAPExplainer, LIMEExplainer), `radiomics`
  (PyRadiomicsExtractor), `api` (create_app) — all with `__all__`.
- Switched from `src/` layout to flat layout (`scanvidence/` at repo root),
  matching the pgmpy structural pattern.
- All imports now use `from scanvidence.xxx import ...` instead of
  `from tumorxai.xxx import ...`.

### Added
- Test suites for calibration, evaluation, quantum, preprocessing, API
  endpoints, and the CLI entry point — the fast tier grew from 14 to 45
  tests (still seconds, still no GPU or real data).
- `.codecov.yml` coverage gate: project coverage drift (auto target, 1%
  threshold) and patch coverage (60%) checks on every PR.
- Dev extras now include `fastapi` and `httpx` so the API tests run in
  the standard `pip install -e ".[dev]"` environment.
- `AGENTS.md` — agent-focused working instructions (commands,
  conventions, safety rules, CI gotchas).
- Org logo (`logo/logo.jpg`) in the README header, pgmpy-style.
- `scanvidence.__main__` — CLI entry point (`python -m scanvidence detect`)
  so the Dockerfile's default command works.
- `scanvidence/py.typed` — PEP 561 marker for typed-package consumers.
- `scanvidence.models` re-exports `TumorClassifier`, `AlzheimersClassifier`,
  and `TumorSegmentor` (matching the documented imports).
- Python 3.13 and 3.14: added to CI test matrix and classifiers;
  documented in README (radiomics extra excluded from 3.13/3.14 CI).
- `scanvidence.data.datasets.OASISDataset` — OASIS-1/2/3/4 adapter
  (primary dataset for the Alzheimer's track), with `references.bib`
  entries (Marcus 2007/2010, LaMontagne 2019).
- `docs/references.bib` — BibTeX bibliography for datasets, methods, and
  evaluation, with pgmpy-style keys and pipeline-stage keywords.
- README: research design section documenting hypotheses H1a–H3, the
  pre-specified evaluation protocol, ethics/scope, and reproducibility
  commitments from the research proposal.
- `docs/ARCHITECTURE.md`: subpackage → hypothesis map and evaluation
  protocol section.
- Unified architecture for **brain tumor** and **Alzheimer's disease** detection.
- `scanvidence.global_vars` — pgmpy-style global Config singleton and logger.
- `scanvidence.base.types` — shared Pydantic types (`ScanInput`, `DetectionResult`, `XAIResult`).
- `base.py` abstract base classes in every subpackage (pgmpy pattern).
- `scanvidence.tasks` — task orchestrators (`BrainTumorTask`, `AlzheimersTask`).
- `scanvidence.api` — FastAPI REST API with unified `/detect/{task}` endpoint.
- `scanvidence.data.datasets` — dataset adapters for BraTS and ADNI.
- `scanvidence.data.NIfTILoader` / `DICOMLoader` — format-specific loaders.
- `scanvidence.preprocessing.Pipeline` — composable transform pipeline.
- `scanvidence.calibration` — `TemperatureScaling`, `MCDropout`.
- `scanvidence.utils` — device detection, checkpoint utilities.
- YAML task configs in `configs/`.
- `__init__.py` re-exports for clean imports across all subpackages.

## [0.1.0] — Phase 1: Foundation
- Repository scaffold: packaging, CI, pre-commit, issue/PR templates.
- Reference implementation and test for patient-level data splitting.
