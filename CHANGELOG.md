# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/). Tag a
version at the end of each phase (see `CONTRIBUTING.md`) so every reported
result maps to an exact commit.

## [Unreleased]

### Changed
- **BREAKING**: Renamed package from `tumor-xai` / `tumorxai` to `scanvidence`
  (aligns with the Scanvidence GitHub organization).
- Switched from `src/` layout to flat layout (`scanvidence/` at repo root),
  matching the pgmpy structural pattern.
- All imports now use `from scanvidence.xxx import ...` instead of
  `from tumorxai.xxx import ...`.

### Added
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
