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
