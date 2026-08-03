# Scanvidence

Explainable, uncertainty-aware medical imaging detection — a unified
platform for **brain tumor** and **Alzheimer's disease** detection, with a
quantum-vs-classical feature-selection ablation. Every pipeline stage
(data, preprocessing, models, radiomics, XAI, calibration, evaluation) is
composable, modular, and backed by a strict patient-level data-splitting
guarantee. Public-benchmark-data only (BraTS, ADNI).

Part of the [Scanvidence](https://github.com/Scanvidence) organization.

|  | **[Documentation](docs/ARCHITECTURE.md)** · **[Contributing](CONTRIBUTING.md)** · **[Changelog](CHANGELOG.md)** |
|---|---|
| **Open Source** | [![GitHub License](https://img.shields.io/github/license/Scanvidence/scanvidence)](https://github.com/Scanvidence/scanvidence/blob/main/LICENSE) |
| **CI/CD** | [![CI](https://img.shields.io/github/actions/workflow/status/Scanvidence/scanvidence/ci.yml?logo=github)](https://github.com/Scanvidence/scanvidence/actions/workflows/ci.yml) [![Nightly](https://img.shields.io/github/actions/workflow/status/Scanvidence/scanvidence/nightly.yml?logo=github&label=nightly)](https://github.com/Scanvidence/scanvidence/actions/workflows/nightly.yml) [![codecov](https://codecov.io/gh/Scanvidence/scanvidence/graph/badge.svg)](https://codecov.io/gh/Scanvidence/scanvidence) |
| **Code** | [![Python Versions](https://img.shields.io/pypi/pyversions/scanvidence)](https://pypi.org/project/scanvidence/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

## Key Features

| Feature | Description |
|--------|-------------|
| [**Task-level API**](scanvidence/tasks/) | End-to-end orchestrators (`BrainTumorTask`, `AlzheimersTask`) that wire preprocessing → model → calibration → XAI into one `run()` call. |
| [**Data Loaders & Splitting**](scanvidence/data/) | BraTS and ADNI loaders with strict **patient-level splitting** — no patient's records ever appear in more than one partition. |
| [**Preprocessing**](scanvidence/preprocessing/) | Skull stripping, normalization, registration, and a composable `Pipeline`. |
| [**Model Zoo**](scanvidence/models/) | Classification (tumor, Alzheimer's) and segmentation model wrappers. |
| [**Radiomics**](scanvidence/radiomics/) | PyRadiomics feature extraction from segmentation masks. |
| [**Explainability (XAI)**](scanvidence/xai/) | Grad-CAM, SHAP, and LIME with quantitative validation of explanations. |
| [**Uncertainty Calibration**](scanvidence/calibration/) | Temperature scaling and MC-Dropout for well-calibrated, uncertainty-aware predictions. |
| [**Quantum Feature Selection**](scanvidence/quantum/) | QUBO-based feature selection — quantum-vs-classical ablation (stretch objective, optional extra). |
| [**Evaluation**](scanvidence/evaluation/) | Metrics with bootstrap CIs and paired statistical tests (DeLong, McNemar). |
| [**Deployment**](scanvidence/api/) | FastAPI REST API for clinical deployment and a Streamlit research demo. |

## Quickstart

### Installation

```bash
pip install -e ".[dev]"
pre-commit install
pytest -m "not slow"
```

Optional extras: `quantum`, `segmentation`, `demo`, `api`, `tracking`, `docs`.

### Examples

#### Task-level API (brain tumor detection)

```python
from scanvidence.tasks import BrainTumorTask

# Load config, run the full pipeline, get a calibrated, explained result.
task = BrainTumorTask.from_config("configs/brain_tumor.yaml")
result = task.run(scan_path="patient_001.nii.gz")

result.prediction      # class label
result.confidence      # calibrated probability
result.uncertainty     # MC-Dropout estimate
result.explanations    # Grad-CAM / SHAP / LIME outputs
```

#### Component-level API (Alzheimer's disease)

```python
from scanvidence.tasks import AlzheimersTask

task = AlzheimersTask.from_config("configs/alzheimers.yaml")
result = task.run(scan_path="subject_042.nii.gz")
```

#### Patient-level data splitting

```python
from scanvidence.data.splitting import patient_level_split

records = [...]  # list of patient scan records
train, val, test = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=42)
# No patient appears in more than one partition — guaranteed.
```

#### Quantum-vs-classical feature-selection ablation

```python
from scanvidence.quantum import QUBOSelector
from scanvidence.radiomics import PyRadiomicsExtractor

features = PyRadiomicsExtractor().transform(mask, image)
qubo = QUBOSelector()  # stretch objective — optional extra
selected = qubo.fit_select(features, labels)
```

## Architecture

```
Scan Input → Preprocess → Model → Calibrate → Explain → Evaluate
                                                  ↑
                                        Quantum Feature Selection
                                           (stretch objective)
```

The library follows a **pgmpy-style** architecture:

- Every subpackage has a `base.py` defining abstract base classes
- `__init__.py` files re-export key symbols for clean imports
- Global config via `scanvidence.global_vars` (device, backend, dtype)
- One class per file, PascalCase filenames for class files

## Why no GPU jobs in CI

GitHub-hosted runners don't have GPUs. `ci.yml` tests **logic** — data
splitting, metric math, config loading, QUBO formulation — on tiny
synthetic inputs, in seconds, on every PR. Real training runs locally or
on the lab GPU/Colab and gets logged to W&B/MLflow.

## Status

`scanvidence/data/splitting.py` is a real, working implementation —
verified against every case in `tests/test_data/test_splitting.py`.
Every other subpackage contains base classes and documented stubs
describing what belongs there; that's the team's project work.

### Resources and Links

- **Architecture Notes:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Configuration Examples:** [configs/brain_tumor.yaml](configs/brain_tumor.yaml) · [configs/alzheimers.yaml](configs/alzheimers.yaml)
- **Contributing Guide:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **CI:** [.github/workflows/ci.yml](.github/workflows/ci.yml) · [.github/workflows/nightly.yml](.github/workflows/nightly.yml)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/Scanvidence/scanvidence/issues)

## Contributing

We welcome all contributions — not just code — to Scanvidence. Please
refer to our [contributing guide](CONTRIBUTING.md) for more details.
Before opening a PR, keep `tests/test_data/test_splitting.py` green: that
test is not allowed to be skipped, weakened, or marked xfail.
