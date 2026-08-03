# Scanvidence

Explainable, uncertainty-aware medical imaging detection — a unified
platform for **brain tumor** and **Alzheimer's disease** detection, with a
quantum-vs-classical feature-selection ablation. Every prediction is
accompanied by calibrated confidence, uncertainty flags, and
quantitatively validated explanations, and every claim is backed by
locked patient-level test partitions and pre-specified statistical tests.

Part of the [Scanvidence](https://github.com/Scanvidence) organization —
final-year research project (Integral University, Lucknow), supervised by
Dr. Roshan Jahan. **Public benchmark data only: no real patient data, no
clinical deployment claims — all outputs are research artifacts.**

|  | **[Documentation](docs/ARCHITECTURE.md)** · **[Contributing](CONTRIBUTING.md)** · **[Changelog](CHANGELOG.md)** |
|---|---|
| **Open Source** | [![GitHub License](https://img.shields.io/github/license/Scanvidence/scanvidence)](https://github.com/Scanvidence/scanvidence/blob/main/LICENSE) |
| **CI/CD** | [![CI](https://img.shields.io/github/actions/workflow/status/Scanvidence/scanvidence/ci.yml?logo=github)](https://github.com/Scanvidence/scanvidence/actions/workflows/ci.yml) [![Nightly](https://img.shields.io/github/actions/workflow/status/Scanvidence/scanvidence/nightly.yml?logo=github&label=nightly)](https://github.com/Scanvidence/scanvidence/actions/workflows/nightly.yml) [![codecov](https://codecov.io/gh/Scanvidence/scanvidence/graph/badge.svg)](https://codecov.io/gh/Scanvidence/scanvidence) |
| **Code** | [![Python Versions](https://img.shields.io/pypi/pyversions/scanvidence)](https://pypi.org/project/scanvidence/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

## Key Features

| Feature | Description |
|--------|-------------|
| [**Task-level API**](scanvidence/tasks/) | End-to-end orchestrators (`BrainTumorTask`, `AlzheimersTask`) that wire preprocessing → model → calibration → XAI → evaluation into one `run()` call. |
| [**Data Loaders & Splitting**](scanvidence/data/) | BraTS, OASIS, ADNI, TCGA, and Figshare loaders with strict **patient-level splitting** — no patient's records ever appear in more than one partition. |
| [**Preprocessing**](scanvidence/preprocessing/) | Skull stripping, normalization, registration, and a composable `Pipeline`. |
| [**Model Zoo**](scanvidence/models/) | 2D slice classifiers (ResNet, DenseNet, EfficientNet, ViT) and 3D nnU-Net segmentation. |
| [**Radiomics**](scanvidence/radiomics/) | PyRadiomics feature extraction (first-order, shape, GLCM, GLRLM, GLSZM, GLDM) from segmentation masks. |
| [**Explainability (XAI)**](scanvidence/xai/) | Grad-CAM, Grad-CAM++, Integrated Gradients, SHAP, and LIME — validated quantitatively against tumor masks, not visual plausibility. |
| [**Uncertainty Calibration**](scanvidence/calibration/) | Temperature scaling, Platt scaling, isotonic regression, and MC-Dropout (ECE, Brier, reliability diagrams). |
| [**Quantum Feature Selection**](scanvidence/quantum/) | QUBO-based feature selection (IBM Quantum / D-Wave / classical fallback) vs. LASSO, RFE, MI, genetic algorithms, simulated annealing. |
| [**Evaluation**](scanvidence/evaluation/) | Bootstrap CIs, DeLong, McNemar, Wilcoxon signed-rank, Holm correction — the pre-specified protocol from the research proposal. |
| [**Deployment**](scanvidence/api/) | FastAPI REST API and a Streamlit research demo ("RESEARCH USE ONLY" banner). |

## Research Design

The pipeline is built to answer five pre-specified, falsifiable
hypotheses from the research proposal:

| Hypothesis | Claim | Decision rule |
|---|---|---|
| **H1a** | Classification non-inferiority: calibrated vs. uncalibrated classifier | One-sided α = 0.025; paired 97.5% bootstrap bound on ΔAUC, margin −0.02 |
| **H1b** | Segmentation non-inferiority: explanation-ready workflow vs. nnU-Net baseline | Paired 97.5% bootstrap bound on ΔDice, margin 0.02 |
| **H2a** | Spatial explanation validity: Grad-CAM / Grad-CAM++ / Integrated Gradients focus on tumor regions | Heatmap–mask IoU, pointing-game accuracy, false-focus rate vs. spatially shuffled baseline |
| **H2b** | Feature-attribution validity: SHAP is a faithful feature-level explanation | Top-k ablation faithfulness, bootstrap rank stability (never compared to image masks) |
| **H3** | Quantum (QUBO) feature selection matches classical baselines on a fixed feature budget (≤ 50 features) | Stability via mean pairwise Jaccard across folds + downstream performance; null results are complete findings |

Every result uses a **locked test partition** (70/15/15, stratified
patient-level), 5-fold stratified group cross-validation inside training
only, 2,000 stratified patient-bootstrap resamples, and Holm-adjusted
_p_-values within each endpoint family. A prospective power check runs
once eligible patient counts are confirmed; if the test cohort is
underpowered, the margins are retained and non-inferiority is reported as
not demonstrated — the decision rule is never changed after seeing
results.

## Quickstart

### Installation

```bash
pip install -e ".[dev]"
pre-commit install
pytest -m "not slow and not gpu"
```

Optional extras: `radiomics`, `quantum`, `segmentation`, `demo`, `api`, `tracking`, `docs`.

### Examples

#### Task-level API (brain tumor detection)

```python
from scanvidence.tasks import BrainTumorTask

# Load config, run the full pipeline, get a calibrated, explained result.
task = BrainTumorTask.from_config("configs/brain_tumor.yaml")
result = task.run(scan_path="patient_001.nii.gz")

result.prediction  # class label
result.confidence  # calibrated probability
result.uncertainty  # MC-Dropout estimate
result.explanations  # Grad-CAM / SHAP / LIME outputs
```

#### Task-level API (Alzheimer's disease)

```python
from scanvidence.tasks import AlzheimersTask

# OASIS is the primary dataset for the Alzheimer's track (T1w MRI,
# demented / nondemented labels).
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

#### Quantum-vs-classical feature-selection ablation (H3)

```python
from scanvidence.quantum import QUBOSelector
from scanvidence.radiomics import PyRadiomicsExtractor

features = PyRadiomicsExtractor().transform(mask, image)
qubo = QUBOSelector()  # stretch objective — optional extra
selected = qubo.fit_select(features, labels)
```

## Architecture

The **unified architecture** serves both detection domains with the same
composable pipeline:

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
- Parallel classification / segmentation / radiomics branches with a
  shared preprocessing front end — the QUBO branch is optional and never
  gates the core pipeline

## Ethics and Scope

- **No real patient data** — public benchmarks only (BraTS/BraTS-GLI,
  TCGA-GBM/LGG, Figshare for tumors; OASIS for Alzheimer's
  disease), each used under its own license with attribution and no
  redistribution of raw scans.
- **Research use only** — every demo output carries a "RESEARCH USE ONLY
  — NOT A DIAGNOSIS" banner.
- **Out of scope this phase** — real patient data, hospital partnerships,
  ethics-committee-gated clinical validation, fMRI analysis. The
  regulatory path forward (ICMR guidelines, CDSCO SaMD classification,
  DPDP Act) is documented in the proposal as future work, not obligation.

## Reproducibility

- Fixed seeds across Python, NumPy, PyTorch, and CUDA; deterministic
  convolution algorithms where compatible.
- Docker image with pinned dependency versions; versioned configs and
  experiment logs (W&B / MLflow).
- Negative results are logged, not hidden — including a QUBO ablation
  that ties or loses.

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
- **References (BibTeX):** [docs/references.bib](docs/references.bib)
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
