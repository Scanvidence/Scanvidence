# Architecture

This mirrors the proposal's branched research architecture — each pipeline
stage is a subpackage, not a script, so every stage is independently
importable, testable, and ownable. Follows the
[pgmpy](https://github.com/pgmpy/pgmpy) structural pattern: base classes
in every subpackage, `__init__.py` re-exports, global config singleton.

```
Scan Input → Preprocess → Model → Calibrate → Explain → Evaluate
                                                  ↑
                                        Quantum Feature Selection
                                           (stretch objective)
```

## Subpackage Map

| Pipeline Stage | Package | Purpose |
|---|---|---|
| Global config | `scanvidence.global_vars` | Config singleton, logger, data home |
| Shared types | `scanvidence.base` | `ScanInput`, `DetectionResult`, `XAIResult` |
| Data handling | `scanvidence.data` | Loaders (NIfTI, DICOM), datasets (BraTS, ADNI), patient-level splitting |
| Preprocess | `scanvidence.preprocessing` | Skull stripping, normalization, registration, composable Pipeline |
| Models — classification | `scanvidence.models.classification` | `TumorClassifier`, `AlzheimersClassifier` |
| Models — segmentation | `scanvidence.models.segmentation` | `TumorSegmentor` (nnU-Net) |
| Models — backbone | `scanvidence.models.backbone` | Shared architectures (ResNet, ViT) |
| Radiomics | `scanvidence.radiomics` | PyRadiomics feature extraction |
| Calibration | `scanvidence.calibration` | Temperature scaling, MC-Dropout |
| Quantum (stretch) | `scanvidence.quantum` | QUBO feature selection |
| Explain | `scanvidence.xai` | Grad-CAM, SHAP, LIME with quantitative validation |
| Evaluate | `scanvidence.evaluation` | Metrics, bootstrap CIs, statistical tests |
| Tasks | `scanvidence.tasks` | End-to-end orchestrators per detection domain |
| API | `scanvidence.api` | FastAPI REST endpoints for clinical deployment |
| Demo | `scanvidence.demo` | Streamlit research demonstration |
| Utilities | `scanvidence.utils` | Device detection, checkpoints, helpers |

## Key Design Patterns

### 1. Base Class Hierarchy (pgmpy pattern)

Every subpackage has a `base.py` defining abstract base classes:

```python
from scanvidence.xai.base import BaseExplainer
from scanvidence.xai.GradCAM import GradCAM  # extends BaseExplainer
```

### 2. `__init__.py` Re-Exports

```python
# Clean imports via __init__.py
from scanvidence.models import TumorClassifier       # not .classification.TumorClassifier
from scanvidence.xai import GradCAM                   # not .xai.GradCAM.GradCAM
from scanvidence.tasks import BrainTumorTask
```

### 3. Task Orchestration

```python
from scanvidence.tasks import BrainTumorTask, AlzheimersTask

# Each task wires: data → preprocess → model → calibrate → explain
result = BrainTumorTask.from_config("configs/brain_tumor.yaml").run("scan.nii.gz")
result = AlzheimersTask.from_config("configs/alzheimers.yaml").run("scan.nii.gz")
```

### 4. Global Config Singleton

```python
import scanvidence
scanvidence.config.set_device("cuda:0")
scanvidence.config.DTYPE = "float16"
```

## The One Module Every Other Module Trusts

`scanvidence.data.splitting` is upstream of every result in the proposal.
H1a and H1b are non-inferiority claims measured on a *locked* test
partition; that only means anything if `patient_level_split` genuinely
never leaks a patient across partitions. Treat changes to this module with
the same weight as changes to the pre-specified decision rules themselves.

## Why a Package, Not Notebooks

Notebooks are fine for exploration — keep them in `notebooks/`, gitignored
outputs via `nbstripout`. But anything a metric, hypothesis test, or the
demo depends on belongs in `scanvidence/` as an importable, tested module.
The rule of thumb: if a number from it goes into the thesis, it isn't
allowed to live only in a notebook cell.
