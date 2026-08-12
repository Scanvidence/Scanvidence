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

The **unified architecture** serves both detection domains — brain tumor
and Alzheimer's disease — with the same composable pipeline: parallel
classification, segmentation, and radiomics branches share one
preprocessing front end, and the QUBO branch is optional and never gates
the core pipeline.

## Subpackage Map

| Pipeline Stage | Package | Purpose |
|---|---|---|
| Global config | `scanvidence.global_vars` | Config singleton, logger, data home |
| Shared types | `scanvidence.base` | `ScanInput`, `DetectionResult`, `XAIResult` |
| Data handling | `scanvidence.data` | Loaders (NIfTI, DICOM), datasets (BraTS, OASIS, ADNI, TCGA, Figshare), patient-level splitting |
| Preprocess | `scanvidence.preprocessing` | Skull stripping, normalization, registration, composable Pipeline |
| Models — classification | `scanvidence.models.classification` | `TumorClassifier`, `AlzheimersClassifier` |
| Models — segmentation | `scanvidence.models.segmentation` | `TumorSegmentor` (nnU-Net) |
| Models — backbone | `scanvidence.models.backbone` | Shared architectures (ResNet, ViT); `SegResNetB0` reference segmentation CNN |
| Models — training | `scanvidence.training` | `BraTSPatchDataset` (96-cubed patching), Dice+CE loss, ET/TC/WT Dice, `SegmentationTrainer`, training CLI |
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
from scanvidence.models import TumorClassifier  # not .classification.TumorClassifier
from scanvidence.xai import GradCAM  # not .xai.GradCAM.GradCAM
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

## Subpackage → Hypothesis Map

| Subpackage | Proposal hypothesis / claim |
|---|---|
| `scanvidence.data.splitting` | Leakage-safe 70/15/15 locked partitions for H1a, H1b |
| `scanvidence.models.classification` | H1a — calibrated vs. uncalibrated classification (ΔAUC ≥ −0.02) |
| `scanvidence.models.segmentation` | H1b — nnU-Net baseline (ΔDice ≥ 0.02); SwinUNETR optional comparator |
| `scanvidence.xai` | H2a — spatial validity (IoU, pointing game, FFR vs. shuffled); H2b — SHAP ablation faithfulness |
| `scanvidence.calibration` | H1a prerequisite — temperature scaling / Platt / isotonic on the calibration subset; ECE, Brier |
| `scanvidence.quantum` | H3 — QUBO vs. classical selectors; stability via mean pairwise Jaccard |
| `scanvidence.radiomics` | H2b, H3 — fixed radiomics feature budget (≤ 50 candidates) |
| `scanvidence.evaluation` | All — DeLong, McNemar, Wilcoxon, 2,000-resample bootstrap CIs, Holm correction |

## Evaluation Protocol (as Pre-specified in the Proposal)

- Stratified **patient-level** train/validation/test partitions (70/15/15
  where cohort size permits); no patient in more than one partition.
- 5-fold stratified **group** cross-validation *inside training only* for
  model selection; a separate calibration subset within validation.
- Locked test set used only for final reporting: ROC-AUC with paired
  DeLong tests, per-patient errors with paired McNemar tests, Dice /
  Hausdorff with paired Wilcoxon signed-rank tests, primary-metric CIs
  from 2,000 stratified patient-bootstrap resamples.
- Holm-adjusted _p_-values at two-sided α = 0.05 within each endpoint
  family; the one-sided non-inferiority tests (H1a, H1b) are the
  exception.
- Prospective power check on eligible patient counts; if underpowered,
  report achieved CI width and "non-inferiority not demonstrated" —
  decision rules are never revised after observing results.

## Why a Package, Not Notebooks

Notebooks are fine for exploration — keep them in `notebooks/`, gitignored
outputs via `nbstripout`. But anything a metric, hypothesis test, or the
demo depends on belongs in `scanvidence/` as an importable, tested module.
The rule of thumb: if a number from it goes into the thesis, it isn't
allowed to live only in a notebook cell.

## References

The full bibliography for every method and benchmark used in the pipeline
— BraTS/ADNI, preprocessing, segmentation, classification, radiomics,
calibration, XAI, quantum feature selection, and statistical evaluation —
lives in [`docs/references.bib`](references.bib), in BibTeX format. Key
conventions follow the pgmpy project: readable `<author>_<year>` keys and
a `keyword` field that groups entries by pipeline stage. Cite these when
writing up results; keep new citations here, not only in thesis drafts.
