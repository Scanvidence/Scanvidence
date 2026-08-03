"""Dataset loaders and the patient-level splitting logic every
evaluation claim depends on.

See :func:`patient_level_split` — the most safety-critical function
in the project.
"""

from .splitting import assert_no_leakage, patient_level_split
