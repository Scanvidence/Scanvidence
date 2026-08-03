"""Patient-level dataset splitting.

Every evaluation claim in the proposal — H1a, H1b, and the locked-test-set
protocol described in the Evaluation Plan — depends on one guarantee: no
patient's records ever appear in more than one partition. This module is
the single place that guarantee is implemented. If the splitting strategy
changes, keep the guarantee and keep tests/test_data/test_splitting.py
green; that test is not allowed to be skipped, weakened, or marked xfail.
"""

from __future__ import annotations

import random
from collections import defaultdict


def patient_level_split(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    patient_key: str = "patient_id",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test partitions, grouped by patient.

    All records sharing a patient ID are assigned to the same partition,
    so no patient can appear on both sides of an evaluation.

    Parameters
    ----------
    records : list of dict
        Each record must contain `patient_key`.
    ratios : tuple of float
        (train, val, test) proportions. Must sum to 1.0.
    seed : int
        Controls the patient-level shuffle. Same seed -> same split,
        which is what makes a reported result reproducible.
    patient_key : str
        Dictionary key identifying the patient/case ID.

    Returns
    -------
    (train, val, test) : tuple of list of dict

    Raises
    ------
    ValueError
        If `ratios` does not sum to 1.0, or a record is missing
        `patient_key`.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")

    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if patient_key not in r:
            raise ValueError(f"record missing '{patient_key}': {r}")
        by_patient[r[patient_key]].append(r)

    # Sort first so the shuffle below is deterministic across platforms,
    # then shuffle with the given seed.
    patient_ids = sorted(by_patient.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    train = [r for pid in train_ids for r in by_patient[pid]]
    val = [r for pid in val_ids for r in by_patient[pid]]
    test = [r for pid in test_ids for r in by_patient[pid]]
    return train, val, test


def assert_no_leakage(*partitions: list[dict], patient_key: str = "patient_id") -> None:
    """Raise AssertionError if any patient appears in more than one partition.

    Intended as a runtime guard callable right before training starts —
    not just a test-suite check — so a leaking split fails loudly instead
    of silently inflating a reported metric.
    """
    id_sets = [{r[patient_key] for r in part} for part in partitions]
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            overlap = id_sets[i] & id_sets[j]
            if overlap:
                raise AssertionError(
                    f"Patient-level leakage detected between partitions "
                    f"{i} and {j}: {sorted(overlap)[:5]}"
                    f"{'...' if len(overlap) > 5 else ''}"
                )
