"""Tests for patient-level data splitting.

This is the single most important correctness test in the project.
Every non-inferiority claim (H1a, H1b) and every leakage-safety claim in
the proposal depends on the guarantee checked here. This test must never
be skipped, weakened, or marked xfail — see CONTRIBUTING.md.
"""

from __future__ import annotations

import pytest

from scanvidence.data.splitting import assert_no_leakage, patient_level_split


def _fake_cohort(n_patients: int = 20, images_per_patient: int = 5) -> list[dict]:
    return [
        {"patient_id": f"P{pid:03d}", "path": f"img_{pid}_{k}.png"}
        for pid in range(n_patients)
        for k in range(images_per_patient)
    ]


def test_no_patient_appears_in_more_than_one_partition():
    records = _fake_cohort()
    train, val, test = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=0)

    train_ids = {r["patient_id"] for r in train}
    val_ids = {r["patient_id"] for r in val}
    test_ids = {r["patient_id"] for r in test}

    assert train_ids.isdisjoint(val_ids), "Leakage: a patient appears in both train and val"
    assert train_ids.isdisjoint(test_ids), "Leakage: a patient appears in both train and test"
    assert val_ids.isdisjoint(test_ids), "Leakage: a patient appears in both val and test"


def test_assert_no_leakage_passes_on_a_clean_split():
    records = _fake_cohort()
    train, val, test = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=0)
    assert_no_leakage(train, val, test)  # should not raise


def test_assert_no_leakage_catches_a_deliberately_broken_split():
    records = _fake_cohort(n_patients=5, images_per_patient=2)
    # Deliberately leak P000 into both partitions.
    train = [r for r in records if r["patient_id"] in {"P000", "P001", "P002"}]
    test = [r for r in records if r["patient_id"] in {"P000", "P003", "P004"}]
    with pytest.raises(AssertionError, match="leakage"):
        assert_no_leakage(train, test)


def test_all_records_accounted_for():
    records = _fake_cohort()
    train, val, test = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=0)
    assert len(train) + len(val) + len(test) == len(records)


def test_split_is_deterministic_given_seed():
    records = _fake_cohort()
    a = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=42)
    b = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=42)
    assert [r["patient_id"] for r in a[0]] == [r["patient_id"] for r in b[0]]


def test_different_seeds_generally_differ():
    records = _fake_cohort(n_patients=50, images_per_patient=2)
    a = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=1)
    b = patient_level_split(records, ratios=(0.7, 0.15, 0.15), seed=2)
    assert {r["patient_id"] for r in a[0]} != {r["patient_id"] for r in b[0]}


@pytest.mark.parametrize("ratios", [(0.7, 0.15, 0.15), (0.8, 0.1, 0.1), (0.6, 0.2, 0.2)])
def test_split_respects_approximate_ratios(ratios):
    records = _fake_cohort(n_patients=100, images_per_patient=3)
    train, _, _ = patient_level_split(records, ratios=ratios, seed=0)
    train_patients = len({r["patient_id"] for r in train})
    assert abs(train_patients / 100 - ratios[0]) < 0.05


def test_rejects_ratios_that_dont_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        patient_level_split(_fake_cohort(), ratios=(0.7, 0.2, 0.2), seed=0)


def test_rejects_records_missing_patient_key():
    with pytest.raises(ValueError, match="missing"):
        patient_level_split([{"path": "no_id_here.png"}])
