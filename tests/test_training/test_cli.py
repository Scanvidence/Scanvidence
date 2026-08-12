"""CLI tests: gating ladder on synthetic volumes (CPU, no GPU needed)."""

import json

import pytest

from scanvidence.training.cli import main


def test_missing_data_root_returns_two():
    assert main(["overfit", "--data-root", "/nonexistent/path"]) == 2


def test_empty_data_root_reports_exit_code_two(monkeypatch):
    monkeypatch.delenv("BRATS_DATA_ROOT", raising=False)
    assert main(["train", "--data-root", ""]) == 2


def test_unknown_subcommand_is_an_argparse_error():
    with pytest.raises(SystemExit) as exc:
        main(["prognosticate"])
    assert exc.value.code == 2


@pytest.mark.slow
def test_overfit_gate_passes_and_writes_run_json(synthetic_brats, tmp_path):
    root, _ = synthetic_brats
    out = tmp_path / "runs" / "b0"
    code = main(
        [
            "overfit",
            "--data-root",
            root,
            "--out-dir",
            str(out),
            "--patch",
            "32",
            "--widths",
            "4,8,16,32",
            "--epochs",
            "30",
            "--lr",
            "1e-3",
            "--accum",
            "1",
            "--overfit-dice",
            "0.5",
            "--workers",
            "0",
            "--log-every",
            "0",
        ]
    )
    assert code == 0
    run = json.loads((out / "run.json").read_text())
    assert run["config"]["run_meta"]["split"]["train_cases"] == 2
    assert len(run["config"]["run_meta"]["split"]["train_hash"]) == 64
    assert run["config"]["run_meta"]["architecture"]["parameter_count"] > 0
    assert (out / "best.pt").exists()


@pytest.mark.slow
def test_overfit_gate_fails_when_threshold_unreachable(synthetic_brats, tmp_path):
    root, _ = synthetic_brats
    code = main(
        [
            "overfit",
            "--data-root",
            root,
            "--out-dir",
            str(tmp_path / "runs" / "b0f"),
            "--patch",
            "16",
            "--widths",
            "2,4,8,16",
            "--epochs",
            "1",
            "--overfit-dice",
            "0.99",
            "--workers",
            "0",
            "--log-every",
            "0",
        ]
    )
    assert code == 1


def test_run_common_splits_patient_level_without_leakage(synthetic_brats):
    from scanvidence.data.datasets import BraTSDataset
    from scanvidence.training.cli import _partition_hashes, _run_common

    root, ids = synthetic_brats
    import argparse

    args = argparse.Namespace(
        val_frac=0.2, seed=17, data_root=root, track="GLI", command="pilot", max_cases=2
    )
    records = BraTSDataset(root, track="GLI").discover()
    meta, train, val = _run_common(args, records)
    assert len(train) + len(val) == len(records)
    assert {r["patient_id"] for r in train}.isdisjoint(r["patient_id"] for r in val)
    assert len(meta["split"]["train_hash"]) == 64
    assert meta["split"]["train_hash"] == _partition_hashes(train)
    assert meta["split"]["commit_me_when_frozen"] is True
