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


def test_train_accepts_resume_flag():
    from scanvidence.training.cli import _build_parser

    args = _build_parser().parse_args(["train", "--data-root", "/tmp", "--resume", "run.pt"])
    assert args.resume == "run.pt"
    assert args.command == "train"


def test_train_accepts_resume_flag_without_path():
    from scanvidence.training.cli import _build_parser

    args = _build_parser().parse_args(["train", "--data-root", "/tmp", "--resume"])
    assert args.resume == "AUTO"
    assert args.command == "train"


def test_run_train_auto_resume_uses_out_dir_checkpoint(monkeypatch, tmp_path):
    import argparse
    from pathlib import Path

    import torch

    import scanvidence.training.cli as cli

    class _DummyOptimizer:
        def load_state_dict(self, _state):
            return None

    class _DummyModel:
        parameter_count = 1

        def load_state_dict(self, _state):
            return None

    class _DummyHistory:
        best_val_dice = 0.5
        best_epoch = 1
        step_time_s = [0.01]

    class _DummyTrainer:
        def __init__(self):
            self.optimizer = _DummyOptimizer()
            self.scaler = None
            self.device = torch.device("cpu")

        def fit(self, **_kwargs):
            return _DummyHistory()

    data_root = tmp_path / "data"
    data_root.mkdir()
    out_dir = tmp_path / "runs" / "b0"
    record = {
        "patient_id": "p1",
        "available_sequences": ["t1n", "t1c", "t2w", "t2f"],
        "seg_path": "seg.nii.gz",
    }

    monkeypatch.setattr(cli.BraTSDataset, "discover", lambda self: [record])
    monkeypatch.setattr(cli, "_usable_records", lambda records: (records, 0))
    monkeypatch.setattr(cli, "_run_common", lambda _args, _usable: ({"split": {}}, [record], []))
    monkeypatch.setattr(cli, "_set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "_make_loader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cli, "_hardware_info", lambda: {})
    monkeypatch.setattr(cli, "set_rng_states", lambda _state: None)
    monkeypatch.setattr(
        cli, "_build_model_and_trainer", lambda *_args, **_kwargs: (_DummyModel(), _DummyTrainer())
    )

    loaded_paths: list[str] = []

    def _fake_torch_load(path, *args, **kwargs):
        loaded_paths.append(str(path))
        return {"state_dict": {}, "optimizer": {}, "epoch": 10, "rng": None}

    monkeypatch.setattr(cli.torch, "load", _fake_torch_load)

    args = argparse.Namespace(
        command="train",
        data_root=str(data_root),
        track="GLI",
        patch=96,
        accum=8,
        amp=True,
        max_grad_norm=1.0,
        lr=1e-4,
        weight_decay=1e-5,
        seed=17,
        workers=0,
        out_dir=str(out_dir),
        val_frac=0.2,
        widths="16,32,64,128",
        num_classes=4,
        dropout=0.0,
        foreground_prob=0.5,
        augment=True,
        log_every=0,
        profile_steps=0,
        remap_legacy_four=False,
        deterministic=True,
        max_mem_fraction=0.0,
        epochs=2,
        max_cases=0,
        resume="AUTO",
    )

    code = cli._run(args)
    assert code == 0
    assert loaded_paths == [str(Path(out_dir) / "best.pt")]


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
