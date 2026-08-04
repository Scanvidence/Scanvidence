"""Tests for the scanvidence CLI entry point."""

import pytest

from scanvidence.__main__ import _build_parser, main


def test_version_flag_prints_version(capsys):
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--version"])
    out = capsys.readouterr().out
    assert "scanvidence" in out


def test_detect_subcommand_runs_end_to_end(tmp_path, monkeypatch, capsys):
    config = tmp_path / "task.yaml"
    config.write_text("model:\n  name: resnet50\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            "scanvidence",
            "detect",
            "--task",
            "brain_tumor",
            "--config",
            str(config),
            "--scan",
            "patient_001.nii.gz",
        ],
    )
    main()
    out = capsys.readouterr().out
    assert '"prediction": "unknown"' in out
    assert '"confidence": 0.0' in out
