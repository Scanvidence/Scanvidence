"""CLI entry point: ``python -m scanvidence``.

Thin wrapper so the Docker image's default command works and so a scan
can be run end-to-end from the terminal:

    python -m scanvidence detect --task brain_tumor \\
        --config configs/brain_tumor.yaml --scan patient_001.nii.gz
"""

from __future__ import annotations

import argparse

from scanvidence import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanvidence",
        description="Explainable, uncertainty-aware medical imaging detection.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser("detect", help="Run a detection task on a scan.")
    detect.add_argument(
        "--task",
        required=True,
        choices=["brain_tumor", "alzheimers"],
        help="Which detection task to run.",
    )
    detect.add_argument("--config", required=True, help="Path to the task YAML config.")
    detect.add_argument("--scan", required=True, help="Path to the input scan.")
    return parser


def main() -> None:
    """Run the scanvidence CLI."""
    args = _build_parser().parse_args()

    if args.command == "detect":
        from scanvidence.tasks import AlzheimersTask, BrainTumorTask

        task_cls = BrainTumorTask if args.task == "brain_tumor" else AlzheimersTask
        result = task_cls.from_config(args.config).run(args.scan)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
