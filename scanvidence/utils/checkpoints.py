"""Checkpoint save/load utilities."""

from pathlib import Path


def save_checkpoint(state: dict, path: str, filename: str = "checkpoint.pt") -> str:
    """Save a training checkpoint.

    Parameters
    ----------
    state : dict
        Checkpoint state (model weights, optimizer state, epoch, etc.).
    path : str
        Directory to save the checkpoint in.
    filename : str
        Checkpoint filename.

    Returns
    -------
    filepath : str
        Full path to the saved checkpoint.
    """
    import torch

    Path(path).mkdir(parents=True, exist_ok=True)
    filepath = str(Path(path) / filename)
    torch.save(state, filepath)
    return filepath


def load_checkpoint(filepath: str) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    filepath : str
        Path to the checkpoint file.

    Returns
    -------
    state : dict
        Checkpoint state.
    """
    import torch

    return torch.load(filepath, map_location="cpu", weights_only=False)
