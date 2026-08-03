"""Device detection and management utilities."""


def get_device(prefer: str | None = None) -> str:
    """Detect the best available compute device.

    Parameters
    ----------
    prefer : str or None
        Preferred device (``"cuda"``, ``"mps"``, ``"cpu"``). If None,
        auto-detects in order: CUDA → MPS → CPU.

    Returns
    -------
    device : str
        Device string suitable for ``torch.device()``.
    """
    if prefer is not None:
        return prefer

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
