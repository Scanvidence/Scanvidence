"""Global configuration and logger for the scanvidence package.

Follows the pgmpy pattern: a Config singleton and a NullHandler logger
exposed at the package root, so users can do:

    import scanvidence
    scanvidence.config.DEVICE = "cuda:0"
    scanvidence.logger.setLevel(logging.DEBUG)
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger("scanvidence")
logger.addHandler(logging.NullHandler())

SCANVIDENCE_DATA_HOME = os.path.join(Path.home(), ".scanvidence")


class Config:
    """Runtime configuration for the scanvidence package.

    Attributes
    ----------
    BACKEND : str
        Computation backend. Currently only ``"torch"``.
    DTYPE : str
        Default floating-point dtype for tensors.
    DEVICE : str or None
        Torch device string. Set via :meth:`set_device`.
    SHOW_PROGRESS : bool
        Whether to display tqdm progress bars.
    """

    def __init__(self):
        self.BACKEND = "torch"
        self.DTYPE = "float32"
        self.DEVICE = None
        self.SHOW_PROGRESS = True

    def set_device(self, device=None):
        """Set the torch device for computation.

        Parameters
        ----------
        device : str or None
            ``"cuda"`` / ``"cuda:0"`` / ``"cpu"`` / ``"mps"``.
            If ``None``, auto-detects CUDA → MPS → CPU.
        """
        import torch

        if device is None:
            if torch.cuda.is_available():
                self.DEVICE = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.DEVICE = torch.device("mps")
            else:
                self.DEVICE = torch.device("cpu")
        else:
            self.DEVICE = torch.device(device)

    def __repr__(self):
        return (
            f"Config(BACKEND={self.BACKEND!r}, DTYPE={self.DTYPE!r}, "
            f"DEVICE={self.DEVICE!r}, SHOW_PROGRESS={self.SHOW_PROGRESS!r})"
        )


config = Config()
