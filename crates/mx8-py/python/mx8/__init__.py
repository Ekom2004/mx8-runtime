import os
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_coordinator = _pkg_dir / "mx8d-coordinator"
if not _coordinator.is_file():
    _coordinator = _pkg_dir / "mx8d-coordinator.exe"
if _coordinator.is_file():
    os.environ.setdefault("MX8_COORDINATOR_BIN", str(_coordinator))

from .mx8 import *  # noqa: F401,F403

__doc__ = mx8.__doc__
if hasattr(mx8, "__all__"):
    __all__ = mx8.__all__
