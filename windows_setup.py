"""Configure PATH so the ThorLabs camera SDK DLLs can be located.

This module adds the local ``SDK`` and ``dlls`` directories (if present) to the
process PATH. Importing and calling :func:`configure_path` before importing the
``thorlabs_tsi_sdk`` package ensures that the required native libraries are
available.
"""

from __future__ import annotations

import os
from pathlib import Path


def _add_dir(path: Path) -> None:
    """Helper to prepend *path* to ``PATH`` and register it for DLL loading."""

    path_str = str(path)
    os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:  # pragma: no cover - only executes on Windows
            os.add_dll_directory(path_str)
        except OSError:
            pass


def configure_path() -> None:
    """Add SDK and DLL directories (recursively) to ``PATH`` for Windows.

    The ThorLabs SDK expects ``thorlabs_tsi_camera_sdk.dll`` to be discoverable.
    When running from the source tree, the DLLs may live in ``dlls`` and ``SDK``
    folders next to this file. This function searches those folders and all
    subdirectories, prepending each to ``PATH`` so the SDK can locate its native
    dependencies.
    """

    base = Path(__file__).resolve().parent
    for root in (base / "SDK", base / "dlls"):
        if not root.exists():
            continue
        # Include the root directory and all subfolders
        _add_dir(root)
        for sub in root.rglob("*"):
            if sub.is_dir():
                _add_dir(sub)

