"""Pytest hooks for e2e — ensures repo root is importable."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Optional vendored deps (e.g. chaincraft) under lib/pythonX.Y/site-packages
_py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
_lib_site = _REPO_ROOT / "lib" / _py_tag / "site-packages"
if _lib_site.is_dir() and str(_lib_site) not in sys.path:
    sys.path.insert(1, str(_lib_site))
