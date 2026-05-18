#!/usr/bin/env python
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_OPENCLIP = Path(__file__).resolve().parents[1] / "utils" / "openclip_encoder.py"
_SPEC = importlib.util.spec_from_file_location("_shared_openclip_encoder", _REPO_OPENCLIP)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load shared OpenCLIP encoder from {_REPO_OPENCLIP}")

_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

OpenCLIPNetwork = _MODULE.OpenCLIPNetwork
OpenCLIPNetworkConfig = _MODULE.OpenCLIPNetworkConfig

__all__ = ["OpenCLIPNetwork", "OpenCLIPNetworkConfig"]
