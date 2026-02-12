"""Basic import tests for LanPaint.

The ComfyUI runtime dependencies (e.g. `comfy`) are intentionally optional for unit tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _import_custom_node_package() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    init_py = repo_root / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        "LanPaint",
        init_py,
        submodule_search_locations=[str(repo_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["LanPaint"] = module
    spec.loader.exec_module(module)
    return module


def test_package_imports_without_comfy() -> None:
    LanPaint = _import_custom_node_package()

    assert isinstance(LanPaint.NODE_CLASS_MAPPINGS, dict)
    assert isinstance(LanPaint.NODE_DISPLAY_NAME_MAPPINGS, dict)
    assert "LanPaint_KSampler" in LanPaint.NODE_CLASS_MAPPINGS
    assert LanPaint.WEB_DIRECTORY == "./web"
