from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BUNDLE_ROOT.parent
PERFORMANCE_ROOT = BUNDLE_ROOT / "performance"
NEGOTIATION_ROOT = BUNDLE_ROOT / "negotiation"

for path in (REPO_ROOT, BUNDLE_ROOT, PERFORMANCE_ROOT, NEGOTIATION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

