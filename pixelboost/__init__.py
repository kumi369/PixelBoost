"""PixelBoost package."""

from __future__ import annotations

import site
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PACKAGES_DIR = PROJECT_ROOT / ".python_packages"

if LOCAL_PACKAGES_DIR.exists():
    site.addsitedir(str(LOCAL_PACKAGES_DIR))

try:
    site.addsitedir(site.getusersitepackages())
except Exception:
    pass
