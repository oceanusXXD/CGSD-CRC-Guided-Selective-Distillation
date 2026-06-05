"""Select uncertain rows with guide-label targets and proxy-label strata."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.selection_common import main


if __name__ == "__main__":
    main(method="pcss", description=__doc__ or "")
