"""Make the repository root importable for the test suite.

Several tests import top-level packages that live outside ``src/`` (e.g.
``deploy.export_onnx``). ``python -m pytest`` happens to work because the
invocation puts the current directory on ``sys.path``, but plain ``pytest``
(as CI runs it) does not. This conftest guarantees the root is importable
under both invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
