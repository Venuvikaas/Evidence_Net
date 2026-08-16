"""Data retention and storage discipline controls (Phase 16).

Purges temporary uploads, unpromoted artifacts, and stale scratch files older
than specified retention thresholds.
"""

from __future__ import annotations

import time
from pathlib import Path


def cleanup_scratch_directory(
    scratch_dir: Path | str = Path("scratch"), max_age_seconds: float = 86400.0
) -> int:
    """Purge temporary files in scratch directory older than max_age_seconds (default 24h).

    Returns number of files deleted.
    """
    path = Path(scratch_dir)
    if not path.is_dir():
        return 0

    now = time.time()
    deleted_count = 0

    for item in path.rglob("*"):
        if item.is_file():
            # Check modification time
            try:
                mtime = item.stat().st_mtime
                if now - mtime > max_age_seconds:
                    item.unlink()
                    deleted_count += 1
            except OSError:
                continue

    return deleted_count
