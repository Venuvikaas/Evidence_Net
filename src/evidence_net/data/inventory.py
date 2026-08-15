"""Read-only inventory of a dataset directory.

Produces ``FileEntry`` records with relative path, extension, byte size,
sha256 hash, readable status, and — where readable — dimensions, channels,
data type, and numerical range. Never mutates the source directory.
"""

from __future__ import annotations

import os
from pathlib import Path

from evidence_net.data.loaders import inspect_file
from evidence_net.data.manifests import FileEntry
from evidence_net.reporting.run_bundle import hash_file

JUNK_DIR_NAMES = ("__MACOSX",)
HIDDEN_PREFIX = "."
SUPPORTED_EXTENSIONS = (".npy",)


def _is_junk_dir(name: str) -> bool:
    return name in JUNK_DIR_NAMES or name.startswith(HIDDEN_PREFIX)


def inventory_directory(root: Path) -> list[FileEntry]:
    """Inventory all supported files under ``root``, excluding junk.

    Files that fail to load are recorded as unreadable rather than skipped,
    so the audit can report them.
    """
    entries: list[FileEntry] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _is_junk_dir(d)]
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            relative = path.relative_to(root).as_posix()
            metadata = inspect_file(path)
            if metadata is None:
                entries.append(
                    FileEntry(
                        relative_path=relative,
                        extension=path.suffix,
                        byte_size=path.stat().st_size,
                        sha256=hash_file(path),
                        readable=False,
                    )
                )
                continue
            entries.append(
                FileEntry(
                    relative_path=relative,
                    extension=path.suffix,
                    byte_size=path.stat().st_size,
                    sha256=hash_file(path),
                    readable=True,
                    dimensions=metadata["dimensions"],
                    channels=metadata["channels"],
                    dtype=metadata["dtype"],
                    range=metadata["range"],
                )
            )
    entries.sort(key=lambda e: e.relative_path)
    return entries
