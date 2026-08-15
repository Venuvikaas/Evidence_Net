"""Tests for the official train pairing adapter."""

from pathlib import Path

import numpy as np
import pytest

from evidence_net.data.pairing import (
    PairingError,
    audit_pairing,
    discover_train_structure,
)


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((8, 8), dtype=np.float32))


def _make_train(root: Path, n: int = 3) -> tuple[Path, Path]:
    gt = root / "train" / "GT"
    noisy = root / "train" / "NoisyLR"
    for i in range(n):
        _write(gt / f"{i:06d}.npy")
        _write(noisy / f"{i:06d}.npy")
    return gt, noisy


def test_discover_structure_ignores_junk(tmp_path: Path) -> None:
    root = tmp_path / "train"
    gt, noisy = _make_train(root)
    (root / "__MACOSX").mkdir()
    (root / ".hidden").mkdir()
    gt_found, noisy_found = discover_train_structure(root)
    assert gt_found == gt
    assert noisy_found == noisy


def test_discover_structure_requires_exactly_one_each(tmp_path: Path) -> None:
    root = tmp_path / "train"
    (root / "GT").mkdir(parents=True)
    (root / "NoisyLR").mkdir()
    (root / "extra" / "NoisyLR").mkdir(parents=True)

    with pytest.raises(PairingError, match="exactly one NoisyLR"):
        discover_train_structure(root)


def test_audit_pairing_clean(tmp_path: Path) -> None:
    gt, noisy = _make_train(tmp_path / "train")
    report = audit_pairing(gt, noisy)
    assert report.is_clean
    assert len(report.pairs) == 3


def test_audit_pairing_reports_missing_and_duplicates(tmp_path: Path) -> None:
    gt, noisy = _make_train(tmp_path / "train", n=3)
    # delete one noisy partner and duplicate another
    (noisy / "000001.npy").unlink()
    np.save(noisy / "000000_dup.npy", np.zeros((8, 8), dtype=np.float32))
    report = audit_pairing(gt, noisy)
    assert not report.is_clean
    assert "000001" in report.unmatched_gt
    assert "000000_dup" in report.unmatched_noisy
    assert len(report.pairs) == 2


def test_audit_pairing_duplicate_names(tmp_path: Path) -> None:
    gt, noisy = _make_train(tmp_path / "train", n=2)
    # add a second GT file with the same stem (via a differently-named copy)
    np.save(gt / "000000_alt.npy", np.zeros((8, 8), dtype=np.float32))
    report = audit_pairing(gt, noisy)
    # 000000_alt has no noisy partner -> unmatched, reported not silent
    assert "000000_alt" in report.unmatched_gt
