"""Tests for the restoration comparison report module."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from evidence_net.inference.baseline import evaluate_restorer
from evidence_net.models.reference import deterministic_reconstruction
from evidence_net.reporting.comparison_report import (
    PANEL_ORDER,
    write_comparison_report,
    write_comparison_sheet,
    write_png,
)


def test_write_png_creates_valid_png(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    write_png(path, np.full((4, 4), 0.5))
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # The final chunk must be IEND (tag followed by its CRC).
    (length,) = struct.unpack(">I", data[-12:-8])
    assert data[-8:-4] == b"IEND"
    assert length == 0
    # IHDR width/height decode correctly (chunk tag starts at byte 12).
    assert data[12:16] == b"IHDR"
    width, height, depth, color = struct.unpack(">IIBB", data[16:26])
    assert (width, height, depth, color) == (4, 4, 8, 0)  # grayscale 8-bit


def test_write_png_idat_is_valid_zlib(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    write_png(path, np.linspace(0.0, 1.0, 16).reshape(4, 4))
    data = path.read_bytes()
    pos = 8
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if tag == b"IDAT":
            idat += chunk
        pos += 12 + length
    decompressed = zlib.decompress(idat)
    assert len(decompressed) == 4 * (1 + 4)  # 4 filter bytes + 4 rows of 4 px


def test_write_png_rejects_3d(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_png(tmp_path / "x.png", np.zeros((2, 2, 2)))


def test_write_comparison_sheet_writes_png_and_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    input_ = rng.random((8, 8))
    target = rng.random((8, 8))
    prediction = np.clip(input_ * 0.5 + 0.25, 0, 1)
    sheet = write_comparison_sheet(
        tmp_path,
        0,
        input_,
        prediction,
        target,
        {"psnr": 12.3, "ssim": 0.9},
    )
    assert sheet.name == "comparison-000000.png"
    assert sheet.is_file()
    metrics = json.loads((tmp_path / "comparison-000000.json").read_text())
    assert metrics == {"psnr": 12.3, "ssim": 0.9}
    # Montage is 5 panels wide at 8 px each.
    assert sheet.stat().st_size > 0


def test_write_comparison_report_tabulates_aggregates(tmp_path: Path) -> None:
    inputs = [np.full((8, 8), 0.0), np.full((8, 8), 0.5)]
    targets = [np.full((16, 16), 0.0), np.full((16, 16), 0.5)]
    result = evaluate_restorer(
        "deterministic",
        inputs,
        targets,
        ["000000", "000001"],
        deterministic_reconstruction,
        n_boot=50,
        seed=0,
    )
    report = write_comparison_report(
        tmp_path,
        {"deterministic": result},
        sample_ids=["000000", "000001"],
        n_samples=2,
        split_label="validation",
    )
    text = report.read_text()
    assert "validation" in text
    assert "deterministic" in text
    assert "PSNR" in text
    assert "source group" in text
    assert "input, output, target, error, edges" in text
    assert all(name in text for name in PANEL_ORDER)
