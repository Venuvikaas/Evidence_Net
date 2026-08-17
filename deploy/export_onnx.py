"""ONNX Export Pipeline for EVIDENCE-Net promoted models (Phase 15).

Exports the **promoted** Base Reconstruction and Detail Proposal checkpoints
(``checkpoints/train-base-gate2/best.pt`` and
``checkpoints/train-proposal-gate3v2/best.pt``) to ONNX, on the frozen
128x128 -> 256x256 grid, with dynamic batch/spatial axes.

Integrity contract (same rule as the rest of the project):

- The exported graphs must load in ONNX Runtime and reproduce the PyTorch
  outputs within tolerance (verified by the parity test in
  ``tests/decision_parity/test_onnx_parity.py``).
- Export **fails loudly** rather than writing a placeholder asset: a fake
  ONNX file would silently ship a non-functional model.
- Calibration parity is recorded as ``not-defined``: the promoted pipeline
  records a calibration version but this service does not re-fit it, so
  there is no calibration artifact to compare at inference time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.factory import build_model
from evidence_net.models.proposal import BoundedDetailProposal, DetailProposer
from evidence_net.training.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

# Promoted frozen checkpoints (same paths the review API loads).
BASE_CHECKPOINT = REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt"
PROPOSAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt"

# Frozen contract grid: degraded input 128x128 -> output 256x256.
INPUT_GRID = 128
OUTPUT_GRID = 256


def load_checkpoint(path: Path, expected_type: type) -> Any:
    """Load a promoted checkpoint through the model factory (16-channel arch).

    The promoted proposal checkpoint is a ``BoundedDetailProposal`` (frozen
    Base + proposer); the caller may request the wrapped ``DetailProposer``.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Promoted checkpoint not found: {path}. Run the release pipeline first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ModelConfig(
        name=payload["config"]["model"]["name"],
        hidden_channels=payload["config"]["model"]["hidden_channels"],
        depth=payload["config"]["model"]["depth"],
        amplitude=payload["config"]["model"].get("amplitude", 0.1),
    )
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    # The promoted proposal artifact wraps the proposer (frozen Base inside);
    # unwrap it so we export the bounded-proposal head itself.
    if expected_type is DetailProposer and isinstance(model, BoundedDetailProposal):
        proposer = model.proposer
        if not isinstance(proposer, DetailProposer):
            raise TypeError("promoted proposal checkpoint has no DetailProposer")
        return proposer
    if not isinstance(model, expected_type):
        raise TypeError(
            f"checkpoint {path} loaded as {type(model).__name__}, expected {expected_type.__name__}"
        )
    return model


def export_base_model(
    output_path: Path | str,
    checkpoint_path: Path | str | None = None,
    opset_version: int = 14,
) -> Path:
    """Export the promoted BaseReconstruction to ONNX (128x128 -> 256x256)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = BASE_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    model = load_checkpoint(ckpt, BaseReconstruction)

    dummy_input = torch.randn(1, 1, INPUT_GRID, INPUT_GRID, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,  # legacy exporter: no onnxscript dependency
            input_names=["input"],
            output_names=["base_reconstruction"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "base_reconstruction": {0: "batch_size", 2: "out_height", 3: "out_width"},
            },
        )
    except Exception as exc:  # pragma: no cover - fail loudly, never fake
        raise RuntimeError(f"ONNX export failed for Base model: {exc}") from exc

    # The file must be a real ONNX graph, not a placeholder.
    import onnx

    onnx.load(str(out_path))
    print(f"Exported promoted Base model to {out_path}")
    return out_path


def export_proposal_model(
    output_path: Path | str,
    checkpoint_path: Path | str | None = None,
    opset_version: int = 14,
) -> Path:
    """Export the promoted DetailProposer to ONNX ((y, b) -> d on 256x256)."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = PROPOSAL_CHECKPOINT if checkpoint_path is None else Path(checkpoint_path)
    model = load_checkpoint(ckpt, DetailProposer)

    dummy_y = torch.randn(1, 1, INPUT_GRID, INPUT_GRID, dtype=torch.float32)
    dummy_b = torch.randn(1, 1, OUTPUT_GRID, OUTPUT_GRID, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            (dummy_y, dummy_b),
            str(out_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,  # legacy exporter: no onnxscript dependency
            input_names=["y", "b"],
            output_names=["detail_proposal"],
            dynamic_axes={
                "y": {0: "batch_size", 2: "in_height", 3: "in_width"},
                "b": {0: "batch_size", 2: "out_height", 3: "out_width"},
                "detail_proposal": {0: "batch_size", 2: "out_height", 3: "out_width"},
            },
        )
    except Exception as exc:  # pragma: no cover - fail loudly, never fake
        raise RuntimeError(f"ONNX export failed for Proposal model: {exc}") from exc

    import onnx

    onnx.load(str(out_path))
    print(f"Exported promoted Proposal model to {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the promoted EVIDENCE-Net models to ONNX (Phase 15)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="deploy/onnx", help="Directory for ONNX files"
    )
    parser.add_argument("--base-ckpt", type=str, default=None, help="Base checkpoint path")
    parser.add_argument("--proposal-ckpt", type=str, default=None, help="Proposal checkpoint path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    export_base_model(out_dir / "base.onnx", checkpoint_path=args.base_ckpt)
    export_proposal_model(out_dir / "proposal.onnx", checkpoint_path=args.proposal_ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
