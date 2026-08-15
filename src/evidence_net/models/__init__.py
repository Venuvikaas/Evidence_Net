"""Restoration, proposal, benefit-predictor, and diagnostic models (Phases 3-9)."""

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.direct import DirectRestoration
from evidence_net.models.factory import build_model, model_summary
from evidence_net.models.validate import (
    check_gradients_flow,
    check_output_contract,
    check_tiled_parity,
    save_load_roundtrip,
    tiled_inference,
)

__all__ = [
    "BaseReconstruction",
    "DirectRestoration",
    "build_model",
    "check_gradients_flow",
    "check_output_contract",
    "check_tiled_parity",
    "model_summary",
    "save_load_roundtrip",
    "tiled_inference",
]
