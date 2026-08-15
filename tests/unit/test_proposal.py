"""Proposal fusion identities and boundedness (Phase 4 box 5).

Verifies the Phase 4 contract: the proposal is amplitude-bounded
``|d| <= alpha``, gate 0 returns exactly the Base, gate 1 returns exactly
the ungated candidate, and the Base parameters stay frozen during proposal
training.
"""

from __future__ import annotations

import torch

from evidence_net.models.base import BaseReconstruction
from evidence_net.models.factory import build_model
from evidence_net.models.proposal import BoundedDetailProposal, DetailProposer, fuse
from evidence_net.training.config import ModelConfig


def _proposal(
    hidden_channels: int = 8, depth: int = 2, amplitude: float = 0.1
) -> BoundedDetailProposal:
    base = BaseReconstruction(hidden_channels=hidden_channels, depth=depth)
    proposer = DetailProposer(hidden_channels=hidden_channels, depth=depth, amplitude=amplitude)
    return BoundedDetailProposal(base, proposer)


def test_gate_zero_returns_base() -> None:
    model = _proposal()
    y = torch.rand(1, 1, 16, 16)
    b, d, _c = model.propose(y)
    gated = fuse(b, d, 0.0)
    assert torch.equal(gated, b)
    assert not torch.equal(gated, b + d)


def test_gate_one_returns_candidate() -> None:
    model = _proposal()
    y = torch.rand(1, 1, 16, 16)
    b, d, c = model.propose(y)
    gated = fuse(b, d, 1.0)
    assert torch.allclose(gated, c)
    assert torch.allclose(gated, b + d)


def test_forward_matches_ungated_candidate() -> None:
    model = _proposal()
    y = torch.rand(1, 1, 16, 16)
    assert torch.allclose(model(y), model.propose(y)[2])


def test_proposal_is_amplitude_bounded() -> None:
    model = _proposal(amplitude=0.05)
    y = torch.rand(1, 1, 16, 16)
    _b, d, _c = model.propose(y)
    assert d.abs().max().item() <= 0.05 + 1e-6


def test_factory_builds_proposal_with_amplitude() -> None:
    config = ModelConfig(name="proposal", hidden_channels=8, depth=2, amplitude=0.2)
    model = build_model(config)
    assert isinstance(model, BoundedDetailProposal)
    assert isinstance(model.proposer, DetailProposer)
    assert model.proposer.amplitude == 0.2


def test_base_is_frozen_during_proposal_training() -> None:
    model = _proposal()
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert any(parameter.requires_grad for parameter in model.proposer.parameters())
    # A backward pass must not touch base parameters.
    y = torch.rand(1, 1, 16, 16, requires_grad=True)
    _b, d, _c = model.propose(y)
    d.sum().backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert any(parameter.grad is not None for parameter in model.proposer.parameters())


def test_patch_gate_broadcast() -> None:
    model = _proposal()
    y = torch.rand(1, 1, 16, 16)
    b, d, c = model.propose(y)
    # 4x4 patch gate map expanded to the 32x32 pixel grid (nearest neighbor).
    patch_gate = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ).view(1, 1, 4, 4)
    gate = torch.nn.functional.interpolate(patch_gate, size=(32, 32), mode="nearest-exact")
    gated = fuse(b, d, gate)
    # Top-left quadrant matches base; bottom-right matches the candidate
    # (the model clamps the composed image, so compare against the clamp).
    assert torch.allclose(gated[0, 0, :16, :16], b[0, 0, :16, :16])
    assert torch.allclose(gated[0, 0, 16:, 16:], torch.clamp(b + d, 0.0, 1.0)[0, 0, 16:, 16:])


def test_proposal_module_state_dict_roundtrip() -> None:
    model = _proposal()
    payload = model.state_dict()
    rebuilt = _proposal()
    rebuilt.load_state_dict(payload)
    y = torch.rand(1, 1, 16, 16)
    with torch.no_grad():
        assert torch.allclose(model(y), rebuilt(y))
