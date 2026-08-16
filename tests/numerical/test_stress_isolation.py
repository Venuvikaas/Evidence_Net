"""Hidden stress isolation tests (Phase 10, structural-risk-v1 section 4).

The final stress definitions are frozen and hash-verified, and training code
never reads them. The suites must consume the frozen parameters, never
ad-hoc values.
"""

from __future__ import annotations

import pytest

from evidence_net.stress_tests.acquisition import build_acquisition_suite
from evidence_net.stress_tests.hidden_stress import (
    HiddenStressError,
    content_hash,
    load_hidden_stress,
    stress_params,
    training_isolation_clean,
)
from evidence_net.stress_tests.structural import (
    EdgeShift,
    FalsePeriodicity,
    build_candidate_suite,
)


def test_hidden_stress_definitions_are_frozen_and_hashed() -> None:
    data = load_hidden_stress()
    assert data["schema"] == "hidden-stress-v1"
    assert data["frozen"] is True
    assert content_hash(data) == data["hash"]


def test_tampered_definitions_are_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from evidence_net.stress_tests import hidden_stress

    data = load_hidden_stress()
    data["perturbation"]["edge_shift_px"] = 99  # tamper
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(hidden_stress, "HIDDEN_STRESS_PATH", path)
    with pytest.raises(HiddenStressError):
        load_hidden_stress()


def test_training_package_never_reads_stress_definitions() -> None:
    offenders = training_isolation_clean()
    assert offenders == [], f"training code references stress definitions: {offenders}"


def test_suites_consume_the_frozen_parameters() -> None:
    params = stress_params()
    perturbation = params["perturbation"]
    acquisition = params["acquisition"]
    # The suites built from the frozen file carry the frozen values.
    edge_shift = build_candidate_suite(names=("edge-shift",))[0]
    assert isinstance(edge_shift, EdgeShift)
    assert edge_shift.params["edge_shift_px"] == perturbation["edge_shift_px"]
    periodicity = build_candidate_suite(names=("false-periodicity",))[0]
    assert isinstance(periodicity, FalsePeriodicity)
    assert periodicity.params["period"] == perturbation["period"]
    noise = build_acquisition_suite(names=("sensor-noise",))[0]
    assert noise.params["noise_sigma"] == acquisition["noise_sigma"]
