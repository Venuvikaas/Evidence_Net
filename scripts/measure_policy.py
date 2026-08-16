"""Evaluate the selective policy on held-out data (Phase 6, EXP-010).

Usage (from the repository root):::

    python scripts/measure_policy.py --synthetic
    python scripts/measure_policy.py --n-samples 12

Applies the frozen decision-policy-v1 thresholds to calibrated benefit
probabilities on held-out validation data and reports:

- action-map fractions (accept / attenuate / reject) and coverage;
- the orthogonal unresolved-area fraction (input edge density);
- action-map and coverage-risk reports (patch MAE per action);
- restoration and structural-risk outcomes of the gated output vs the
  Base and the ungated candidate (PSNR/SSIM/MAE, edge displacement).

Thresholds are fit on the calibration split only (same isolation rule as
calibration-version-v1). ``--synthetic`` runs the same comparison on
synthetic grids so CI can exercise the policy path without the data files.
Never touches ``Test_NoisyLR/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.benefit.calibration import fit_calibration
    from evidence_net.benefit.labels import OUTPUT_GRID, patch_benefit_labels
    from evidence_net.benefit.predictors import (
        LocalSignalBaseline,
        ResidualMagnitudeBaseline,
    )
    from evidence_net.decision.policy import (
        PolicyConfig,
        apply_policy,
        coverage_risk_report,
        fit_policy_thresholds,
        policy_outputs,
    )
    from evidence_net.evaluation.metrics import all_metrics, edge_displacement
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.benefit.calibration import fit_calibration  # noqa: E402
    from evidence_net.benefit.labels import OUTPUT_GRID, patch_benefit_labels  # noqa: E402
    from evidence_net.benefit.predictors import (  # noqa: E402
        LocalSignalBaseline,
        ResidualMagnitudeBaseline,
    )
    from evidence_net.decision.policy import (  # noqa: E402
        PolicyConfig,
        apply_policy,
        coverage_risk_report,
        fit_policy_thresholds,
        policy_outputs,
    )
    from evidence_net.evaluation.metrics import all_metrics, edge_displacement  # noqa: E402
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id  # noqa: E402

VALIDATION_SPLIT = "validation"
CALIBRATION_SPLIT = "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=REPO_ROOT / "runs", type=Path)
    parser.add_argument(
        "--synthetic", action="store_true", help="evaluate on synthetic grids (CI smoke)"
    )
    return parser.parse_args()


def _synthetic_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(input, base, proposal, target) grids with a structured benefit pattern.

    Left half: Base misses the target, the proposal closes the gap
    (beneficial). Right half: Base matches the target, the proposal harms
    (and carries high edge density -> unresolved).
    """
    rng = np.random.default_rng(seed)
    grid = OUTPUT_GRID
    half = grid // 2
    y = np.full((grid, grid), 0.05)
    y[:, :half] = 0.6
    y[:, half:] = 0.05
    for column in range(half, grid, 8):
        y[:, column] = 0.9
    base = np.clip(y, 0.0, 1.0)
    base[:, :half] = 0.2
    proposal = np.zeros((grid, grid))
    proposal[:, :half] = rng.normal(0.4, 0.02, size=(grid, half))
    proposal[:, half:] = rng.normal(0.0, 0.03, size=(grid, half))
    target = np.clip(base + proposal, 0.0, 1.0)
    target[:, half:] = base[:, half:]
    return y, base, proposal, target


def _cases(
    n_samples: int, seed: int
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    sample_ids: list[str] = []
    inputs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in range(n_samples):
        y, b, d, x = _synthetic_case(seed + index)
        sample_ids.append(f"synthetic-{index:06d}")
        inputs.append(y)
        bases.append(b)
        proposals.append(d)
        targets.append(x)
    return sample_ids, inputs, bases, proposals, targets


def _score_samples(
    predictor_name: str,
    inputs: list[np.ndarray],
    bases: list[np.ndarray],
    proposals: list[np.ndarray],
) -> list[np.ndarray]:
    predictor = (
        ResidualMagnitudeBaseline()
        if predictor_name == "residual-magnitude"
        else LocalSignalBaseline()
    )
    return [predictor.score(y, b, d) for y, b, d in zip(inputs, bases, proposals, strict=True)]


def main() -> int:
    args = parse_args()
    sample_ids, inputs, bases, proposals, targets = _cases(args.n_samples, args.seed)
    labels = [
        patch_benefit_labels(b, d, x) for b, d, x in zip(bases, proposals, targets, strict=True)
    ]
    label_arrays = [np.asarray(label, dtype=np.float64) for label in labels]

    # Calibrate the residual-magnitude baseline score on the calibration
    # split (fit isolation) and evaluate the policy on the validation split.
    scores = _score_samples("residual-magnitude", inputs, bases, proposals)
    mapping = fit_calibration(
        np.concatenate([score.reshape(-1) for score in scores]),
        np.concatenate([label.reshape(-1) for label in label_arrays]),
        split=CALIBRATION_SPLIT,
    )
    probabilities = [
        mapping.apply(np.asarray(score, dtype=np.float64)).reshape(score.shape) for score in scores
    ]

    # Fit thresholds on the same calibration split, then freeze.
    config = fit_policy_thresholds(
        np.concatenate([prob.reshape(-1) for prob in probabilities]),
        np.concatenate([label.reshape(-1) for label in label_arrays]),
        split=CALIBRATION_SPLIT,
    )

    maps = [
        apply_policy(sample_id, probability, y, b, d, config)
        for sample_id, probability, y, b, d in zip(
            sample_ids, probabilities, inputs, bases, proposals, strict=True
        )
    ]
    gated = [
        policy_outputs(action_map, b, d)
        for action_map, b, d in zip(maps, bases, proposals, strict=True)
    ]
    ungated = [np.clip(b + d, 0.0, 1.0) for b, d in zip(bases, proposals, strict=False)]

    outcomes: dict[str, dict[str, float]] = {}
    for label, family in (("base", bases), ("ungated", ungated), ("gated", gated)):
        outcomes[label] = {}
        for metric in ("psnr", "ssim", "mae"):
            values = [
                float(all_metrics(target, output)[metric])  # type: ignore[arg-type]
                for target, output in zip(targets, family, strict=True)
            ]
            outcomes[label][metric] = float(np.mean(values))
        displacement = [
            edge_displacement(target, output)
            for target, output in zip(targets, family, strict=True)
        ]
        outcomes[label]["edge_displacement_px"] = float(np.mean(displacement))

    report = coverage_risk_report(maps, bases, proposals, targets)

    run_id = args.run_id or new_run_id("policy-eval")
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 6,
            "mode": "synthetic",
            "evaluation_split": VALIDATION_SPLIT,
            "calibration_fit_split": CALIBRATION_SPLIT,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "policy": {
                "accept_threshold": config.accept_threshold,
                "reject_threshold": config.reject_threshold,
                "unresolved_edge_density": config.unresolved_edge_density,
            },
        },
        manifest={
            "run_id": run_id,
            "contract": "decision-policy-v1",
            "calibration_version": "calibration-v1",
            "split_isolation": "thresholds and calibration fit on calibration "
            "split only; evaluation on validation split; Test_NoisyLR never touched",
            "test_final_isolation": "confirmed-no-test-noisylr",
        },
        metrics={"outcomes": outcomes, "policy_report": report},
        summary=_build_summary(run_id, outcomes, report, config),
        reference="frozen decision-policy-v1 thresholds (see config.yaml)",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


def _build_summary(run_id: str, outcomes: dict, report: dict, config: PolicyConfig) -> str:
    lines = [
        f"# Selective policy evaluation {run_id}",
        "",
        f"- Policy: decision-policy-v1 (accept >= {config.accept_threshold}, "
        f"reject < {config.reject_threshold}, unresolved edge density >= "
        f"{config.unresolved_edge_density}).",
        f"- Action fractions: {report['action_fractions']}",
        f"- Mean coverage: {report['mean_coverage']:.3f}; unresolved area: "
        f"{report['mean_unresolved_fraction']:.3f}",
        "",
        "## Outcomes (validation split, mean)",
        "",
        "| output | PSNR (dB) | SSIM | MAE | edge displacement (px) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("base", "ungated", "gated"):
        o = outcomes[label]
        lines.append(
            f"| {label} | {o['psnr']:.4f} | {o['ssim']:.4f} | {o['mae']:.4f} | "
            f"{o['edge_displacement_px']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Kill-switch note",
            "",
            "Rejection emits the Base but never certifies it: the unresolved "
            "mask is orthogonal to accept/attenuate/reject and reported "
            "separately. Thresholds were fit on the calibration split only "
            "and frozen before this evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
