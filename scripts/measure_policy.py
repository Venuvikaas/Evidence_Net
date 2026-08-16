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
    from evidence_net.benefit.labels import OUTPUT_GRID, PATCH_SIZE, patch_benefit_labels
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
    from evidence_net.benefit.labels import (  # noqa: E402
        OUTPUT_GRID,
        PATCH_SIZE,
        patch_benefit_labels,
    )
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
    n_samples: int, seed: int, *, synthetic: bool, split: str = VALIDATION_SPLIT
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if synthetic:
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
    return _real_cases(n_samples, seed, split=split)


def _real_cases(
    n_samples: int, seed: int, *, split: str
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Real cases from the frozen Base/Proposal checkpoints on one split."""
    import torch

    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.models.factory import build_model
    from evidence_net.models.proposal import BoundedDetailProposal
    from evidence_net.training.config import ModelConfig
    from evidence_net.training.dataset import RestorationDataset
    from evidence_net.training.trainer import set_seed

    set_seed(seed)
    paths = resolve_dataset_paths()
    dataset = RestorationDataset(paths.train_dir, split=split, n_samples=n_samples, seed=seed)

    def load_model(checkpoint: Path) -> torch.nn.Module:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_config = ModelConfig(
            name=payload["config"]["model"]["name"],
            hidden_channels=payload["config"]["model"]["hidden_channels"],
            depth=payload["config"]["model"]["depth"],
            amplitude=payload["config"]["model"].get("amplitude", 0.1),
        )
        model = build_model(model_config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        return model

    proposal = load_model(REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt")
    # The proposal checkpoint is a full BoundedDetailProposal: calling it
    # directly returns the *candidate* (b + d). Use ``propose`` so ``d`` is
    # the bounded detail residual (matching measure_oracle.py).
    if not isinstance(proposal, BoundedDetailProposal):
        raise SystemExit("FAIL: proposal checkpoint is not a BoundedDetailProposal")
    sample_ids: list[str] = []
    inputs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    with torch.no_grad():
        for index in range(len(dataset)):
            input_tensor, target_tensor, sample_id = dataset[index]
            batch = input_tensor[None]
            y = up(batch).squeeze().numpy()
            x = target_tensor.squeeze(0).numpy()
            b, d, _c = proposal.propose(batch)
            sample_ids.append(sample_id)
            inputs.append(y)
            bases.append(b.squeeze().numpy())
            proposals.append(d.squeeze().numpy())
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

    # Calibration split: scores, calibration fit, and threshold fit only.
    # Validation split: policy evaluation only (never the reverse).
    if args.synthetic:
        cal_ids, cal_inputs, cal_bases, cal_proposals, cal_targets = _cases(
            max(4, args.n_samples // 2), args.seed, synthetic=True
        )
        sample_ids, inputs, bases, proposals, targets = _cases(
            args.n_samples, args.seed + 10_000, synthetic=True
        )
    else:
        cal_ids, cal_inputs, cal_bases, cal_proposals, cal_targets = _cases(
            max(4, args.n_samples // 2), args.seed, synthetic=False, split=CALIBRATION_SPLIT
        )
        sample_ids, inputs, bases, proposals, targets = _cases(
            args.n_samples, args.seed + 10_000, synthetic=False, split=VALIDATION_SPLIT
        )

    def _label_grids(
        grids_b: list[np.ndarray],
        grids_d: list[np.ndarray],
        grids_x: list[np.ndarray],
    ) -> list[np.ndarray]:
        return [
            np.asarray(patch_benefit_labels(b, d, x), dtype=np.float64)
            for b, d, x in zip(grids_b, grids_d, grids_x, strict=True)
        ]

    cal_labels = _label_grids(cal_bases, cal_proposals, cal_targets)

    # Calibrate the residual-magnitude baseline score on the calibration
    # split (fit isolation) and evaluate the policy on the validation split.
    cal_scores = _score_samples("residual-magnitude", cal_inputs, cal_bases, cal_proposals)
    mapping = fit_calibration(
        np.concatenate([score.reshape(-1) for score in cal_scores]),
        np.concatenate([label.reshape(-1) for label in cal_labels]),
        split=CALIBRATION_SPLIT,
    )
    scores = _score_samples("residual-magnitude", inputs, bases, proposals)
    probabilities = [
        mapping.apply(np.asarray(score, dtype=np.float64)).reshape(score.shape) for score in scores
    ]

    # Fit thresholds on the calibration split, then freeze. (The validation
    # split is only ever evaluated, never used to choose thresholds.)
    cal_probabilities = [
        mapping.apply(np.asarray(score, dtype=np.float64)).reshape(score.shape)
        for score in cal_scores
    ]
    band_fit_issue: str | None = None
    try:
        config = fit_policy_thresholds(
            np.concatenate([prob.reshape(-1) for prob in cal_probabilities]),
            np.concatenate([label.reshape(-1) for label in cal_labels]),
            split=CALIBRATION_SPLIT,
        )
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        # Benefit prediction is at chance on real data (Gate 4 negative
        # result), so the calibrated probability is near-constant and the
        # declared accept/reject bands are degenerate. Record the finding and
        # fall back to the default-accept policy (the candidate is better
        # than the Base on most patches); abstention is carried by the
        # orthogonal unresolved mask, never by certifying the Base.
        band_fit_issue = f"{type(exc).__name__}: {exc}"
        config = PolicyConfig()
        print(
            f"NOTE: policy band fit degenerate ({band_fit_issue}); "
            "using default-accept + unresolved abstention",
            file=sys.stderr,
        )

    # With a degenerate band fit (benefit at chance), the honest policy is
    # default-accept: the candidate is better than the Base on most patches,
    # so every patch is accepted and abstention is carried solely by the
    # orthogonal unresolved mask (never by certifying the Base).
    if band_fit_issue is not None:
        policy_probabilities = [np.ones_like(prob) for prob in probabilities]
    else:
        policy_probabilities = probabilities

    maps = [
        apply_policy(sample_id, probability, y, b, d, config)
        for sample_id, probability, y, b, d in zip(
            sample_ids, policy_probabilities, inputs, bases, proposals, strict=True
        )
    ]
    gated = [
        policy_outputs(action_map, b, d)
        for action_map, b, d in zip(maps, bases, proposals, strict=True)
    ]
    ungated = [np.clip(b + d, 0.0, 1.0) for b, d in zip(bases, proposals, strict=False)]
    # Abstention family: unresolved patches fall back to the Base (never
    # certified); everything else keeps the gated candidate.
    abstained: list[np.ndarray] = []
    for action_map, base, proposal in zip(maps, bases, proposals, strict=True):
        gate_map = np.repeat(np.repeat(action_map.gates, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
        fallback = np.repeat(
            np.repeat(action_map.unresolved.astype(np.float64), PATCH_SIZE, axis=0),
            PATCH_SIZE,
            axis=1,
        )
        keep = gate_map * (1.0 - fallback)
        abstained.append(np.clip(base + keep * proposal, 0.0, 1.0))

    outcomes: dict[str, dict[str, float]] = {}
    for label, family in (
        ("base", bases),
        ("ungated", ungated),
        ("gated", gated),
        ("abstained", abstained),
    ):
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
            "mode": "synthetic" if args.synthetic else "real",
            "evaluation_split": VALIDATION_SPLIT,
            "calibration_fit_split": CALIBRATION_SPLIT,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "policy": {
                "accept_threshold": config.accept_threshold,
                "reject_threshold": config.reject_threshold,
                "unresolved_edge_density": config.unresolved_edge_density,
                "band_fit_issue": band_fit_issue,
                "applied_as": "default-accept" if band_fit_issue is not None else "declared-bands",
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
        summary=_build_summary(run_id, outcomes, report, config, band_fit_issue),
        reference="frozen decision-policy-v1 thresholds (see config.yaml)",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


def _build_summary(
    run_id: str, outcomes: dict, report: dict, config: PolicyConfig, band_fit_issue: str | None
) -> str:
    band_note = ""
    if band_fit_issue is not None:
        band_note = (
            f"\n- Band fit: DEGENERATE ({band_fit_issue}) — benefit prediction "
            "is at chance on real data (Gate 4 negative result), so the declared "
            "accept/attenuate/reject bands are unusable; the policy was applied as "
            "default-accept + unresolved abstention."
        )
    lines = [
        f"# Selective policy evaluation {run_id}",
        "",
        f"- Policy: decision-policy-v1 (accept >= {config.accept_threshold}, "
        f"reject < {config.reject_threshold}, unresolved edge density >= "
        f"{config.unresolved_edge_density}).",
        band_note.strip("\n"),
        f"- Action fractions: {report['action_fractions']}",
        f"- Mean coverage: {report['mean_coverage']:.3f}; unresolved area: "
        f"{report['mean_unresolved_fraction']:.3f}",
        "",
        "## Outcomes (validation split, mean)",
        "",
        "| output | PSNR (dB) | SSIM | MAE | edge displacement (px) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for label in ("base", "ungated", "gated", "abstained"):
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
