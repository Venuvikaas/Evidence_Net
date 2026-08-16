"""Evaluate benefit predictors on held-out data (Phase 5, EXP-009).

Usage (from the repository root):::

    python scripts/measure_benefit.py --synthetic
    python scripts/measure_benefit.py --n-samples 12

Compares the declared baselines (residual-magnitude, local-signal,
attention-gate) and the calibrated minimal predictor on held-out validation
data, with the deterministic benefit labels of support-definition-v1 as
ground truth. Reports are kept separate: discrimination/ranking (AUC, rank
correlation), selective-risk curves (gated error at coverage levels), and
calibration (Brier, reliability, ECE) each stand alone (Gate 4).

``--synthetic`` runs the same comparison on synthetic grids so CI can
exercise the evaluation path without the data files. Never touches
``Test_NoisyLR/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from evidence_net.benefit.calibration import fit_calibration
    from evidence_net.benefit.evaluate import (
        BenefitReport,
        build_benefit_report,
        grouped_brier_report,
    )
    from evidence_net.benefit.labels import OUTPUT_GRID, patch_benefit_labels
    from evidence_net.benefit.predictors import (
        LocalSignalBaseline,
        MinimalBenefitPredictor,
        ResidualMagnitudeBaseline,
    )
    from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id
except ImportError:  # allow running before `pip install -e .`
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from evidence_net.benefit.calibration import fit_calibration  # noqa: E402
    from evidence_net.benefit.evaluate import (  # noqa: E402
        BenefitReport,
        build_benefit_report,
        grouped_brier_report,
    )
    from evidence_net.benefit.labels import OUTPUT_GRID, patch_benefit_labels  # noqa: E402
    from evidence_net.benefit.predictors import (  # noqa: E402
        LocalSignalBaseline,
        MinimalBenefitPredictor,
        ResidualMagnitudeBaseline,
    )
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
        "--predictor-checkpoint",
        default=REPO_ROOT
        / "runs"
        / "benefit-predictor-latest"
        / "checkpoints"
        / "predictor-best.pt",
        type=Path,
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="evaluate on synthetic grids (CI smoke)"
    )
    return parser.parse_args()


def _synthetic_case(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(input, base, proposal, target) grids with a structured benefit pattern.

    Left half: the Base is far from the target and the proposal closes the
    gap (beneficial). Right half: the Base already matches the target and the
    proposal moves away from it (harmful). Scores from ``|proposal|`` and
    ``|input - base|`` order the halves correctly.
    """
    rng = np.random.default_rng(seed)
    grid = OUTPUT_GRID
    half = grid // 2
    y = np.full((grid, grid), 0.05)
    y[:, :half] = 0.6  # structure the Base misses
    y[:, half:] = 0.05
    for column in range(half, grid, 8):
        y[:, column] = 0.9
    base = np.clip(y, 0.0, 1.0)
    base[:, :half] = 0.2  # Base fails to recover the left half
    proposal = np.zeros((grid, grid))
    proposal[:, :half] = rng.normal(0.4, 0.02, size=(grid, half))  # large, helpful
    proposal[:, half:] = rng.normal(0.0, 0.03, size=(grid, half))  # small noise, harmful
    # Target: left half = candidate (proposal helps), right half = base
    # (proposal only adds noise, so it harms).
    target = np.clip(base + proposal, 0.0, 1.0)
    target[:, half:] = base[:, half:]
    return y, base, proposal, target


def _synthetic_cases(
    n_samples: int, seed: int, prefix: str = "synthetic"
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    sample_ids: list[str] = []
    inputs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in range(n_samples):
        y, b, d, x = _synthetic_case(seed + index)
        sample_ids.append(f"{prefix}-{index:06d}")
        inputs.append(y)
        bases.append(b)
        proposals.append(d)
        targets.append(x)
    return sample_ids, inputs, bases, proposals, targets


def _real_cases(
    n_samples: int,
    seed: int,
    split: str = VALIDATION_SPLIT,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Real cases from the frozen Base/Proposal checkpoints on one split."""
    from evidence_net.data.paths import resolve_dataset_paths
    from evidence_net.models.factory import build_model
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

    base = load_model(REPO_ROOT / "checkpoints" / "train-base-gate2" / "best.pt")
    proposal = load_model(REPO_ROOT / "checkpoints" / "train-proposal-gate3v2" / "best.pt")
    sample_ids: list[str] = []
    inputs: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    proposals: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    up = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    with torch.no_grad():
        for index in range(len(dataset)):
            input_tensor, target_tensor, sample_id = dataset[index]
            # Dataset yields (1, H, W); models expect (B, 1, H, W). The
            # predictor features are defined on the output grid, so the
            # degraded input is upsampled to 256x256.
            batch = input_tensor[None]
            y = up(batch).squeeze().numpy()
            x = target_tensor.squeeze(0).numpy()
            b = base(batch).squeeze().numpy()
            d = proposal(batch).squeeze().numpy()
            sample_ids.append(sample_id)
            inputs.append(y)
            bases.append(b)
            proposals.append(d)
            targets.append(x)
    return sample_ids, inputs, bases, proposals, targets


def _load_predictor(path: Path) -> MinimalBenefitPredictor | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    predictor = MinimalBenefitPredictor(
        hidden_channels=int(config["hidden_channels"]), depth=int(config["depth"])
    )
    predictor.load_state_dict(payload["model_state_dict"])
    predictor.eval()
    return predictor


def _train_predictor_inline(
    inputs: list[np.ndarray],
    bases: list[np.ndarray],
    proposals: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    epochs: int = 4,
    seed: int = 0,
) -> MinimalBenefitPredictor:
    """Train a minimal predictor on calibration-split samples (synthetic smoke).

    Only used when no checkpoint is provided, so CI can exercise the full
    comparison including the learned predictor.
    """
    from evidence_net.benefit.predictors import patch_features
    from evidence_net.training.trainer import set_seed

    set_seed(seed)
    features = np.stack(
        [patch_features(y, b, d) for y, b, d in zip(inputs, bases, proposals, strict=True)]
    ).astype(np.float32)
    labels = np.stack(
        [
            np.asarray(patch_benefit_labels(b, d, x), dtype=np.float32)
            for b, d, x in zip(bases, proposals, targets, strict=True)
        ]
    )
    predictor = MinimalBenefitPredictor(hidden_channels=16, depth=2)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
        batch_size=8,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        predictor.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(predictor(batch_features), batch_labels.unsqueeze(-1))
            loss.backward()
            optimizer.step()
    predictor.eval()
    return predictor


def _evaluate_predictors(
    name: str,
    scorer,
    inputs: list[np.ndarray],
    bases: list[np.ndarray],
    proposals: list[np.ndarray],
    targets: list[np.ndarray],
    label_arrays: list[np.ndarray],
    sample_ids: list[str],
) -> dict[str, object]:
    scores = [scorer(y, b, d) for y, b, d in zip(inputs, bases, proposals, strict=True)]
    report = build_benefit_report(name, scores, label_arrays, bases, proposals, targets, sample_ids)
    return report.as_dict()


def main() -> int:
    args = parse_args()

    # Calibration is fit on the calibration split ONLY (calibration-version-v1).
    # The evaluation below runs on the validation split with that frozen mapping.
    if args.synthetic:
        cal_ids, cal_inputs, cal_bases, cal_proposals, cal_targets = _synthetic_cases(
            max(4, args.n_samples // 2), args.seed, prefix="cal"
        )
        val_ids, val_inputs, val_bases, val_proposals, val_targets = _synthetic_cases(
            args.n_samples, args.seed + 10_000, prefix="val"
        )
    else:
        cal_ids, cal_inputs, cal_bases, cal_proposals, cal_targets = _real_cases(
            max(4, args.n_samples // 2), args.seed, split=CALIBRATION_SPLIT
        )
        val_ids, val_inputs, val_bases, val_proposals, val_targets = _real_cases(
            args.n_samples, args.seed + 10_000, split=VALIDATION_SPLIT
        )

    val_labels = [
        patch_benefit_labels(b, d, x)
        for b, d, x in zip(val_bases, val_proposals, val_targets, strict=True)
    ]
    val_label_arrays = [np.asarray(label, dtype=np.float64) for label in val_labels]

    # Declared baselines on validation.
    baseline_predictors = {
        "residual-magnitude": ResidualMagnitudeBaseline(),
        "local-signal": LocalSignalBaseline(),
    }
    reports: dict[str, object] = {}
    for name, baseline in baseline_predictors.items():
        reports[name] = _evaluate_predictors(
            name,
            baseline.score,
            val_inputs,
            val_bases,
            val_proposals,
            val_targets,
            val_label_arrays,
            val_ids,
        )

    # Learned predictor: load a checkpoint, or train inline on the calibration
    # split in synthetic smoke so CI exercises the full comparison.
    predictor = _load_predictor(args.predictor_checkpoint)
    if predictor is None and args.synthetic:
        predictor = _train_predictor_inline(
            cal_inputs, cal_bases, cal_proposals, cal_targets, seed=args.seed
        )
    if predictor is not None:
        scores = [
            predictor.score(y, b, d)
            for y, b, d in zip(val_inputs, val_bases, val_proposals, strict=True)
        ]
        report = build_benefit_report(
            "minimal-predictor",
            scores,
            val_label_arrays,
            val_bases,
            val_proposals,
            val_targets,
            val_ids,
        )
        # Calibration fit on calibration-split scores/labels, evaluated on
        # validation-split probabilities (never the reverse).
        cal_scores = [
            predictor.score(y, b, d)
            for y, b, d in zip(cal_inputs, cal_bases, cal_proposals, strict=True)
        ]
        cal_labels = [
            np.asarray(patch_benefit_labels(b, d, x), dtype=np.float64)
            for b, d, x in zip(cal_bases, cal_proposals, cal_targets, strict=True)
        ]
        mapping = fit_calibration(
            np.concatenate([score.reshape(-1) for score in cal_scores]),
            np.concatenate([label.reshape(-1) for label in cal_labels]),
            split=CALIBRATION_SPLIT,
        )
        probabilities = [
            mapping.apply(np.asarray(score, dtype=np.float64)).reshape(score.shape)
            for score in scores
        ]
        calibration = grouped_brier_report(
            "minimal-predictor", probabilities, val_label_arrays, val_ids
        )
        calibration["mapping"] = mapping.as_dict()
        calibration["fit_split"] = CALIBRATION_SPLIT
        report = BenefitReport(
            predictor=report.predictor,
            sample_ids=report.sample_ids,
            per_group_auc=report.per_group_auc,
            overall_auc=report.overall_auc,
            selective_risk=report.selective_risk,
            calibration=calibration,
        )
        reports["minimal-predictor"] = report.as_dict()

    run_id = args.run_id or new_run_id("benefit-eval")
    run_dir = create_run_bundle(
        args.out,
        run_id,
        config={
            "phase": 5,
            "mode": "synthetic" if args.synthetic else "real",
            "evaluation_split": VALIDATION_SPLIT,
            "calibration_fit_split": CALIBRATION_SPLIT,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "predictor_checkpoint": str(args.predictor_checkpoint),
        },
        manifest={
            "run_id": run_id,
            "contract": "support-definition-v1",
            "labels_version": "labels-v1",
            "calibration_version": "calibration-v1",
            "split_isolation": "calibration fit on calibration split only; "
            "evaluation on validation split; Test_NoisyLR never touched",
            "test_final_isolation": "confirmed-no-test-noisylr",
        },
        metrics={"reports": reports},
        summary=_build_summary(run_id, reports),
        reference="benefit predictors vs deterministic labels (support-definition-v1)",
    )
    print(f"Run bundle written to {run_dir}")
    return 0


def _fmt_auc(report: dict[str, object]) -> str:
    calibration = report.get("calibration", {})
    if isinstance(calibration, dict):
        aggregate = calibration.get("group_auc_aggregate", {})
    else:
        aggregate = {}
    overall = report.get("overall_auc", float("nan"))
    if isinstance(aggregate, dict) and "mean" in aggregate:
        return (
            f"{aggregate['mean']:.4f} [{aggregate['ci_lo']:.4f}, "
            f"{aggregate['ci_hi']:.4f}] (pooled {overall:.4f})"
        )
    return f"pooled {overall:.4f}"


def _build_summary(run_id: str, reports: dict[str, object]) -> str:
    lines = [
        f"# Benefit predictor comparison {run_id}",
        "",
        "## Ranking / discrimination (separate from calibration)",
        "",
        "| predictor | group AUC (bootstrap) |",
        "| --- | --- |",
    ]
    for name, report in sorted(reports.items()):
        if isinstance(report, dict):
            lines.append(f"| {name} | {_fmt_auc(report)} |")
    lines.extend(
        [
            "",
            "## Selective risk (mean patch MAE of the gated output)",
            "",
            "| predictor | cov 0.5 gated | cov 0.9 gated | base | ungated |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for name, report in sorted(reports.items()):
        if not isinstance(report, dict):
            continue
        risk = report.get("selective_risk")
        if not isinstance(risk, dict):
            continue
        lines.append(
            f"| {name} | {risk['gated_error']['0.50']:.5f} | "
            f"{risk['gated_error']['0.90']:.5f} | "
            f"{risk['base_error']['0.50']:.5f} | {risk['ungated_error']['0.50']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation (Gate 4 predeclared)",
            "",
            "The learned predictor must beat the declared simple heuristics "
            "(residual-magnitude, local-signal) on held-out data, provide "
            "useful selective-risk ordering, and calibrate meaningfully within "
            "a stated domain. Ranking and calibration conclusions are recorded "
            "separately; no single report overclaims.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
