"""Generate the dataset audit: ranges, clipping, structure, degradation,
alignment, and duplicates.

Reads the frozen source manifests (immutable), runs the documented audit
methods, and writes a run bundle under ``runs/audit-<timestamp>/`` with
metrics, a summary, and inspectable artifact examples. Run from the
repository root::

    python scripts/audit_dataset.py [--align-sample N] [--near-sample N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evidence_net.data.audit import (
    alignment_audit,
    compatibility_summary,
    degradation_summary,
    exact_duplicate_groups,
    export_alignment_examples,
    near_duplicate_groups,
    range_summary,
)
from evidence_net.data.manifests import SourceManifest, write_manifest
from evidence_net.data.pairing import audit_pairing, discover_train_structure, pair_integrity_report
from evidence_net.data.paths import DatasetPathError, resolve_dataset_paths
from evidence_net.reporting.run_bundle import create_run_bundle, new_run_id

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "data" / "manifests"

TRAIN_MANIFEST = MANIFESTS_DIR / "official-train-source-v1.json"
TEST_MANIFEST = MANIFESTS_DIR / "official-test-noisylr-source-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--align-sample", type=int, default=200, help="pairs for alignment audit")
    parser.add_argument(
        "--near-sample", type=int, default=0, help="files for near-dup audit (0=all)"
    )
    parser.add_argument(
        "--write-uncertainty",
        action="store_true",
        help="record dataset-level target-alignment uncertainty into the train "
        "source manifest and re-freeze it",
    )
    return parser.parse_args()


def _build_uncertainty_record(alignment: dict[str, Any]) -> dict[str, Any]:
    """Dataset-level target-alignment uncertainty from the audit."""
    return {
        "method": alignment["method"],
        "n_pairs": alignment["n_pairs"],
        "best_offsets": alignment["best_offsets"],
        "residual_stats": alignment["residual_stats"],
        "estimate": "dataset-level; no dominant 2x phase observed",
        "note": "recorded from the Phase 1 alignment audit; per-pair "
        "uncertainty is not yet available",
    }


def _load_manifest(path: Path) -> SourceManifest:
    if not path.is_file():
        print(
            f"FAIL: manifest not found: {path} (run scripts/inventory_dataset.py first)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    return SourceManifest.from_dict(data)


def main() -> int:
    args = parse_args()
    try:
        datasets = resolve_dataset_paths()
    except DatasetPathError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    train = _load_manifest(TRAIN_MANIFEST)
    test = _load_manifest(TEST_MANIFEST)

    train_inputs = [e for e in train.files if "NoisyLR" in e.relative_path]
    train_targets = [e for e in train.files if "/GT/" in e.relative_path]
    test_inputs = [e for e in test.files if "NoisyLR" in e.relative_path]

    gt_dir, noisy_dir = discover_train_structure(datasets.train_dir)
    pair_report = audit_pairing(gt_dir, noisy_dir)
    pairs = pair_report.pairs

    rng = np.random.default_rng(0)
    metrics = {
        "train_inputs": range_summary(train_inputs),
        "train_targets": range_summary(train_targets),
        "test_inputs": range_summary(test_inputs),
        "pairing": pair_report.summary(),
        "pair_integrity_manifest": pair_integrity_report(train.files),
        "compatibility_train_test_inputs": compatibility_summary(train_inputs, test_inputs),
        "alignment": alignment_audit(
            pairs, sample_limit=args.align_sample, rng=np.random.default_rng(1)
        ),
        "degradation": degradation_summary(
            pairs, sample_limit=args.align_sample, rng=np.random.default_rng(2)
        ),
        "exact_duplicates_train": {
            "groups": len(exact_duplicate_groups(train.files)),
            "files": sum(len(v) for v in exact_duplicate_groups(train.files).values()),
        },
        "near_duplicates_train": {},
    }

    near_groups = near_duplicate_groups(
        train.files, datasets.train_dir, sample_limit=args.near_sample, rng=rng
    )
    metrics["near_duplicates_train"] = {
        "groups": len(near_groups),
        "files": sum(len(v) for v in near_groups.values()),
        "examples": list(near_groups.items())[:5],
    }

    if args.write_uncertainty:
        uncertainty = _build_uncertainty_record(metrics["alignment"])
        for entry in train.files:
            entry.target_uncertainty = uncertainty
        train.dataset_hash = train.compute_dataset_hash()
        write_manifest(TRAIN_MANIFEST, train)
        print(f"Updated target uncertainty in {TRAIN_MANIFEST.name}")

    summary_lines = [
        "# Dataset audit",
        "",
        f"- Train manifest: {train.dataset_id} ({len(train.files)} files)",
        f"- Test manifest: {test.dataset_id} ({len(test.files)} files)",
        f"- Pairs: {pair_report.summary()}",
        f"- Compatibility (train vs test inputs): "
        f"{metrics['compatibility_train_test_inputs']['compatible']}",
        "",
        "See metrics.json for full statistics.",
    ]

    run_id = new_run_id("audit")
    run_dir = create_run_bundle(
        REPO_ROOT / "runs",
        run_id,
        config={
            "align_sample": args.align_sample,
            "near_sample": args.near_sample,
            "seed_policy": "numpy default_rng(0/1/2/3) fixed seeds",
            "source_manifests": [TRAIN_MANIFEST.name, TEST_MANIFEST.name],
        },
        manifest={
            "train_manifest": train.dataset_id,
            "train_manifest_hash": train.dataset_hash,
            "test_manifest": test.dataset_id,
            "test_manifest_hash": test.dataset_hash,
        },
        metrics=metrics,
        summary="\n".join(summary_lines),
        reference=f"source-manifests: {TRAIN_MANIFEST.name}, {TEST_MANIFEST.name}",
    )
    example_names = export_alignment_examples(pairs, run_dir / "artifacts")
    (run_dir / "artifacts" / "examples.json").write_text(
        json.dumps(example_names, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Inspectable alignment examples\n\n"
            + "\n".join(f"- {name}" for name in example_names)
            + "\n"
        )
    print(f"Audit bundle written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
