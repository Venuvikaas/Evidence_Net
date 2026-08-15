# EVIDENCE-Net

Evidence-aware selective restoration and validation for semiconductor-like
structural imagery. EVIDENCE-Net separates a lower-intervention **Base
Reconstruction** from an explicit **Bounded Detail Proposal**, estimates a
calibrated **Proposal-Benefit Probability**, exposes independent
measurement-consistency, stability, and familiarity diagnostics, applies a
versioned **Decision Policy** (accept / attenuate / reject / abstain), and
marks **Unresolved Regions** where neither output is sufficiently validated.

It does **not** claim that a score proves a reconstructed structure physically
existed. Every probability names its exact event and calibration domain.

- Full product definition: [`docs/product-definition.md`](docs/product-definition.md)
- Governing execution plan: [`EXECUTION.md`](EXECUTION.md)
- Governance ledgers: [`DECISIONS.md`](DECISIONS.md), [`EXPERIMENTS.md`](EXPERIMENTS.md),
  [`FAILURES.md`](FAILURES.md), [`CHANGELOG.md`](CHANGELOG.md), [`BACKLOG.md`](BACKLOG.md)

## Status

**Phases 0–1 complete** — repository skeleton, contracts, automation spine, and
the official dataset foundation are in place. The official `train/` dataset
(3200 NoisyLR→GT pairs) is paired, audited, split deterministically, and the
isolated `Test_NoisyLR/` set (400 inputs) is registered without touching any
development decision.

| Phase | State |
| --- | --- |
| 0 — Project bootstrap and contracts | ✅ complete |
| 1 — Domain and data foundation | ✅ complete (decision: continue, ADR-005) |
| 2+ — Evaluation, models, diagnostics, product | pending (see `EXECUTION.md`) |

## Phase 1 summary

- Frozen source manifests: `data/manifests/official-train-source-v1.json`
  (6400 files) and `official-test-noisylr-source-v1.json` (400 files), each
  with per-file sha256 and the test set kept free of development labels.
- Pairing: 3200/3200 clean pairs by 6-digit base name; 0 unmatched, 0
  duplicated, 0 ambiguous. Exact and near duplicates: 0.
- Alignment: no dominant 2× phase — target-alignment uncertainty recorded in
  the train manifest (`target_uncertainty`).
- Splits (seed 0): train 2551 / validation 328 / calibration 164 /
  heldout-source 157 / heldout-degradation 0 (reserved); frozen in
  `dataset-splits-v1.json` and aggregated in `dataset-manifest-v1.json`.
- Key scripts: `validate_dataset_paths.py`, `resolve_dataset_paths.py`,
  `inventory_dataset.py`, `audit_dataset.py`, `build_splits.py`,
  `dryrun_loader.py`, `verify_manifests.py`.
- Docs: `docs/data-card.md`, `docs/train-structure.md`,
  `docs/test-noisylr-structure.md`, `docs/data-provenance.md`,
  `docs/grouping-and-splits.md`.

## Repository layout

```text
evidence-net/
├── configs/                 # versioned configuration (data, modality, model, ...)
├── data/
│   ├── manifests/           # frozen, versioned data manifests
│   └── fixtures/            # deterministic smoke fixtures
├── src/evidence_net/        # package: data, models, decision, losses, evaluation,
│                            #          stress_tests, training, inference, reporting, api
├── scripts/                 # check_env.py, smoke.py, ...
├── tests/                   # unit, numerical, integration, calibration,
│                            # decision_parity, regression
├── runs/                    # generated run bundles (never committed)
├── artifacts/               # generated artifacts (never committed)
├── frontend/                # review UI (Phase 13)
└── docs/                    # contracts and scientific cards
```

The official datasets (`train/`, `Test_NoisyLR/`) live **outside** this
repository, in the project parent directory. They are never committed to Git.
`Test_NoisyLR/` is isolated evaluation input and must never influence
development decisions (training, validation, calibration, or policy tuning).

## Setup (clean clone)

Requirements: Python >= 3.10.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Reproduction commands

```bash
# Environment check (packages, directories, optional devices)
python scripts/check_env.py

# Smoke pipeline: loads a fixture and writes a run bundle to runs/<run_id>/
python scripts/smoke.py

# Quality gates
ruff check .
ruff format --check .
mypy src/evidence_net scripts
pytest

# All pre-commit checks (formatting, lint, typing, large files, secrets)
pre-commit run --all-files
```

CI runs lint, format check, type check, unit tests, environment check, and the
smoke pipeline on every push (`.github/workflows/ci.yml`).

## One change per checked box

Work proceeds sequentially through the boxes in `EXECUTION.md`. Each completed
box ends with a Conventional Commit; contracts and experiments are recorded in
the governance ledgers before code relies on them.
