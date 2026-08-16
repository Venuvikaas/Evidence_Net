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

**Phases 0–4 complete; four-developer handoff accepted (ADR-008)** —
repository skeleton, contracts, data foundation, trusted evaluation harness,
a frozen learned Base Reconstruction, and a bounded Detail Proposal with
oracle-measured headroom are in place. The official `train/` dataset
(3200 NoisyLR→GT pairs) is paired, audited, split deterministically; the
isolated `Test_NoisyLR/` set (400 inputs) is registered without touching any
development decision; the Base beats the classical baselines (Gate 2:
continue) and the oracle shows selective acceptance of the proposal improves
MAE by 6.3% and PSNR by 3 dB over the equal-capacity direct model (Gate 3:
continue).

After the handoff, four lanes work in parallel per
[`docs/four-developer-workflow.md`](docs/four-developer-workflow.md):

- **A** — Benefit Prediction, Calibration, Decision Policy, Abstention
  (Phases 5–6).
- **B** — Measurement Consistency, Stability, Familiarity, Structural Risk,
  Downstream Validation (Phases 7–10).
- **C** — Unified Inference, Metadata, API, Review UI, Human-Interpretation
  Tooling (Phases 11–14).
- **D** — Deployment, Optimization, Security, Monitoring, Release (Phases
  15–17).

Frozen handoff contracts live in [`docs/contracts/`](docs/contracts/README.md),
kill switches in [`docs/kill-switches.md`](docs/kill-switches.md), promoted
checkpoint hashes in
[`docs/handoff/checkpoint-registry.md`](docs/handoff/checkpoint-registry.md),
and lane ownership in [`CODEOWNERS`](CODEOWNERS).

| Phase | State |
| --- | --- |
| 0 — Project bootstrap and contracts | ✅ complete |
| 1 — Domain and data foundation | ✅ complete (decision: continue, ADR-005) |
| 2 — Evaluation harness and classical baselines | ✅ complete (tag `v0.1-data-eval`) |
| 3 — Learned Base Reconstruction | ✅ complete (decision: continue, ADR-006) |
| 4 — Bounded Detail Proposal + oracle study | ✅ complete (tag `v0.3-proposal-oracle`, decision: continue, ADR-007) |
| 5+ — Four parallel lanes after handoff | ready (handoff accepted, ADR-008; see `docs/four-developer-workflow.md`) |

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

## Phase 4 summary

- Proposal: `models/proposal.py` — bounded branch `d = α tanh(h_d(y, b))`
  (|d| ≤ α), ungated candidate `c = b + d`, fusion `x = b + g·d`;
  `proposal/targets.py` (residual targets, stop-gradient),
  `evaluation/proposal_metrics.py` (structural effect summaries).
- Oracle study: `evaluation/oracle.py` + `evaluation/oracle_report.py` —
  pixel and 16×16 patch gates from ground truth, coverage/risk, headroom
  reports with group bootstraps.
- Scripts: `train_proposal.py`, `measure_oracle.py`,
  `analyze_proposal_effects.py`; configs under `configs/model/proposal-*.yaml`.
- Governed oracle study (EXP-004, `runs/oracle-gate3-20260815-205601/`):
  oracle patch MAE 0.0373 vs Base 0.0399 (-6.3%), PSNR 25.66 vs 25.21 dB,
  vs equal-capacity direct 22.60 dB; pixel oracle 26.16 dB; coverage 86.8%.
  Harm concentrates in periodic regions (FAIL-001); Gate 3: continue.

## Phase 3 summary

- PyTorch training stack: `training/config.py` (validated YAML configs),
  `training/trainer.py` (seeded, checkpointing, resume, mixed precision,
  NaN/explosion/empty-batch guards), `training/provenance.py` (run bundles
  with environment capture).
- Models: `models/base.py` (Base Reconstruction `b = U(y) + h_b(f(y))`),
  `models/direct.py` (equal-capacity direct CNN), `models/validate.py`
  (output contract, gradient flow, checkpoint roundtrip, tiled parity).
- Losses: `losses/base_losses.py` — configurable pixel/structural/edge/
  frequency composite.
- Governed comparison (EXP-003, `runs/compare-gate2/`): Base PSNR 25.21 dB
  [23.19, 27.45] vs deterministic anchor 25.08 [23.03, 27.45], SSIM 0.639 vs
  0.599, MAE 0.0399 vs 0.0430; classical 24.46; direct 22.60. Failure
  catalogue (`docs/base-failures.md`): periodic 0.096, edge 0.084, flat
  0.030 MAE.
- Scripts: `scripts/train_base.py`, `scripts/compare_restoration.py`,
  `scripts/catalogue_failures.py`; configs in `configs/model/`.
- Frozen model: `checkpoints/train-base-gate2/best.pt` (promoted, tagged
  `v0.2-base-reconstruction`).

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

Work proceeds sequentially through the boxes in the execution plan. Each
completed box ends with a Conventional Commit; contracts and experiments are
recorded in the governance ledgers before code relies on them. After the
Phase 4 handoff, four lanes work in parallel: one lane objective per PR,
consumed contract versions named in every PR (template in
`.github/PULL_REQUEST_TEMPLATE.md`), and `python scripts/verify_handoff.py`
passing before merge. See `CONTRIBUTING.md`.
