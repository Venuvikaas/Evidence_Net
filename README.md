# EVIDENCE-Net

Evidence-aware selective restoration and validation for semiconductor-like
structural imagery.

EVIDENCE-Net separates a lower-intervention **Base Reconstruction** from an
explicit, amplitude-bounded **Detail Proposal**, and applies a **Decision
Policy** that accepts, attenuates, or rejects the proposal per region. It
reports **Unresolved Regions** where neither output is sufficiently
validated, and exposes independent diagnostics — measurement consistency,
model stability, and structural risk — as separate review layers.

**What it does not claim:** no score in the system proves that a
reconstructed structure physically existed. Every probability names its
exact event and calibration domain, and the release claims only what the
recorded evidence supports.

## Status

**v1.0.0 — validated release (tagged, CI green, 338 tests passing).**

The full pipeline is frozen and verified: the promoted Base and Proposal
checkpoints are hash-pinned, all research gates 1–10 are decided and
recorded (ADR-005…015), and the final one-pass evaluation ran on all 400
supported `Test_NoisyLR/` inputs with output coverage and the output
contract verified (see [release report](docs/release-report-v1.md)).

Key results from the governed real evaluations:

| Gate | Decision | Evidence |
| --- | --- | --- |
| 4 — Proposal-benefit prediction | **Simplify**: predictors at chance; the benefit event is the norm (79.4% of patches beneficial) | EXP-009, ADR-009 |
| 5 — Selective policy | **Continue**: default-accept + unresolved abstention beats the frozen Base (PSNR 25.38 vs 24.89 dB, MAE 0.0382 vs 0.0408) | EXP-010, ADR-010 |
| 6 — Measurement consistency | Keep (bounded operator residuals) | EXP-005, ADR-011 |
| 7 — Model stability | Keep (max perturbation drift 0.015) | EXP-006, ADR-012 |
| 8 — Distribution familiarity | **Not promoted**: 0% shift detection, rare-valid false warnings exceed cap | EXP-007, ADR-013 |
| 9 — Structural risk | Continue (five distinct evidence categories, hidden tests frozen) | EXP-008, ADR-014 |
| 10 — Human interpretation | Registered; participants unavailable — explicit limitation | EXP-011, ADR-015 |

## Honest limitations

Published in full in the [release report](docs/release-report-v1.md):

- Per-patch benefit prediction is not discriminative on the frozen event, so
  no support-aware gating claim is made — the policy defaults to accept.
- The familiarity diagnostic failed its gate on real data and is disabled by
  default.
- Human interpretation is untested: no participants were available; the
  study protocol and capture machinery are ready, but results are never
  simulated.
- The final `Test_NoisyLR/` set has no local targets, so final evaluation
  measures outputs only.
- No industrial modality validation, downstream labels, or expert user study
  was available — this is a research platform for semiconductor-like
  structural imagery.

## Getting started

Requirements: Python >= 3.10. The official datasets (`train/`,
`Test_NoisyLR/`) live **outside** this repository in the project parent
directory; they are never committed to Git, and `Test_NoisyLR/` never
influences training, validation, calibration, or policy tuning.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Then:

```bash
# Environment check (packages, directories, optional devices)
python scripts/check_env.py

# Smoke pipeline: loads a fixture and writes a run bundle to runs/<run_id>/
python scripts/smoke.py

# Frozen evaluation on Test_NoisyLR/ (one pass, outputs-only, coverage +
# contract verified; needs the local checkpoints)
python scripts/run_final_inference.py

# Quality gates
ruff check .
ruff format --check .
mypy src/evidence_net scripts
pytest
pre-commit run --all-files
```

CI runs lint, format, type checks, tests, environment check, smoke
pipelines, and the final-inference smoke on every push
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Repository layout

```text
evidence-net/
├── configs/                 # versioned configuration (data, model, decision, ...)
├── data/
│   ├── manifests/           # frozen, versioned data manifests
│   ├── fixtures/            # deterministic smoke fixtures
│   ├── stress/              # frozen hidden stress definitions
│   └── failures/            # natural failure bank
├── src/evidence_net/        # package: data, models, benefit, decision, evaluation,
│                            #          stress_tests, training, inference, api,
│                            #          monitoring, security, reporting
├── scripts/                 # training, measurement, release, and smoke scripts
├── tests/                   # unit, numerical, integration, calibration,
│                            # decision_parity, regression
├── runs/                    # generated run bundles (never committed)
├── deploy/                  # Dockerfile, docker-compose, ONNX export
├── frontend/                # technical review UI
└── docs/                    # contracts, scientific cards, reports
```

## Key components

- **Models** — Base Reconstruction (`src/evidence_net/models/base.py`) and
  the bounded Detail Proposal (`models/proposal.py`, `|d| <= α`, ungated
  candidate `c = b + d`, fusion `x = b + g·d`).
- **Benefit & calibration** (`src/evidence_net/benefit/`) — deterministic
  benefit labels, declared baselines, a minimal learned predictor, and
  calibration fit on the calibration split only.
- **Decision policy** (`src/evidence_net/decision/`) — accept /
  attenuate / reject actions with an orthogonal unresolved mask; rejection
  never certifies the Base output.
- **Inference & provenance** (`src/evidence_net/inference/`) — unified
  pipeline producing the full artifact contract with per-artifact hashes
  and semantic versions.
- **API & UI** — FastAPI service (`src/evidence_net/api/`) and the review
  frontend (`frontend/`) that displays only backend-computed values.
- **Operations** — deployment (`deploy/`), security controls
  (`src/evidence_net/security/`), and monitoring
  (`src/evidence_net/monitoring/`).

## Documentation

- [Product definition](docs/product-definition.md)
- [Release report v1](docs/release-report-v1.md)
- [Contracts](docs/contracts/README.md)
- [Data card](docs/data-card.md) · [Evaluation protocol](docs/evaluation-protocol.md)
- [Security & privacy operations](docs/security-and-privacy-operations.md)
- [Operational signals & monitoring](docs/operational-signals-and-monitoring.md)
- Governance ledgers: [DECISIONS](DECISIONS.md) ·
  [EXPERIMENTS](EXPERIMENTS.md) · [FAILURES](FAILURES.md) ·
  [CHANGELOG](CHANGELOG.md)

The governing execution plan lives in [`EXECUTION.md`](EXECUTION.md) and the
four-developer workflow in
[`docs/four-developer-workflow.md`](docs/four-developer-workflow.md);
contributions follow [`CONTRIBUTING.md`](CONTRIBUTING.md).
