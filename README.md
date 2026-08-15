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

**Phase 0 complete** — repository skeleton, initial contracts, and automation
spine are in place. The project is reproducible before research begins.

| Phase | State |
| --- | --- |
| 0 — Project bootstrap and contracts | ✅ complete |
| 1 — Domain and data foundation | pending |
| 2+ — Evaluation, models, diagnostics, product | pending (see `EXECUTION.md`) |

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
