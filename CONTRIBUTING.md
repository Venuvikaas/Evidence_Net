# Contributing to EVIDENCE-Net

EVIDENCE-Net is developed by four parallel lanes after the Phase 4 handoff.
This file is the contribution contract every lane follows. The governing
execution plan is
[`EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md`](../EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md)
(kept next to this repository) and its in-repo mirror
[`docs/four-developer-workflow.md`](docs/four-developer-workflow.md).

## Lanes

| Lane | Owner | Phases | Owned paths (CODEOWNERS) |
| --- | --- | --- | --- |
| A | Benefit prediction and decision science | 5, 6, 18 (joint) | `configs/support_definition/`, `configs/calibration/`, `configs/decision_policy/`, `src/evidence_net/decision/`, `tests/calibration/` |
| B | Diagnostics and structural validation | 7, 8, 9, 10, 18 (joint) | `src/evidence_net/stress_tests/`, `tests/numerical/` |
| C | Product and review platform | 11, 12, 13, 14, 18 (joint) | `src/evidence_net/api/`, `src/evidence_net/inference/`, `src/evidence_net/reporting/`, `frontend/`, `tests/integration/`, `tests/regression/` |
| D | Deployment, security, monitoring, release | 15, 16, 17, release for 18 | `.github/workflows/`, `tests/decision_parity/`, `deploy/` |

Shared and frozen paths (`docs/`, `data/`, `configs/model/`,
`src/evidence_net/data|models|evaluation|training|losses|proposal/`,
`tests/unit/`, `scripts/`) are owned by all lanes and always require
cross-lane review. Replace the placeholder owners in `CODEOWNERS` with real
GitHub handles or teams before the first post-handoff pull request.

## Rules

### Branch and pull-request rules

- One lane objective per PR: one implementation objective, one experiment
  question, one pending gate.
- Branch names: `<lane>-<phase>-<short-objective>` (e.g.
  `a-phase5-benefit-labels`, `c-phase12-api-schemas`). The shared `main`
  branch stays releasable; no lane commits directly to `main`.
- Every PR must pass CI (lint, format, mypy, pytest, smoke) and
  `python scripts/verify_handoff.py`.
- Every PR names the exact frozen contract versions it consumes (use the PR
  template) and must be approved by the lane owner of every touched owned
  path. Cross-lane changes additionally require an ADR and affected-owner
  review.
- Commits use Conventional Commits with scopes from the execution plan
  (`benefit`, `calibration`, `decision`, `forward`, `stability`,
  `familiarity`, `stress`, `inference`, `api`, `ui`, `deploy`, `security`,
  `monitoring`, `release`, `data`, `eval`, `docs`, `test`, `chore`, ...).

### Contract-change rules (kill switch)

- A **frozen** contract (see `docs/contracts/`) may only change through a new
  version: ADR proposal → affected-owner review → version increment →
  migration note → rerun decision for affected experiments → consumer
  migration. Old versions stay valid until consumers migrate.
- No lane may silently change another lane's contract. Adding an **optional**
  field is allowed under `error-and-optional-fields-v1` when it names its own
  contract version and stays backward compatible.
- `scripts/verify_handoff.py` fails CI if any frozen contract is missing,
  unversioned, or marked draft.

### Fixture-version rules

- Fixtures live in `data/fixtures/` and are registered in
  `data/fixtures/manifest-v1.json`.
- Every fixture names its schema version and producer version (the contract
  version and code commit that produced it).
- Real (non-synthetic) fixtures come from frozen Phase 4 outputs and are
  allowed for tests and reproduction. Synthetic software-only fixtures exist
  for future optional fields and error cases. **No scientific report may use
  synthetic software-only fixtures.**
- Regenerating a fixture with a different producer requires a new fixture
  version entry; never overwrite a frozen fixture silently.

### Kill switches

Scientific and process kill switches are enumerated in
[`docs/kill-switches.md`](docs/kill-switches.md). A failed research gate is a
kill-switch event: record the evidence, then continue, redesign, or stop per
the predeclared criteria — never hide the result. A lane that fails its gate
does not merge further phase work until the gate decision is recorded.

### Per-developer current-work rule

After the handoff, each developer keeps exactly: one active implementation
objective, one active experiment or integration question, one pending gate,
and one runnable lane path. Anything else goes to `BACKLOG.md`.

## Quality gates (local, before pushing)

```bash
ruff check .
ruff format --check .
mypy src/evidence_net scripts
pytest
python scripts/verify_handoff.py
python scripts/check_env.py
```

Pre-commit runs the same checks plus large-file and secret detection
(`pre-commit install` once).
