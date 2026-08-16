# Four-Developer Workflow (post Phase-4 handoff)

This file mirrors the four-developer execution plan
(`EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md`) as an in-repository
contract: lane assignments, branch and PR rules, frozen contracts,
integration checkpoints, promotion, and kill switches. It becomes the
operating agreement the moment the Phase 4 handoff gate is accepted
(ADR-008).

## 1. Lanes

| Lane | Owner | Phases | Outputs | Owned paths (CODEOWNERS) |
| --- | --- | --- | --- | --- |
| A | Benefit Prediction and Decision Science | 5, 6, 18 (joint) | `SupportDefinition`, Benefit Predictor, `CalibrationVersion`, benefit maps, `DecisionPolicy`, action maps, unresolved masks, Gates 4-5 reports | `configs/support_definition/`, `configs/calibration/`, `configs/decision_policy/`, `src/evidence_net/decision/`, `src/evidence_net/benefit/`, `tests/calibration/` |
| B | Diagnostics and Structural Validation | 7, 8, 9, 10, 18 (joint) | consistency/stability/familiarity diagnostics, shift suites, structural-risk suite, natural failure bank, downstream report, Gates 6-9 reports | `src/evidence_net/stress_tests/`, `tests/numerical/` |
| C | Product and Review Platform | 11, 12, 13, 14, 18 (joint) | unified inference, provenance reports, metadata store, FastAPI service, review UI, human-interpretation workflow, API/UI tests | `src/evidence_net/api/`, `src/evidence_net/inference/`, `src/evidence_net/reporting/`, `frontend/`, `tests/integration/`, `tests/regression/` |
| D | Deployment, Security, Monitoring, Release | 15, 16, 17, release for 18 | containers, exported models, decision-parity reports, security controls, monitoring docs, release scripts, deployment candidate | `.github/workflows/`, `tests/decision_parity/`, `deploy/` |

Shared/frozen paths are owned by all lanes: `docs/`, `docs/contracts/`,
`data/`, `configs/data|modality|model|experiments/`,
`src/evidence_net/data|models|evaluation|training|losses|proposal/`,
`tests/unit/`, `scripts/`. Replace `CODEOWNERS` placeholders with real GitHub
handles or teams before the first post-handoff PR.

## 2. Frozen contracts (handoff freeze)

The nine contracts frozen at the Phase 4 handoff live in `docs/contracts/`:

`dataset-v1`, `tensor-v1`, `metrics-v1`, `artifacts-v1`, `base-output-v1`,
`proposal-output-v1`, `structural-summary-v1`, `oracle-report-v1`,
`error-and-optional-fields-v1`.

- All lanes consume exactly these versions. A lane may add **optional**
  fields under `error-and-optional-fields-v1`; it may never change a frozen
  field.
- Contract change procedure: ADR in `DECISIONS.md` → affected-owner review →
  new version → migration note → rerun decision → consumer migration. Old
  versions remain valid until all consumers migrate.

## 3. Handoff artifacts

- **Checkpoints.** `docs/handoff/checkpoint-registry.md` records sha256
  hashes and reproduction commands for the promoted Base
  (`checkpoints/train-base-gate2/best.pt`, tag `v0.2-base-reconstruction`)
  and Proposal (`checkpoints/train-proposal-gate3v2/best.pt`, tag
  `v0.3-proposal-oracle`). Checkpoints stay outside Git; the registry is the
  contract that pins them.
- **Fixtures.** `data/fixtures/manifest-v1.json` registers fixtures with
  schema and producer versions. Real Phase 4 fixtures (from frozen outputs)
  are for tests/reproduction; synthetic software-only fixtures cover future
  optional fields and errors. No scientific report uses synthetic fixtures.
- **Isolation.** `tests/unit/test_isolation.py` and `tests/unit/test_handoff.py`
  fail CI if `Test_NoisyLR/` enters any development manifest, config, or lane
  path.

## 4. Branch and pull-request rules

- One lane objective per PR; branch names
  `<lane>-<phase>-<short-objective>`; nothing merged to `main` without CI +
  `scripts/verify_handoff.py` + lane-owner approval.
- PRs name consumed contract versions (see `.github/PULL_REQUEST_TEMPLATE.md`).
- Cross-lane changes require an ADR and affected-owner review before merge.

## 5. Integration checkpoints (cross-lane)

- **Integration I — Benefit and policy promotion** (after A passes Gates 4-5):
  A publishes Benefit/Calibration/Decision/Unresolved contracts and fixtures;
  C integrates into inference/API/reports/UI; D extends parity and monitoring;
  B verifies diagnostics stay separate from Benefit semantics.
- **Integration II — Diagnostic promotion** (per B diagnostic): B publishes
  contract, implementation, ablation, applicability limits; A checks the
  frozen decision objective; C enables the separate layer; D extends export/
  parity/monitoring. Unproven diagnostics stay disabled by default.
- **Integration III — Structural validation:** B publishes structural and
  downstream reports tied to exact versions; A evaluates policy on the
  failure bank; C integrates failure browsing/report export; D verifies
  packaging and provenance integrity. Hidden tests stay isolated.
- **Integration IV — Human interpretation:** A validates Benefit/Decision
  wording; B validates diagnostic wording and failure examples; C validates
  interface behavior and study workflow; D validates privacy, retention, and
  audit behavior.
- **Integration V — Controlled final freeze:** all four freeze their outputs,
  approve one release candidate, and pass clean-clone smoke plus
  `Test_NoisyLR/` isolation checks before Phase 18.

## 6. Promotion and release

- Promoted outputs cross their integration checkpoint and are tagged
  (`v0.4-benefit-calibration`, `v0.5-selective-restoration`,
  `v0.6-structural-validation`, ...).
- Phase 18 is a joint release gate: frozen pipeline, one final pass over every
  supported `Test_NoisyLR/` input, preserved output mapping, full provenance,
  published limitations, and no post-evaluation tuning.

## 7. Kill switches

See [`docs/kill-switches.md`](kill-switches.md) for the complete matrix:
per-lane research gates 4-10, global process switches (isolation violation,
contract violation, secrets), and the mechanical enforcement
(`scripts/verify_handoff.py`, CI). A gate failure never blocks the record;
it redirects the lane (continue / redesign / remove / stop).

## 8. Current-work rule

Each developer maintains exactly: one active implementation objective, one
active experiment/integration question, one pending gate, one runnable lane
path. Everything else goes to `BACKLOG.md` (or an ADR proposal) before work
starts.
