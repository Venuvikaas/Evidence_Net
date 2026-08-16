# EVIDENCE-Net — Dataset-Aware Four-Developer Execution Framework

## One developer through Phase 4 · four developers in parallel after Phase 4 · no fixed deadline

This execution plan uses the official local datasets `train/` and `Test_NoisyLR/`. Phases 0–4 remain sequential under one developer. The split begins only after the Base Reconstruction, Bounded Detail Proposal, oracle study, data isolation rules, and shared artifact contracts are frozen and Research Gate 3 records **continue**.

The four post-Phase-4 lanes are:

- **Developer A:** Proposal-Benefit Prediction, Calibration, Decision Policy, and Abstention.
- **Developer B:** Measurement Consistency, Stability, Familiarity, Structural Risk, and Downstream Validation.
- **Developer C:** Unified Inference, Metadata, API, Review UI, and Human-Interpretation Tooling.
- **Developer D:** Deployment, Optimization, Security, Monitoring, Release Engineering, and Operations.

Phase 18 remains a joint release gate.

---

## How to read this file

- **[SOLO]**: initial developer, Phases 0–4.
- **[A]**, **[B]**, **[C]**, **[D]**: post-handoff lane owners.
- **[ALL]**: all four developers.
- **🔴 BLOCKING**: dependent work cannot proceed.
- **🟠 RESEARCH GATE**: continue, redesign, remove, or stop based on evidence.
- **🔗 CONTRACT**: a versioned scientific or software interface.
- **🔁 CHECKPOINT**: contract, integration, regression, or promotion checkpoint.
- **📦 RELEASE GATE**: reproducible internal milestone.

Each checked engineering item must leave code, tests, documentation, reproducible artifacts, and a small Conventional Commit.

---

# PART 1 — Dataset and Repository Contract

## Expected local layout

```text
project-parent/
├── EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md
├── train/                 # official development source; never commit
├── Test_NoisyLR/          # isolated final evaluation input; never commit
└── evidence-net/          # source repository
```

## Non-negotiable dataset rules

1. `train/` is the only official source for model training, architecture selection, validation, calibration, threshold selection, policy design, and development-time held-out evaluation.
2. `Test_NoisyLR/` may be inventoried early only for file compatibility and output mapping.
3. `Test_NoisyLR/` must not influence losses, features, model selection, calibration, policies, stress-test design, UI curation, or demo-case selection.
4. Final `Test_NoisyLR/` inference begins only after the four-lane controlled freeze.
5. Both directories remain outside Git.
6. Every dataset, model, policy, calibration, diagnostic, and output uses an immutable manifest or version identifier.

## Shared project files

```text
DECISIONS.md
EXPERIMENTS.md
FAILURES.md
CHANGELOG.md
CONTRIBUTING.md
BACKLOG.md
docs/contracts/
```

## Reproducible run bundle

```text
runs/<run_id>/
├── config.yaml
├── manifest.json
├── environment.txt
├── metrics.json
├── summary.md
├── logs/
└── artifacts/
```

---

# PART 2 — Sequential Foundation: Phases 0–4

## ☐ PHASE 0 — Bootstrap and Contracts [SOLO] 🔴 BLOCKING

- [ ] Initialize repository and source layout. — commit: `chore: initialize evidence-net repository`
- [ ] Add ignore rules for `train/`, `Test_NoisyLR/`, secrets, runs, checkpoints, caches, environments, and frontend builds. — commit: `chore: add ignore rules`
- [ ] Add `.env.example` with `TRAIN_DATA_DIR` and `TEST_NOISY_LR_DIR`. — commit: `chore(data): add dataset configuration`
- [ ] Configure packaging, formatting, linting, typing, tests, and pre-commit. — commit: `chore: configure development tooling`
- [ ] Add project ledgers and templates. — commit: `docs: add governance ledgers`
- [ ] Define modality, manifest, tensor, artifact, and run-bundle contracts. — commit: `docs(contracts): define initial contracts`
- [ ] Add environment and fixture smoke scripts. — commit: `test: add initial smoke pipeline`
- [ ] Add CI. — commit: `ci: add core quality workflow`
- [ ] Document clean-clone setup. — commit: `docs: add runbook`

### Exit gate

- [ ] Clean clone installs.
- [ ] CI and smoke tests pass.
- [ ] Dataset directories are not tracked.
- [ ] Initial contracts are versioned.

---

## ☐ PHASE 1 — Official Dataset Integration [SOLO] 🔴 BLOCKING

### Source intake

- [ ] Resolve `train/` and `Test_NoisyLR/` relative to project configuration. — commit: `feat(data): resolve official sources`
- [ ] Fail clearly if either directory is missing. — commit: `test(data): validate required sources`
- [ ] Inventory `train/` with paths, hashes, types, dimensions, channels, data types, ranges, and readability. — commit: `feat(data): inventory train data`
- [ ] Inventory `Test_NoisyLR/` separately without model inference. — commit: `feat(data): inventory isolated test inputs`
- [ ] Document the actual `train/` structure and discover pairing from observed files and supplied documentation. — commit: `docs(data): document train structure`
- [ ] Implement and version the training pairing adapter. — commit: `feat(data): implement pairing adapter`
- [ ] Report missing, ambiguous, and duplicate pairs. — commit: `feat(data): audit pair integrity`
- [ ] Document `Test_NoisyLR/` input naming and expected output mapping. — commit: `docs(data): document test contract`
- [ ] Freeze separate train and test source manifests. — commit: `chore(data): freeze source manifests`

### Alignment, leakage, and development splits

- [ ] Measure input-target alignment and store uncertainty. — commit: `feat(data): audit alignment`
- [ ] Detect exact and near duplicates. — commit: `feat(data): detect duplicates`
- [ ] Define source groups preventing related samples from crossing splits. — commit: `docs(data): define source groups`
- [ ] Create training, validation, calibration, source-held-out, and degradation-held-out manifests from `train/` only. — commit: `feat(data): create development splits`
- [ ] Add a test that rejects any `Test_NoisyLR/` path in a development manifest. — commit: `test(data): enforce evaluation isolation`
- [ ] Compare train/test file compatibility without using model outcomes. — commit: `feat(data): verify input compatibility`
- [ ] Publish the data card. — commit: `docs(data): publish data card`

### 🟠 Research Gate 1

Continue only if pairing, alignment, source grouping, development splits, and evaluation isolation are valid or explicitly bounded.

---

## ☐ PHASE 2 — Evaluation Harness and Baselines [SOLO] 🔴 BLOCKING

- [ ] Define primary metrics, secondary diagnostics, and statistical units. — commit: `docs(eval): define metric contracts`
- [ ] Implement PSNR, SSIM, MAE, edge displacement, structural error, and frequency diagnostics. — commit: `feat(eval): implement metrics`
- [ ] Add image- and source-group confidence intervals. — commit: `feat(eval): add grouped statistics`
- [ ] Test metrics with analytical fixtures and controlled perturbations. — commit: `test(eval): validate metrics`
- [ ] Implement deterministic and classical restoration baselines. — commit: `feat(model): add classical baselines`
- [ ] Implement common inference, artifact, evaluation, and report interfaces. — commit: `feat(inference): add baseline pipeline`
- [ ] Extend smoke tests through report generation. — commit: `test: extend evaluation smoke`
- [ ] Tag `v0.1-data-eval`. — commit: `chore(release): tag evaluation foundation`

---

## ☐ PHASE 3 — Learned Base Reconstruction [SOLO] 🔴 BLOCKING

- [ ] Implement structured configuration and reproducible trainer. — commit: `feat(training): add reproducible trainer`
- [ ] Add checkpoint, resume, environment, seed, NaN, range, and tiny-overfit checks. — commit: `test(training): add training guards`
- [ ] Implement strong direct-restoration baseline. — commit: `feat(model): add direct baseline`
- [ ] Implement candidate Base Reconstruction. — commit: `feat(model): add base reconstruction`
- [ ] Implement configurable restoration losses. — commit: `feat(model): add base losses`
- [ ] Run governed classical, direct, and Base comparisons. — commit: `exp(model): compare base models`
- [ ] Catalogue structural failures. — commit: `docs(failures): catalogue base failures`
- [ ] Record whether lower-intervention behavior is supported. — commit: `docs(decision): assess base claim`
- [ ] Promote model and tag `v0.2-base-reconstruction`. — commit: `chore(model): promote base model`

### 🟠 Research Gate 2

Continue only if the Base Reconstruction is independently useful and its structural behavior is understood.

---

## ☐ PHASE 4 — Detail Proposal and Oracle Study [SOLO] 🔴 BLOCKING

- [ ] Freeze proposal tensor and structural-summary contract. — commit: `docs(proposal): define proposal contract`
- [ ] Implement bounded Detail Proposal and ungated candidate. — commit: `feat(proposal): add detail branch`
- [ ] Generate proposal targets from frozen Base outputs. — commit: `feat(proposal): generate targets`
- [ ] Implement proposal magnitude, multiscale energy, edge displacement, and component-change summaries. — commit: `feat(proposal): add structural summaries`
- [ ] Test fusion identities. — commit: `test(proposal): verify gate identities`
- [ ] Implement pixel- and patch-level oracle gating. — commit: `feat(eval): add oracle gating`
- [ ] Implement oracle coverage-risk and structural-impact reports. — commit: `feat(eval): report oracle headroom`
- [ ] Predeclare meaningful oracle-headroom criteria. — commit: `docs(experiment): register oracle study`
- [ ] Train proposal against frozen Base outputs. — commit: `exp(proposal): train detail proposer`
- [ ] Compare Base, ungated candidate, equal-capacity direct model, and oracle-gated outputs. — commit: `exp(proposal): evaluate oracle headroom`
- [ ] Archive natural harmful proposals. — commit: `docs(failures): add proposal failures`

### 🟠 Research Gate 3

Record one decision:

- **continue**
- **redesign proposal**
- **change spatial unit**
- **abandon gated decomposition**

Four-way parallel work begins only if the decision is **continue**.

- [ ] Tag `v0.3-proposal-oracle`. — commit: `chore(release): tag proposal milestone`

---

# PART 3 — Four-Developer Handoff 🔴 BLOCKING

## Contracts to freeze

- [ ] `dataset-v1`
- [ ] `tensor-v1`
- [ ] `metrics-v1`
- [ ] `artifacts-v1`
- [ ] `base-output-v1`
- [ ] `proposal-output-v1`
- [ ] `structural-summary-v1`
- [ ] `oracle-report-v1`
- [ ] `error-and-optional-fields-v1`

— commit: `docs(contracts): freeze phase 4 handoff`

## Artifacts and workflow

- [ ] Publish immutable Base and Proposal checkpoints with hashes and reproduction commands. — commit: `chore(model): publish handoff checkpoints`
- [ ] Publish permitted real Phase 4 fixtures. — commit: `test(fixtures): publish phase 4 fixtures`
- [ ] Publish synthetic software-only fixtures for future optional fields and errors. — commit: `test(fixtures): add integration fixtures`
- [ ] Add CODEOWNERS for A, B, C, and D. — commit: `chore: add four lane ownership`
- [ ] Add branch, pull-request, contract-change, and fixture-version rules. — commit: `docs: add four developer workflow`
- [ ] All four developers reproduce the Phase 4 vertical slice. — commit: `test: reproduce phase 4 handoff`
- [ ] All four environments pass `Test_NoisyLR/` isolation tests. — commit: `test(data): verify team isolation`
- [ ] Record the accepted handoff and integration protocol in `DECISIONS.md`. — commit: `docs: accept phase 4 handoff`

**Only after this gate do four independent lanes begin.**

---

# PART 4 — Four Parallel Phase Sets

## Developer A — Benefit Prediction and Decision Science

### Owned phases

- Phase 5
- Phase 6
- Phase 18 jointly with all developers

### ☐ PHASE 5 — Proposal-Benefit Prediction and Calibration [A]

- [ ] Define `SupportDefinition-v1`. — commit: `docs(benefit): define event v1`
- [ ] Implement versioned benefit-label generation. — commit: `feat(benefit): generate labels`
- [ ] Test beneficial, harmful, tied, and invalid examples. — commit: `test(benefit): validate labels`
- [ ] Implement residual-magnitude and local-signal heuristics. — commit: `feat(benefit): add heuristic baselines`
- [ ] Implement reconstruction-trained attention baseline. — commit: `feat(benefit): add attention baseline`
- [ ] Implement minimal learned Benefit Predictor. — commit: `feat(benefit): add learned predictor`
- [ ] Train separately from Base and Proposal. — commit: `exp(benefit): train predictor`
- [ ] Define `CalibrationVersion-v1`. — commit: `docs(calibration): define calibration contract`
- [ ] Implement calibration while preserving raw scores. — commit: `feat(calibration): calibrate scores`
- [ ] Compare heuristics, attention, and learned predictions across held-out groups. — commit: `exp(benefit): compare predictors`
- [ ] Report ranking separately from calibration. — commit: `docs(eval): report benefit reliability`

#### 🟠 Research Gate 4

The predictor must beat declared simple heuristics, provide useful selective-risk ordering, and have calibration valid within a stated domain.

### ☐ PHASE 6 — Decision Policy and Abstention [A]

- [ ] Define accept, attenuate, reject, and abstain semantics. — commit: `docs(decision): define actions`
- [ ] Define costs, critical-region rules, thresholds, and unresolved semantics. — commit: `docs(decision): define objective`
- [ ] Implement versioned policy schema. — commit: `feat(decision): add policy schema`
- [ ] Implement gating, attenuation, and unresolved mask. — commit: `feat(decision): add selective policy`
- [ ] Test that proposal rejection does not certify the Base output. — commit: `test(decision): preserve fallback uncertainty`
- [ ] Compare policies using validation and calibration data only. — commit: `exp(decision): compare policies`
- [ ] Freeze policy thresholds before held-out use. — commit: `chore(decision): freeze policy v1`

#### 🟠 Research Gate 5

Selective action must improve a predeclared outcome, and abstention must lower risk at useful coverage.

### Developer A outputs

- `SupportDefinition`
- Benefit Predictor
- `CalibrationVersion`
- Proposal-Benefit maps
- `DecisionPolicy`
- Action maps
- Unresolved masks
- Gates 4 and 5 reports

---

## Developer B — Diagnostics and Structural Validation

### Owned phases

- Phase 7
- Phase 8
- Phase 9
- Phase 10
- Phase 18 jointly with all developers

### ☐ PHASE 7 — Measurement Consistency [B]

- [ ] Define bounded modality-specific forward operator contract. — commit: `docs(forward): define forward model`
- [ ] Define parameter bounds, operation order, and stochastic treatment. — commit: `docs(forward): specify semantics`
- [ ] Implement deterministic and stochastic operators. — commit: `feat(forward): implement operators`
- [ ] Report compatibility across the operator family. — commit: `feat(forward): report compatibility`
- [ ] Add non-identifiability and misspecification cases. — commit: `feat(stress): add forward limitations`
- [ ] Evaluate incremental value beyond simple Benefit features. — commit: `exp(forward): measure contribution`

#### 🟠 Research Gate 6

Keep only if it adds held-out value or independently useful review information. It is compatibility, not truth.

### ☐ PHASE 8 — Model Stability [B]

- [ ] Define stability contract and prohibited interpretations. — commit: `docs(stability): define contract`
- [ ] Implement invertible perturbation comparisons. — commit: `feat(stability): add perturbation stability`
- [ ] Implement checkpoint comparisons. — commit: `feat(stability): add checkpoint stability`
- [ ] Add diverse-model comparison only with measured error diversity. — commit: `feat(stability): add diverse comparison`
- [ ] Evaluate incremental value. — commit: `exp(stability): measure contribution`

#### 🟠 Research Gate 7

Agreement remains stability, not correctness.

### ☐ PHASE 9 — Distribution Familiarity and Shift [B]

- [ ] Define representation, reference population, distance, threshold, and applicability. — commit: `docs(familiarity): define contract`
- [ ] Implement a simple reference-distance baseline. — commit: `feat(familiarity): add baseline`
- [ ] Construct source, severity, degradation, and acquisition shifts from permitted development data. — commit: `feat(stress): add shift suites`
- [ ] Evaluate rare valid structures separately. — commit: `exp(familiarity): test rare structures`
- [ ] Publish validated familiarity output and applicability limits. — commit: `docs(familiarity): publish validation`

#### 🟠 Research Gate 8

The diagnostic must detect declared shifts without systematically suppressing rare valid structures.

### ☐ PHASE 10 — Structural-Risk and Downstream Validation [B]

- [ ] Implement false-line, deletion, edge-shift, merge, split, false-periodicity, and defect-point candidate manipulations. — commit: `feat(stress): add candidate suite`
- [ ] Build observation-ambiguity cases. — commit: `feat(stress): add ambiguity suite`
- [ ] Add modality-plausible acquisition artifacts. — commit: `feat(stress): add acquisition suite`
- [ ] Curate frozen natural failures. — commit: `docs(failures): curate natural failures`
- [ ] Select one available frozen downstream task. — commit: `docs(downstream): define validation task`
- [ ] Evaluate downstream changes without co-training it. — commit: `exp(downstream): evaluate selective restoration`
- [ ] Freeze hidden final stress definitions. — commit: `chore(stress): freeze hidden tests`
- [ ] Enforce stress-test isolation. — commit: `test(stress): enforce isolation`

#### 🟠 Research Gate 9

Structural claims require separate candidate, ambiguity, acquisition, natural-failure, and downstream evidence.

### Developer B outputs

- Measurement-Consistency Diagnostic
- Stability Diagnostic
- Familiarity Diagnostic
- Shift suites
- Structural-risk suite
- Natural failure bank
- Downstream report
- Gates 6–9 reports

---

## Developer C — Product and Review Platform

### Owned phases

- Phase 11
- Phase 12
- Phase 13
- Phase 14
- Phase 18 jointly with all developers

### ☐ PHASE 11 — Unified Inference, Reports, and Provenance [C]

- [ ] Build sample-to-artifact inference from frozen Phase 4 outputs. — commit: `feat(inference): assemble pipeline`
- [ ] Support optional future Benefit, diagnostics, Decision, and Unresolved fields. — commit: `feat(inference): add optional contracts`
- [ ] Record dataset, model, support, calibration, diagnostic, and policy versions. — commit: `feat(provenance): record versions`
- [ ] Generate Markdown and JSON review packages. — commit: `feat(report): generate review package`
- [ ] Hash artifacts and deterministic references. — commit: `feat(provenance): hash artifacts`
- [ ] Add full-loop smoke and golden regression tests. — commit: `test: add full loop regression`

### ☐ PHASE 12 — Metadata Store and API [C]

- [ ] Implement version, run, artifact, metric, policy, and review models. — commit: `feat(api): add metadata models`
- [ ] Add migrations and initialization. — commit: `feat(api): add migrations`
- [ ] Freeze API schemas from artifact contracts. — commit: `docs(api): define api contract`
- [ ] Add restoration, status, artifact, comparison, stress, metadata, and health routes. — commit: `feat(api): add endpoints`
- [ ] Add upload validation and safe errors. — commit: `feat(api): secure uploads`
- [ ] Test CLI/API output equivalence. — commit: `test(api): verify parity`

### ☐ PHASE 13 — Technical Review Interface [C]

- [ ] Build typed client. — commit: `feat(ui): add typed client`
- [ ] Build synchronized Input, Base, Proposal, Candidate, Final, and available Target panes. — commit: `feat(ui): add restoration workspace`
- [ ] Add positive and negative proposal views. — commit: `feat(ui): add intervention inspector`
- [ ] Add optional separate Benefit, Consistency, Stability, Familiarity, Decision, and Unresolved layers. — commit: `feat(ui): add reliability layers`
- [ ] Add exact legends, applicability, and unavailable states. — commit: `feat(ui): add semantic legends`
- [ ] Add pixel/patch inspector and policy explorer. — commit: `feat(ui): add local inspector`
- [ ] Add calibration, selective-risk, failure, provenance, and report views. — commit: `feat(ui): add reliability workspace`
- [ ] Add accessibility, alignment, component, and end-to-end tests. — commit: `test(ui): verify review interface`

### ☐ PHASE 14 — Human Interpretation and Review Workflow [C; A+B review]

- [ ] A supplies correct Benefit, Calibration, Decision, and Abstention interpretations. — commit: `docs(research): define decision semantics`
- [ ] B supplies ambiguity, failure, shift, and diagnostic cases. — commit: `feat(research): add study cases`
- [ ] C writes the study protocol and builds review-event capture. — commit: `feat(review): add study workflow`
- [ ] Add consent and data-handling documentation if required. — commit: `docs(research): add data handling`
- [ ] Run available pilot or expert review. — commit: `docs(research): report findings`
- [ ] Rename, redesign, or remove systematically misunderstood layers. — commit: `fix(ui): address interpretation failures`

#### 🟠 Research Gate 10

Users must distinguish Proposal Benefit, compatibility, stability, familiarity, rejection, and unresolved output without treating them as physical proof.

### Developer C outputs

- Unified inference
- Provenance reports
- Metadata store
- FastAPI service
- Review UI
- Human-interpretation workflow
- API/UI integration tests

---

## Developer D — Deployment, Security, Monitoring, and Release Operations

### Owned phases

- Phase 15
- Phase 16
- Phase 17
- Release engineering for Phase 18

Developer D starts immediately after the Phase 4 handoff using promoted checkpoints and frozen fixtures. Runtime support for later A/B/C outputs remains optional until those contracts are promoted.

### ☐ PHASE 15 — Deployment, Optimization, and Decision Parity [D; A+B+C verify]

- [ ] Containerize the reference service and define mounted dataset/artifact paths. — commit: `feat(deploy): containerize service`
- [ ] Add local deployment composition with environment-based secrets. — commit: `feat(deploy): add service stack`
- [ ] Document CPU/GPU requirements. — commit: `docs(deploy): document runtimes`
- [ ] Build export pipeline for promoted model components. — commit: `feat(deploy): add model export`
- [ ] Export promoted components to ONNX. — commit: `feat(deploy): export onnx models`
- [ ] Validate tensor and spatial parity using Phase 4 outputs. — commit: `test(deploy): validate phase 4 parity`
- [ ] Extend parity to ranking, calibration, action, and abstention when A promotes them. — commit: `test(deploy): validate decision parity`
- [ ] Extend parity to promoted B diagnostics. — commit: `test(deploy): validate diagnostic parity`
- [ ] Add TensorRT only if justified. — commit: `feat(deploy): add optional tensorrt`
- [ ] Benchmark size, memory, latency, and throughput. — commit: `perf: benchmark pipeline`
- [ ] Validate tiled inference boundaries and map alignment. — commit: `test(inference): validate tiled parity`

### ☐ PHASE 16 — Security, Privacy, and Integrity [D]

- [ ] Keep raw tensors out of ordinary logs. — commit: `test(security): prevent tensor logging`
- [ ] Add retention controls. — commit: `feat(security): add retention controls`
- [ ] Add role-aware access when shared deployment requires it. — commit: `feat(security): add access controls`
- [ ] Validate filenames, file types, sizes, decompression, and paths. — commit: `test(security): harden uploads`
- [ ] Restrict model loading to trusted formats and paths. — commit: `fix(security): restrict model loading`
- [ ] Pin and scan dependencies and images. — commit: `chore(security): add scanning`
- [ ] Verify artifact, policy, calibration, and model hashes. — commit: `feat(security): verify integrity`
- [ ] Audit Git history for secrets and restricted data. — commit: `chore(security): audit repository`

### ☐ PHASE 17 — Monitoring and Operations [D; A+B define scientific signals]

- [ ] Add latency, error, memory, queue, and artifact-write metrics. — commit: `feat(monitoring): add service telemetry`
- [ ] Add input-range and acquisition-statistic drift summaries. — commit: `feat(monitoring): add data health`
- [ ] Track action and unresolved distributions after A promotion. — commit: `feat(monitoring): track decisions`
- [ ] Track diagnostic distributions after B promotion. — commit: `feat(monitoring): track diagnostics`
- [ ] Track active semantic versions. — commit: `feat(monitoring): track versions`
- [ ] Add reviewed-case calibration and selective-risk reporting. — commit: `feat(monitoring): add reliability review`
- [ ] Add deployment parity monitoring. — commit: `feat(monitoring): monitor parity`
- [ ] Document alert meaning and unsupported interpretations. — commit: `docs(monitoring): define operational signals`

### Developer D outputs

- Containers and runtime packages
- Exported models
- Decision-parity reports
- Security controls
- Monitoring and operations documentation
- Release scripts
- Reproducible deployment candidate

---

# PART 5 — Cross-Lane Integration Checkpoints

## 🔁 Integration I — Benefit and policy promotion

Triggered after A passes Gates 4 and 5.

- [ ] A publishes Benefit, Calibration, Decision, and Unresolved contracts and fixtures.
- [ ] C integrates them into inference, API, reports, and UI.
- [ ] D extends runtime parity and monitoring.
- [ ] B checks that diagnostics remain separate from Benefit semantics.
- [ ] End-to-end gate-zero, gate-one, reject, abstain, alignment, provenance, and parity tests pass.

## 🔁 Integration II — Diagnostic promotion

Triggered separately for each B diagnostic.

- [ ] B publishes contract, implementation, ablation, and applicability limits.
- [ ] A checks whether it improves the frozen decision objective.
- [ ] C enables the separate layer and exact legend.
- [ ] D extends export, parity, and monitoring where needed.
- [ ] Unproven diagnostics remain disabled by default.

## 🔁 Integration III — Structural validation

- [ ] B publishes structural and downstream reports tied to exact versions.
- [ ] A evaluates policy behavior on the failure bank.
- [ ] C integrates failure browsing and report export.
- [ ] D verifies packaging and provenance integrity.
- [ ] Hidden tests remain isolated.

## 🔁 Integration IV — Human interpretation

- [ ] A validates Benefit and Decision wording.
- [ ] B validates diagnostic wording and failure examples.
- [ ] C validates interface behavior and study workflow.
- [ ] D validates privacy, retention, and audit behavior.

## 🔁 Integration V — Controlled final freeze

- [ ] A freezes Predictor, Calibration, Policy, and action semantics.
- [ ] B freezes diagnostics, structural evaluation, and statistical protocol.
- [ ] C freezes inference, reports, API, UI, and interpretation language.
- [ ] D freezes deployment, security, monitoring, packaging, and release scripts.
- [ ] All four approve one release candidate.
- [ ] All environments pass clean-clone smoke and `Test_NoisyLR/` isolation checks.

---

# PART 6 — PHASE 18 Joint Final Validation and Release [ALL] 🔴 BLOCKING

## Frozen evaluation

- [ ] Freeze code, models, contracts, calibration, policies, diagnostics, UI, runtime, and packaging. — commit: `chore(release): freeze candidate`
- [ ] Run the frozen pipeline once on every supported `Test_NoisyLR/` input. — commit: `exp(release): run final inference`
- [ ] Preserve original relative input names in the output manifest. — commit: `feat(release): preserve output mapping`
- [ ] Verify one output per supported input and no extras. — commit: `test(release): verify coverage`
- [ ] Validate dimensions, type, range, names, and packaging. — commit: `test(release): validate output contract`
- [ ] Record source, output, model, calibration, diagnostic, policy, UI, and runtime hashes. — commit: `docs(release): record provenance`
- [ ] Generate final permitted evaluation results. — commit: `docs(release): report results`
- [ ] Publish failures, negative findings, and limitations. — commit: `docs(release): publish limitations`
- [ ] Confirm no post-evaluation tuning. — commit: `docs(release): record integrity`

## Release checks

- [ ] All tests pass.
- [ ] Every artifact names required semantic versions.
- [ ] Dataset isolation passes.
- [ ] Runtime parity passes.
- [ ] Security and repository audits pass.
- [ ] Clean-clone deployment passes.
- [ ] Tag validated release. — commit: `chore(release): publish validated release`

---

# PART 7 — Four-Lane Dependency Map

```text
SOLO
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Gate 3 -> Handoff
                                                       |
      +---------------------------+---------------------------+---------------------------+
      |                           |                           |                           |
A     Phase 5 -> Phase 6          |                           |                           |
      Benefit + Policy            |                           |                           |
      |                           |                           |                           |
B     Phase 7 -> 8 -> 9 -> 10     |                           |                           |
      Diagnostics + Validation    |                           |                           |
      |                           |                           |                           |
C     Phase 11 -> 12 -> 13 -> 14  |                           |                           |
      Product + Review            |                           |                           |
      |                           |                           |                           |
D     Phase 15 -> 16 -> 17        |                           |                           |
      Deployment + Operations     |                           |                           |
      +---------------------------+---------------------------+---------------------------+
                                                       |
                                            Integration V -> Phase 18
```

The lanes do not need to finish simultaneously. Promoted artifacts cross explicit integration checkpoints.

---

# PART 8 — Definition of Done

## Scientific

- [ ] Dataset provenance, pairing, alignment, grouping, and uncertainty are documented.
- [ ] `train/` and `Test_NoisyLR/` have separate immutable manifests.
- [ ] `Test_NoisyLR/` never enters development.
- [ ] Base is independently competitive.
- [ ] Oracle gating demonstrates meaningful headroom.
- [ ] Benefit Predictor beats simple heuristics.
- [ ] Ranking and calibration are separate.
- [ ] Abstention reduces risk at stated coverage.
- [ ] Every diagnostic earns its place.
- [ ] Structural threat models remain distinct.

## Product

- [ ] Full output contract is reproducible.
- [ ] Separate diagnostic layers exist.
- [ ] No context-free trust score is shown.
- [ ] Applicability and unresolved regions are visible.
- [ ] Review package includes complete provenance.
- [ ] UI does not reimplement scientific calculations.

## Four-developer engineering

- [ ] Contracts are versioned.
- [ ] Pull requests name consumed contract versions.
- [ ] Folder ownership is respected.
- [ ] Fixtures name schema and producer versions.
- [ ] No scientific report uses synthetic software fixtures.
- [ ] Every promoted output crosses its integration checkpoint.
- [ ] All lanes pass isolation, parity, provenance, and release checks.

## Honest release claim

Claims remain bounded by available evidence. Missing industrial validation, unavailable labels, absent evaluation targets, or limited expert review remain explicit limitations.

---

# Per-developer current-work rule

After Phase 4, each developer maintains:

- one active implementation objective;
- one active experiment or integration question;
- one pending gate;
- one runnable lane path.

No developer may silently change another lane's contract. Cross-lane changes require an ADR, migration impact, affected-owner review, and the smallest useful validation experiment.
