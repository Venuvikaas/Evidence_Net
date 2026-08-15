# EVIDENCE-Net — Solo Developer Execution Framework & Mega Checklist

## One developer · No fixed deadline · one evidence-backed commit per ticked box

This file converts the EVIDENCE-Net final idea into an executable solo-development plan. It assumes ample time, but it does **not** assume unlimited attention, compute, or tolerance for rework. The sequence is intentionally strict: prove each scientific layer before building the next one, keep the command-line research path working before adding product surfaces, and remove components that fail their pre-declared evidence gates.

The plan is based on the product and scientific contract in `evidence_net_final_idea.md`. It preserves the project's full long-term ambition while preventing a single developer from simultaneously maintaining unfinished data, model, backend, frontend, deployment, and validation tracks.

The official dataset is expected to be available locally in two directories placed in the **same parent directory as this execution file**: `./train/` and `./Test_NoisyLR/`. The `train/` directory is the development data source. `Test_NoisyLR/` is treated as isolated evaluation input and must not be used for training, model selection, calibration, policy tuning, or repeated development feedback. The exact internal contents and pairing rules of `train/` must be discovered and recorded during Phase 1 rather than assumed.

---

## How to Read This File

- **[SOLO]** means the single developer owns and completes the task.
- **🔴 BLOCKING** means dependent work must not begin until the box and its exit gate pass.
- **🟠 RESEARCH GATE** means continue, redesign, or stop based on recorded evidence.
- **🔗 CONTRACT** means an interface or scientific definition that must be versioned before code relies on it.
- **🔁 CHECKPOINT** means run the relevant smoke, regression, and reproducibility checks before moving on.
- **🧪 EXPERIMENT** means the task must produce a configuration, run record, metrics, artifacts, and conclusion.
- **📦 RELEASE GATE** means tag a stable internal milestone that can be reproduced later.
- Every implementation box ends with a suggested Conventional Commit.
- Tick a box only when its output, tests, and documentation exist. A partially working notebook is not completion.
- No calendar durations are assigned. Work is ordered by dependency and evidence, not by dates.

---

# PART 1 — Solo-Developer Operating System

## 1. Work sequentially by risk, not by software layer

The riskiest assumptions are scientific:

1. The data and targets are valid.
2. A credible Base Reconstruction can be trained.
3. A bounded proposal creates useful oracle-gating headroom.
4. Proposal benefit can be predicted better than trivial heuristics.
5. Calibration survives held-out groups.
6. Extra diagnostics add value.
7. Abstention lowers risk.

Do not build a polished React application, production database, TensorRT path, or monitoring stack before these assumptions survive their gates. The solo developer should keep only one primary track active at a time.

## 2. Maintain one runnable vertical slice

From the first baseline onward, preserve a thin end-to-end path:

```text
manifest -> sample -> preprocess -> infer -> evaluate -> save artifacts -> generate report
```

Later phases extend this same path with proposal, benefit probability, diagnostics, decisions, abstention, API, and UI. Do not create isolated subsystems that cannot run through the common pipeline.

## 3. Separate exploration from governed experiments

Use notebooks only for exploration. Any result referenced in a decision must be reproducible through a script or configured command.

Required experiment bundle:

```text
runs/<run_id>/
  config.yaml
  manifest.json
  environment.txt
  metrics.json
  summary.md
  artifacts/
  logs/
  checkpoint-or-reference.txt
```

A result does not count if its configuration, data manifest, seed policy, code commit, and output artifacts cannot be recovered.

## 4. Freeze semantic contracts before training against them

The following are versioned contracts:

- Dataset and source-group contract.
- Modality contract.
- Base output and proposal tensor contract.
- Structural-change metric definitions.
- Proposal-benefit event.
- Calibration contract.
- Measurement-consistency output.
- Stability output.
- Familiarity output.
- Decision and abstention policy.
- API artifact schema.

Changing a contract requires a decision-log entry, version increment, migration note, and rerun decision for affected experiments.

## 5. Git workflow for one developer

- Use trunk-based development on `main`.
- Create short-lived branches only for risky refactors or experiments that may be discarded.
- Commit one coherent change per checked box.
- Tag evidence gates and reproducible internal releases.
- Use Conventional Commits with scopes such as `data`, `model`, `proposal`, `benefit`, `calibration`, `forward`, `stability`, `familiarity`, `decision`, `eval`, `api`, `ui`, `deploy`, `docs`, `test`, and `chore`.
- Do not commit datasets, secrets, generated checkpoints, or large artifacts unless the repository policy explicitly supports them.
- Keep `DECISIONS.md`, `EXPERIMENTS.md`, and `FAILURES.md`. Negative results are project assets.

## 6. Context-switching rule

The solo developer should not alternate casually between research, backend, and UI within the same milestone.

Preferred sequence:

1. Define contract.
2. Implement CLI/library path.
3. Test it.
4. Run governed experiments.
5. Record decision.
6. Integrate into the vertical slice.
7. Only then expose it through API and UI.

If a product surface needs future data, use a frozen fixture generated from the output contract. Do not duplicate scientific logic in the frontend.

## 7. Reproducibility before optimization

Before tuning a model heavily:

- Confirm two runs under the stated determinism policy are comparable.
- Save dataset and configuration hashes.
- Record hardware and software environment.
- Verify resume-from-checkpoint behavior.
- Verify evaluation does not alter model state.
- Make training interruption recoverable.

## 8. Decision log format

Every major choice gets a short entry:

```markdown
## ADR-XXX — Decision title
- Status: proposed | accepted | rejected | superseded
- Context:
- Decision:
- Evidence:
- Alternatives rejected:
- Consequences:
- Contracts or experiments affected:
```

## 9. Experiment ledger format

`EXPERIMENTS.md` should record:

```markdown
## EXP-XXX — Hypothesis
- Question:
- Primary metric:
- Secondary diagnostics:
- Baselines:
- Dataset manifest:
- Configs:
- Acceptance rule:
- Result:
- Confidence interval / uncertainty:
- Decision:
- Artifact path:
```

Acceptance rules are written **before** examining final test results.

## 10. Failure archive

`FAILURES.md` contains:

- Failed model configurations.
- Data-quality incidents.
- Misleading metrics.
- Natural hallucination examples.
- Calibration failures.
- Operator misspecification cases.
- UI interpretation failures.
- Deployment parity failures.

Do not delete failed artifacts that explain later design decisions.

## 11. Compute and storage discipline

Ample project time does not justify uncontrolled runs.

- Start every model family with a tiny overfit test.
- Run smoke-scale experiments before full training.
- Define maximum storage retention by artifact class.
- Keep only promoted checkpoints, required comparisons, and failure exemplars.
- Record estimated and actual compute in experiment metadata.
- Stop runs that violate sanity checks or cannot answer the declared question.

## 12. The stop, redesign, and removal rule

A failed research gate is not a request to hide the result or add more complexity.

- If data validity fails, repair the foundation before modeling.
- If oracle gating has negligible headroom, redesign or remove the decomposition.
- If a benefit predictor does not beat simple heuristics, do not build product claims around it.
- If a diagnostic adds no held-out value, remove it.
- If abstention does not reduce measured risk, redesign the policy.
- If users systematically misinterpret a layer, rename or remove it.

## 13. Documentation is part of implementation

A phase is not complete until its scientific meaning, configuration, limitations, and reproduction command are documented. The README should always reflect the latest promoted vertical slice, not the most ambitious future architecture.

---

# PART 2 — Repository and Governance Setup

## Target Repository Layout

The local working directory is expected to look like this before Phase 1 begins:

```text
project-parent/
├── EVIDENCE_NET_EXECUTION.md
├── train/                 # official development data; never commit to Git
├── Test_NoisyLR/          # official isolated evaluation inputs; never commit to Git
└── evidence-net/          # source repository created in Phase 0
```

The repository itself uses:

```text
evidence-net/
├── README.md
├── EXECUTION.md
├── DECISIONS.md
├── EXPERIMENTS.md
├── FAILURES.md
├── CHANGELOG.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── data/
│   ├── modality/
│   ├── model/
│   ├── support_definition/
│   ├── calibration/
│   ├── decision_policy/
│   └── experiments/
├── data/
│   ├── manifests/
│   ├── fixtures/
│   └── README.md
├── src/evidence_net/
│   ├── data/
│   ├── models/
│   ├── decision/
│   ├── losses/
│   ├── evaluation/
│   ├── stress_tests/
│   ├── training/
│   ├── inference/
│   ├── reporting/
│   └── api/
├── frontend/
├── scripts/
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── integration/
│   ├── calibration/
│   ├── decision_parity/
│   └── regression/
├── runs/
├── artifacts/
└── docs/
    ├── architecture.md
    ├── modality-contract.md
    ├── data-card.md
    ├── model-card.md
    ├── support-definitions.md
    ├── calibration-card.md
    ├── forward-model-card.md
    ├── decision-policy.md
    ├── evaluation-protocol.md
    └── demo-script.md
```

---

# PART 3 — Mega Checklist

## ☐ PHASE 0 — Project Bootstrap and Contracts 🔴 BLOCKING

### Repository

- [ ] **[SOLO]** Initialize the repository and add the target folder layout. — commit: `chore: initialize evidence-net repository`
- [ ] **[SOLO]** Add `.gitignore` for environments, secrets, caches, datasets, runs, checkpoints, frontend dependencies, and build artifacts. — commit: `chore: add repository ignore rules`
- [ ] **[SOLO]** Add `.env.example` with empty, documented variables only. — commit: `chore: add environment template`
- [ ] **[SOLO]** Configure Python project metadata, formatting, linting, typing, and test commands. — commit: `chore: configure python development tooling`
- [ ] **[SOLO]** Add pre-commit checks for formatting, linting, accidental large files, and basic secret detection. — commit: `chore: add pre-commit quality checks`
- [ ] **[SOLO]** Create `DECISIONS.md`, `EXPERIMENTS.md`, `FAILURES.md`, and `CHANGELOG.md` with templates. — commit: `docs: add governance ledgers`
- [ ] **[SOLO]** Copy the final idea into `docs/product-definition.md` and link it from the README. — commit: `docs: add product definition`

### Scientific and data contracts

- [ ] **[SOLO]** 🔗 Write `docs/modality-contract.md` with current positioning, assumptions, known unknowns, and claim boundaries. — commit: `docs(modality): define initial modality contract`
- [ ] **[SOLO]** 🔗 Define the dataset manifest schema, source grouping, split labels, target uncertainty, and file hashing. — commit: `docs(data): define dataset manifest contract`
- [ ] **[SOLO]** 🔗 Define tensor conventions: dimensions, channel order, data type, value range, masks, and spatial alignment. — commit: `docs(model): define tensor contracts`
- [ ] **[SOLO]** 🔗 Define artifact naming and run-directory contracts. — commit: `docs: define run and artifact contracts`
- [ ] **[SOLO]** Write the first decision entries for initial modality scope, storage strategy, experiment tracking, and non-goals. — commit: `docs: record foundational decisions`

### Automation spine

- [ ] **[SOLO]** Add `scripts/check_env.py` to validate required packages, devices, directories, and configuration. — commit: `feat(chore): add environment check script`
- [ ] **[SOLO]** Add a minimal `scripts/smoke.py` that imports the package, loads a fixture, and writes a run bundle. — commit: `test: add initial smoke pipeline`
- [ ] **[SOLO]** Add CI for lint, type checks, unit tests, and smoke execution on fixtures. — commit: `ci: add core quality workflow`
- [ ] **[SOLO]** Write README setup and reproduction commands for the current skeleton. — commit: `docs: add initial runbook`

### 🔁 Phase 0 checkpoint

- [ ] Clean clone installs successfully.
- [ ] Environment check passes.
- [ ] Smoke command produces a valid run bundle.
- [ ] CI passes.
- [ ] All initial contracts and decisions are committed.

**Exit Phase 0:** the project is reproducible before research begins.

---

## ☐ PHASE 1 — Domain and Data Foundation 🔴 BLOCKING

### Official Local Dataset Intake 🔴 BLOCKING

Expected local paths, resolved relative to this execution file:

```text
./train/
./Test_NoisyLR/
```

`train/` is used to build the development manifests and, where the supplied structure permits, derive grouped training, validation, and calibration partitions. `Test_NoisyLR/` is isolated as evaluation input. It may be inventoried early for compatibility, but it must not influence model choice, loss design, feature selection, calibration, thresholds, abstention policy, or other development decisions.

- [ ] **[SOLO]** Verify that `EVIDENCE_NET_EXECUTION.md`, `train/`, and `Test_NoisyLR/` share the same parent directory; fail with a clear message if either dataset directory is missing. — commit: `feat(data): validate official local dataset paths`
- [ ] **[SOLO]** Add `TRAIN_DATA_DIR` and `TEST_NOISY_LR_DIR` to `.env.example`, using empty values or documented relative defaults, and keep machine-specific values in the uncommitted `.env`. — commit: `chore(data): configure official dataset paths`
- [ ] **[SOLO]** Add `train/`, `Test_NoisyLR/`, extracted copies, caches, generated previews, and temporary manifests to the applicable parent or repository ignore rules; verify Git does not stage dataset files. — commit: `chore(data): protect official dataset files`
- [ ] **[SOLO]** Implement `scripts/resolve_dataset_paths.py` so paths are resolved from the execution-file parent or explicit environment variables, never from an assumed current working directory. — commit: `feat(data): resolve local dataset directories`
- [ ] **[SOLO]** Generate a read-only inventory for `train/` containing relative path, extension, byte size, hash, readable status, dimensions, channels, data type, and numerical range where readable. — commit: `feat(data): inventory official train directory`
- [ ] **[SOLO]** Generate a separate read-only inventory for `Test_NoisyLR/` with the same non-target metadata and no model inference or score-based inspection. — commit: `feat(data): inventory isolated test directory`
- [ ] **[SOLO]** Inspect and document the internal folder and filename structure of `train/`; identify candidate noisy-input and target relationships only from observed names, metadata, and supplied documentation. Do not invent pairing rules. — commit: `docs(data): document official train structure`
- [ ] **[SOLO]** Implement the discovered `train/` pairing rule as a versioned adapter and report unmatched, duplicated, or ambiguous files. — commit: `feat(data): implement official train pairing adapter`
- [ ] **[SOLO]** Inspect and document the internal structure and required output naming for `Test_NoisyLR/`. Record explicitly whether targets are absent, inaccessible, or separately governed. — commit: `docs(data): document isolated test structure`
- [ ] **[SOLO]** Create separate immutable source manifests: `official-train-source-v1.json` and `official-test-noisylr-source-v1.json`. Keep the test manifest free of development labels and metrics. — commit: `chore(data): freeze official source manifests`
- [ ] **[SOLO]** Compare `train/` and `Test_NoisyLR/` compatibility for extension, dimensions, channels, data type, and raw ranges without using test outcomes to tune the system. — commit: `feat(data): verify train test input compatibility`
- [ ] **[SOLO]** Add an automated isolation test that fails if any `Test_NoisyLR/` path enters a training, validation, calibration, hyperparameter-search, or threshold-selection manifest. — commit: `test(data): enforce test noisylr isolation`
- [ ] **[SOLO]** Record the official local dataset layout, access method, observed contents, provenance information available to the developer, and unresolved restrictions in `docs/data-card.md`. — commit: `docs(data): register official local dataset`

### 🟠 Research Gate 1A — Are the official directories correctly integrated?

Do not continue to learned-model development until:

- [ ] Both local directories resolve reproducibly from configuration.
- [ ] Neither directory is tracked by Git.
- [ ] `train/` has a validated, versioned inventory and pairing adapter.
- [ ] Ambiguous or missing training pairs are reported rather than silently skipped.
- [ ] `Test_NoisyLR/` has a separate immutable inventory.
- [ ] Automated checks prevent `Test_NoisyLR/` from entering development splits.
- [ ] Tensor and preprocessing contracts match the observed files.
- [ ] The data card distinguishes observed facts from unresolved assumptions.

### Data inventory and rights

- [ ] **[SOLO]** Inventory all candidate datasets with source, license, modality, pairing, resolution, and target meaning. — commit: `docs(data): inventory candidate datasets`
- [ ] **[SOLO]** Record which data is real, synthetic, simulated, averaged, or expert-selected. — commit: `docs(data): classify target provenance`
- [ ] **[SOLO]** Exclude or quarantine data without clear permitted use. — commit: `chore(data): quarantine unverified sources`

### Manifest and loaders

- [ ] **[SOLO]** Implement manifest models and schema validation. — commit: `feat(data): implement manifest schema`
- [ ] **[SOLO]** Implement deterministic sample discovery and hashing. — commit: `feat(data): add deterministic sample indexing`
- [ ] **[SOLO]** Implement dataset loaders that preserve raw values and metadata. — commit: `feat(data): add raw-preserving dataset loader`
- [ ] **[SOLO]** Add tests for dimensions, type, channel order, masks, and range preservation. — commit: `test(data): verify tensor contract`
- [ ] **[SOLO]** Add corrupted-file and unsupported-format handling. — commit: `fix(data): validate invalid image inputs`

### Pairing, alignment, and leakage

- [ ] **[SOLO]** Implement exact pairing checks and report missing or duplicate partners. — commit: `feat(data): add pair integrity audit`
- [ ] **[SOLO]** Quantify alignment error with a documented method. — commit: `feat(data): add alignment audit`
- [ ] **[SOLO]** Store target-alignment uncertainty in the manifest. — commit: `feat(data): record target uncertainty`
- [ ] **[SOLO]** Detect exact and near duplicates. — commit: `feat(data): add duplicate detection`
- [ ] **[SOLO]** Define the source-group hierarchy that prevents leakage across repeated structures or acquisition sessions. — commit: `docs(data): define leakage-safe grouping`
- [ ] **[SOLO]** From `train/` only, implement deterministic model-training, validation, and calibration splitting by source group. Do not create a development split from `Test_NoisyLR/`. — commit: `feat(data): implement grouped development splits`
- [ ] **[SOLO]** Reserve explicit source-held-out and degradation-held-out groups from `train/` for development-time robustness checks; preserve `Test_NoisyLR/` for final evaluation inference. — commit: `feat(data): add held-out development groups`

### Data audit report

- [ ] **[SOLO]** Generate raw-range, clipping, size, structure, and degradation summaries. — commit: `feat(data): generate dataset audit metrics`
- [ ] **[SOLO]** Generate alignment and duplicate reports with inspectable examples. — commit: `feat(data): export quality audit artifacts`
- [ ] **[SOLO]** Create `docs/data-card.md` with provenance, limitations, splits, target meaning, and uncertainty. — commit: `docs(data): publish initial data card`
- [ ] **[SOLO]** Freeze `dataset-manifest-v1` and record its hash. — commit: `chore(data): freeze dataset manifest v1`

### 🟠 Research Gate 1 — Is the dataset fit for supervised restoration?

- [ ] Predeclare minimum data-validity conditions in `EXPERIMENTS.md`.
- [ ] Review pairing, alignment, leakage, target uncertainty, and rights.
- [ ] Record one decision: **continue**, **repair**, **change scope**, or **use benchmark-only positioning**.
- [ ] If conditions fail, do not begin model comparison.

### 🔁 Phase 1 checkpoint

- [ ] `scripts/audit_dataset.py` reproduces the audit.
- [ ] Split manifests are immutable and hashed.
- [ ] Loaders pass unit and smoke tests.
- [ ] The data card states what the clean target actually represents.
- [ ] `train/` supplies only permitted development manifests.
- [ ] `Test_NoisyLR/` remains absent from training, validation, calibration, and policy-selection manifests.
- [ ] A dry-run loader can read every supported `Test_NoisyLR/` input without running final evaluation.

**Exit Phase 1:** the official local datasets are integrated, the development data foundation can support the claims that will be tested, and the final evaluation inputs remain isolated.

---

## ☐ PHASE 2 — Evaluation Harness and Classical Baselines 🔴 BLOCKING

### Metric contracts

- [ ] **[SOLO]** 🔗 Define primary and secondary metric implementations and spatial units. — commit: `docs(eval): define metric contracts`
- [ ] **[SOLO]** Implement PSNR, SSIM, MAE, edge displacement, structural error, and frequency diagnostics. — commit: `feat(eval): implement restoration metrics`
- [ ] **[SOLO]** Add image- and source-group bootstrap confidence intervals. — commit: `feat(eval): add grouped uncertainty estimates`
- [ ] **[SOLO]** Add tests using analytically simple fixtures and known perturbations. — commit: `test(eval): validate metric behavior`
- [ ] **[SOLO]** Ensure pixels are never reported as independent sample counts. — commit: `fix(eval): enforce grouped statistics`

### Classical reference path

- [ ] **[SOLO]** Implement deterministic resizing or reconstruction appropriate to the input contract. — commit: `feat(model): add deterministic reference reconstruction`
- [ ] **[SOLO]** Add at least one classical denoising or restoration reference suitable for the current benchmark. — commit: `feat(model): add classical restoration baseline`
- [ ] **[SOLO]** Implement common inference, artifact export, and evaluation interfaces. — commit: `feat(inference): add baseline inference pipeline`
- [ ] **[SOLO]** Generate comparison sheets with input, output, target, error, edges, and metrics. — commit: `feat(report): add restoration comparison report`
- [ ] **[SOLO]** Extend the smoke script through evaluation and report generation. — commit: `test: extend restoration smoke path`

### 📦 Internal Release Gate

- [ ] Tag `v0.1-data-eval` after a clean-clone reproduction.

**Exit Phase 2:** every later model can be judged through the same trusted harness.

---

## ☐ PHASE 3 — Learned Base Reconstruction 🔴 BLOCKING

### Training infrastructure

- [ ] **[SOLO]** Implement configuration loading and validation. — commit: `feat(training): add structured configuration`
- [ ] **[SOLO]** Implement a reusable trainer with checkpointing, resume, mixed precision option, and controlled seeds. — commit: `feat(training): add reproducible trainer`
- [ ] **[SOLO]** Add experiment bundle creation and environment capture. — commit: `feat(training): persist experiment provenance`
- [ ] **[SOLO]** Add tiny-batch overfit and single-step gradient tests. — commit: `test(training): add model sanity checks`
- [ ] **[SOLO]** Add early failure checks for NaN, exploding loss, invalid range, and empty batches. — commit: `fix(training): add numerical failure guards`

### Direct and Base models

- [ ] **[SOLO]** Implement a strong direct-restoration baseline. — commit: `feat(model): implement direct restoration baseline`
- [ ] **[SOLO]** Implement the candidate Base Reconstruction through the same interface. — commit: `feat(model): implement base reconstruction`
- [ ] **[SOLO]** Implement base losses with configurable pixel, structural, edge, and frequency terms. — commit: `feat(model): add base reconstruction losses`
- [ ] **[SOLO]** Test output dimensions, ranges, gradients, checkpoint restore, and tiled parity. — commit: `test(model): validate base reconstruction path`

### 🧪 Experiment series

- [ ] **[SOLO]** Predeclare baseline comparison and acceptance rules. — commit: `docs(experiment): register base comparison`
- [ ] **[SOLO]** Run smoke-scale learning experiments and reject broken configurations. — commit: `test(experiment): validate base training configs`
- [ ] **[SOLO]** Run governed classical, direct, and Base comparisons. — commit: `exp(model): compare restoration baselines`
- [ ] **[SOLO]** Characterize natural errors by lines, boundaries, periodic regions, isolated points, and rare structures. — commit: `docs(eval): catalogue base model failures`
- [ ] **[SOLO]** Record whether “lower intervention” is supported by measured behavior. — commit: `docs(decision): assess base reconstruction claim`

### 🟠 Research Gate 2 — Is the Base Reconstruction credible?

Continue only if:

- It is independently useful against the declared baselines.
- Its behavior is understood by structural group.
- Any lower-intervention claim is supported or removed.
- The evaluation harness exposes failures rather than only averages.

If it is merely weaker, retain the neutral name “Base Reconstruction” and redesign the objective before proceeding.

### 📦 Internal Release Gate

- [ ] Promote one model and tag `v0.2-base-reconstruction`.

**Exit Phase 3:** a frozen, reproducible Base model exists for proposal research.

---

## ☐ PHASE 4 — Bounded Detail Proposal and Oracle Study 🔴 BLOCKING

### Proposal implementation

- [ ] **[SOLO]** 🔗 Freeze the proposal tensor and structural-summary contract. — commit: `docs(proposal): define detail proposal contract`
- [ ] **[SOLO]** Implement the bounded proposal head and ungated candidate. — commit: `feat(proposal): implement bounded detail branch`
- [ ] **[SOLO]** Implement target residual generation with frozen Base outputs. — commit: `feat(proposal): generate residual targets`
- [ ] **[SOLO]** Implement magnitude, edge, multi-scale energy, and structural-change summaries. — commit: `feat(proposal): add structural effect metrics`
- [ ] **[SOLO]** Add tests proving gate zero returns Base and gate one returns candidate. — commit: `test(proposal): verify fusion identities`
- [ ] **[SOLO]** Add connected-component and edge-displacement checks for controlled fixtures. — commit: `test(proposal): validate structural summaries`

### Oracle tooling

- [ ] **[SOLO]** Implement oracle pixel and patch decisions from ground truth. — commit: `feat(eval): add oracle gating`
- [ ] **[SOLO]** Implement oracle coverage-risk and structural-impact reports. — commit: `feat(eval): report oracle headroom`
- [ ] **[SOLO]** Predeclare meaningful oracle-headroom criteria. — commit: `docs(experiment): register oracle study`

### 🧪 Experiment series

- [ ] **[SOLO]** Train the proposal against frozen Base outputs. — commit: `exp(proposal): train bounded detail proposer`
- [ ] **[SOLO]** Compare Base, ungated candidate, equal-capacity direct model, and oracle-gated outputs. — commit: `exp(proposal): measure oracle gating headroom`
- [ ] **[SOLO]** Evaluate proposal benefit and harm by structural and degradation group. — commit: `docs(eval): analyze proposal effects`
- [ ] **[SOLO]** Archive natural harmful proposals for later stress testing. — commit: `docs(failures): add natural proposal failures`

### 🟠 Research Gate 3 — Does the decomposition have value?

- [ ] Record one decision: **continue**, **redesign proposal**, **change spatial unit**, or **abandon gated decomposition**.
- [ ] Continue only if the oracle finds meaningful headroom beyond Base and equal-capacity direct restoration on declared outcomes.
- [ ] Do not begin the benefit predictor merely because the proposal looks visually interesting.

### 📦 Internal Release Gate

- [ ] Tag `v0.3-proposal-oracle` if the decomposition passes.

**Exit Phase 4:** the project has evidence that selective proposal acceptance is worth predicting.

---

## ☐ PHASE 5 — Proposal-Benefit Definition, Predictor, and Calibration 🔴 BLOCKING

### Semantic contract

- [ ] **[SOLO]** 🔗 Define `SupportDefinition-v1`: exact event, region size, utility, labels, population, and limitations. — commit: `docs(benefit): define proposal benefit event v1`
- [ ] **[SOLO]** Implement deterministic label generation and version its outputs. — commit: `feat(benefit): generate benefit labels`
- [ ] **[SOLO]** Test label behavior on known beneficial, harmful, and tied examples. — commit: `test(benefit): validate event labels`

### Simple baselines first

- [ ] **[SOLO]** Implement residual-magnitude predictor. — commit: `feat(benefit): add residual magnitude baseline`
- [ ] **[SOLO]** Implement local-signal heuristic predictor. — commit: `feat(benefit): add local signal baseline`
- [ ] **[SOLO]** Implement a reconstruction-trained attention gate baseline. — commit: `feat(benefit): add attention gate baseline`
- [ ] **[SOLO]** Implement discrimination, ranking, selective-risk, and calibration reports separately. — commit: `feat(eval): add benefit evaluation suite`

### Learned predictor

- [ ] **[SOLO]** Implement the minimal Proposal-Benefit Predictor using Base, proposal, and local input features. — commit: `feat(benefit): implement minimal benefit predictor`
- [ ] **[SOLO]** Train it separately from the proposal and Base models. — commit: `exp(benefit): train two-stage predictor`
- [ ] **[SOLO]** Add external-proposal evaluation using proposals not seen during predictor training. — commit: `feat(eval): test predictor transfer`

### Calibration

- [ ] **[SOLO]** 🔗 Define `CalibrationVersion-v1`, including independent unit and confidence-interval method. — commit: `docs(calibration): define calibration contract v1`
- [ ] **[SOLO]** Implement candidate calibration approaches and preserve pre-calibration scores. — commit: `feat(calibration): add score calibration`
- [ ] **[SOLO]** Generate reliability, Brier-style, and grouped calibration reports as applicable to the event. — commit: `feat(eval): report benefit calibration`
- [ ] **[SOLO]** Add regression tests preventing use of test data for calibration. — commit: `test(calibration): enforce split isolation`

### 🧪 Experiment series

- [ ] **[SOLO]** Predeclare primary benefit-prediction comparison. — commit: `docs(experiment): register benefit predictor study`
- [ ] **[SOLO]** Compare heuristics, attention gate, and calibrated predictor. — commit: `exp(benefit): compare benefit predictors`
- [ ] **[SOLO]** Evaluate by source, degradation, and structure group. — commit: `exp(benefit): evaluate held-out groups`
- [ ] **[SOLO]** Record ranking and calibration conclusions separately. — commit: `docs(decision): evaluate benefit predictor`

### 🟠 Research Gate 4 — Is proposal benefit predictably useful?

Continue only if the learned predictor:

- Beats declared simple heuristics on held-out data.
- Provides useful selective-risk ordering.
- Has calibration that is meaningful within a stated domain.
- Retains value on external proposal behavior or has its limitation explicitly bounded.

If not, simplify, redefine the event, redesign the proposal, or stop the support-aware claim.

### 📦 Internal Release Gate

- [ ] Tag `v0.4-benefit-calibration`.

**Exit Phase 5:** the primary probability has an explicit meaning and evidence.

---

## ☐ PHASE 6 — Decision Policy, Gating, and Abstention 🔴 BLOCKING

### Policy contract

- [ ] **[SOLO]** 🔗 Define action semantics for accept, attenuate, reject, and abstain. — commit: `docs(decision): define action contract`
- [ ] **[SOLO]** Define costs, critical regions, threshold selection, and unresolved semantics. — commit: `docs(decision): define policy objectives`
- [ ] **[SOLO]** Implement a versioned policy configuration. — commit: `feat(decision): add policy schema`

### Implementation

- [ ] **[SOLO]** Implement threshold-based accept and reject policy. — commit: `feat(decision): implement basic gating policy`
- [ ] **[SOLO]** Implement attenuation with documented mapping. — commit: `feat(decision): add proposal attenuation`
- [ ] **[SOLO]** Implement a separate unresolved mask. — commit: `feat(decision): add unresolved regions`
- [ ] **[SOLO]** Ensure rejected proposal does not automatically imply resolved Base output. — commit: `test(decision): preserve fallback uncertainty`
- [ ] **[SOLO]** Add action-map and coverage-risk reports. — commit: `feat(eval): report decision outcomes`

### 🧪 Experiment series

- [ ] **[SOLO]** Compare threshold and attenuation policies on validation and calibration data only. — commit: `exp(decision): compare selective policies`
- [ ] **[SOLO]** Freeze policy thresholds before final held-out evaluation. — commit: `chore(decision): freeze policy v1`
- [ ] **[SOLO]** Evaluate restoration, structural risk, coverage, and unresolved area. — commit: `exp(decision): evaluate policy on held-out data`

### 🟠 Research Gate 5 — Does selective action reduce risk?

Continue only if:

- Selective gating improves a pre-declared endpoint without unacceptable regressions.
- Abstention lowers measured risk at a usable coverage.
- The policy does not disguise unresolved Base errors.

### 📦 Internal Release Gate

- [ ] Tag `v0.5-selective-restoration`.

**Exit Phase 6:** the core EVIDENCE-Net loop works without optional diagnostics.

---

## ☐ PHASE 7 — Measurement-Consistency Diagnostic

### Forward-model contract

- [ ] **[SOLO]** 🔗 Document image-formation assumptions and operator family. — commit: `docs(forward): define forward model v1`
- [ ] **[SOLO]** Define parameter bounds, stochastic treatment, and operation-order representation. — commit: `docs(forward): specify operator semantics`
- [ ] **[SOLO]** List known non-identifiability cases and prohibited claims. — commit: `docs(forward): document compatibility limits`

### Implementation

- [ ] **[SOLO]** Implement deterministic parts of the bounded forward family. — commit: `feat(forward): implement deterministic operators`
- [ ] **[SOLO]** Implement stochastic evaluation with controlled sampling and variance reporting. — commit: `feat(forward): add stochastic consistency evaluation`
- [ ] **[SOLO]** Report residual distribution across operators rather than minimum only. — commit: `feat(forward): report operator compatibility distribution`
- [ ] **[SOLO]** Add analytical and fixture-based operator tests. — commit: `test(forward): validate degradation operators`

### Counterexamples and incremental value

- [ ] **[SOLO]** Build examples where different clean candidates re-degrade similarly. — commit: `feat(stress): add non-identifiability cases`
- [ ] **[SOLO]** Build operator-misspecification tests. — commit: `feat(stress): add forward misspecification suite`
- [ ] **[SOLO]** Compare benefit prediction and policy with and without forward features. — commit: `exp(forward): measure diagnostic contribution`

### 🟠 Research Gate 6 — Does consistency add held-out value?

- [ ] Retain the diagnostic only if it improves a declared held-out outcome or provides independently useful review information.
- [ ] Label it compatibility, never truth.
- [ ] Remove it from the predictor if it adds complexity without value.

**Exit Phase 7:** measurement consistency is either evidence-backed and bounded, or cleanly excluded.

---

## ☐ PHASE 8 — Model Stability Diagnostic

- [ ] **[SOLO]** 🔗 Define what stability measures and what it cannot imply. — commit: `docs(stability): define diagnostic contract`
- [ ] **[SOLO]** Implement invertible test-time perturbation comparisons. — commit: `feat(stability): add perturbation stability`
- [ ] **[SOLO]** Implement checkpoint snapshot comparisons. — commit: `feat(stability): add checkpoint stability`
- [ ] **[SOLO]** Train or integrate genuinely varied model candidates only when their error diversity can be measured. — commit: `feat(stability): add diverse model comparison`
- [ ] **[SOLO]** Implement pairwise error-diversity metrics. — commit: `feat(eval): measure model diversity`
- [ ] **[SOLO]** Compare incremental value after controlling for simple features. — commit: `exp(stability): evaluate stability contribution`

### 🟠 Research Gate 7 — Does stability add value?

- [ ] Keep only validated stability sources.
- [ ] Do not convert agreement into probability of truth.
- [ ] Remove costly ensembles that fail incremental-value tests.

**Exit Phase 8:** stability is a measured diagnostic, not confidence theater.

---

## ☐ PHASE 9 — Distribution Familiarity and Shift

- [ ] **[SOLO]** 🔗 Define feature representation, reference population, distance, threshold, and applicability. — commit: `docs(familiarity): define diagnostic contract`
- [ ] **[SOLO]** Implement the simplest familiarity baseline. — commit: `feat(familiarity): add reference distance baseline`
- [ ] **[SOLO]** Construct source, degradation, severity, and acquisition shifts. — commit: `feat(stress): add distribution shift suites`
- [ ] **[SOLO]** Evaluate detection and false warnings by shift group. — commit: `exp(familiarity): evaluate shift detection`
- [ ] **[SOLO]** Test rare valid defects separately from invalid unfamiliar inputs. — commit: `exp(familiarity): test rare structure behavior`
- [ ] **[SOLO]** Integrate familiarity into warnings or abstention only after validation. — commit: `feat(decision): apply familiarity policy`

### 🟠 Research Gate 8 — Does familiarity improve policy safety?

- [ ] Verify it detects declared shifts.
- [ ] Verify it does not systematically suppress rare valid structures.
- [ ] Bind calibration claims to the validated familiarity domain.

**Exit Phase 9:** shift warnings have a reproducible definition and measured behavior.

---

## ☐ PHASE 10 — Structural-Risk and Hallucination Test Program 🔴 BLOCKING FOR VALIDATED RELEASE

### Candidate manipulation

- [ ] **[SOLO]** Implement false-line insertion. — commit: `feat(stress): add false line perturbation`
- [ ] **[SOLO]** Implement real-line deletion. — commit: `feat(stress): add line deletion perturbation`
- [ ] **[SOLO]** Implement edge shifts. — commit: `feat(stress): add edge shift perturbation`
- [ ] **[SOLO]** Implement merge and split perturbations. — commit: `feat(stress): add topology perturbations`
- [ ] **[SOLO]** Implement false periodicity and defect-like point perturbations. — commit: `feat(stress): add periodic and point perturbations`

### Observation ambiguity

- [ ] **[SOLO]** Create clean-candidate pairs that map to similar observations. — commit: `feat(stress): add ambiguity generator`
- [ ] **[SOLO]** Measure whether policy confidence or action changes appropriately. — commit: `exp(stress): evaluate ambiguity behavior`

### Acquisition artifacts

- [ ] **[SOLO]** Implement modality-plausible pre-inference artifacts. — commit: `feat(stress): add acquisition artifacts`
- [ ] **[SOLO]** Separate these from candidate manipulations in all reports. — commit: `fix(report): separate stress threat models`

### Natural failure bank

- [ ] **[SOLO]** Curate unedited errors from frozen models. — commit: `docs(failures): curate natural failure bank`
- [ ] **[SOLO]** Add metadata, source run, local effect, and review notes. — commit: `feat(stress): index natural failures`

### Downstream consequence

- [ ] **[SOLO]** Select one available frozen downstream task or measurement. — commit: `docs(downstream): define validation task`
- [ ] **[SOLO]** Implement evaluation without co-training the downstream system. — commit: `feat(downstream): add frozen task evaluation`
- [ ] **[SOLO]** Measure decision changes by action and diagnostic group. — commit: `exp(downstream): evaluate selective restoration impact`

### Test isolation

- [ ] **[SOLO]** Freeze hidden final perturbation parameters. — commit: `chore(stress): freeze hidden test definitions`
- [ ] **[SOLO]** Ensure support training cannot read final stress definitions. — commit: `test(stress): enforce test isolation`

### 🟠 Research Gate 9 — Does the system address structural risk?

- [ ] Evaluate localization, coverage-risk, and downstream harm.
- [ ] Include failures and confidence intervals.
- [ ] Do not claim hallucination resistance from candidate manipulation alone.

### 📦 Internal Release Gate

- [ ] Tag `v0.6-structural-validation`.

**Exit Phase 10:** the trustworthiness claim is tested against distinct threat models.

---

## ☐ PHASE 11 — Unified Inference, Reporting, and Provenance 🔴 BLOCKING

- [ ] **[SOLO]** Implement the full inference pipeline from sample to all required artifacts. — commit: `feat(inference): assemble evidence-net pipeline`
- [ ] **[SOLO]** Add Base, proposal, candidate, diagnostics, decision, final, and unresolved outputs. — commit: `feat(inference): export complete output contract`
- [ ] **[SOLO]** Store model, dataset, support definition, calibration, forward model, and decision policy versions. — commit: `feat(provenance): record semantic versions`
- [ ] **[SOLO]** Implement structural-change and applicability warnings. — commit: `feat(report): add risk and applicability warnings`
- [ ] **[SOLO]** Generate Markdown and JSON review reports. — commit: `feat(report): generate review package`
- [ ] **[SOLO]** Add artifact hashes and deterministic references. — commit: `feat(provenance): hash run artifacts`
- [ ] **[SOLO]** Extend the smoke script through the full core loop. — commit: `test: add full-loop smoke test`
- [ ] **[SOLO]** Add golden-set regression checks. — commit: `test(regression): add promoted model golden set`

### 🔁 Phase 11 checkpoint

- [ ] One command produces the full output contract.
- [ ] Every artifact is traceable to exact semantic versions.
- [ ] Reports state what each probability and diagnostic means.
- [ ] The full-loop smoke test passes from a clean environment.

**Exit Phase 11:** the research system is usable without a web interface.

---

## ☐ PHASE 12 — Metadata Store and API

### Persistence

- [ ] **[SOLO]** Implement database models for manifests, versions, runs, artifacts, metrics, policies, and reviews. — commit: `feat(api): add metadata models`
- [ ] **[SOLO]** Add migrations and database initialization. — commit: `feat(api): add persistence migrations`
- [ ] **[SOLO]** Keep large tensors and images in artifact storage rather than database blobs. — commit: `feat(storage): add artifact references`

### API

- [ ] **[SOLO]** 🔗 Freeze API schemas from the output contract. — commit: `docs(api): define restoration api contract`
- [ ] **[SOLO]** Implement restoration creation and status retrieval. — commit: `feat(api): add restoration endpoints`
- [ ] **[SOLO]** Implement artifact and diagnostic retrieval. — commit: `feat(api): serve restoration artifacts`
- [ ] **[SOLO]** Implement support-definition, calibration, and policy metadata endpoints. — commit: `feat(api): expose semantic metadata`
- [ ] **[SOLO]** Implement comparison and stress-test endpoints. — commit: `feat(api): add comparison and stress routes`
- [ ] **[SOLO]** Add upload validation, internal filenames, size limits, and structured errors. — commit: `feat(api): secure input validation`
- [ ] **[SOLO]** Add API integration tests using frozen fixtures. — commit: `test(api): cover restoration workflow`
- [ ] **[SOLO]** Add a health endpoint that checks dependencies without leaking secrets. — commit: `feat(api): add safe health check`

### 🔁 Phase 12 checkpoint

- [ ] CLI and API produce equivalent run contracts.
- [ ] Failed inference does not corrupt run metadata.
- [ ] API errors do not expose paths, secrets, or raw tensors.

**Exit Phase 12:** the validated core is available through a stable service contract.

---

## ☐ PHASE 13 — Technical Review Interface

### Foundation

- [ ] **[SOLO]** Initialize the React and TypeScript frontend. — commit: `feat(ui): initialize review application`
- [ ] **[SOLO]** Generate typed client models from the frozen API contract. — commit: `feat(ui): add typed api client`
- [ ] **[SOLO]** Build loading, empty, error, and unavailable-diagnostic states. — commit: `feat(ui): add application states`

### Restoration Workspace

- [ ] **[SOLO]** Build synchronized input, Base, proposal, candidate, final, and target panes. — commit: `feat(ui): add synchronized restoration workspace`
- [ ] **[SOLO]** Add linked pan, zoom, coordinates, and region selection. — commit: `feat(ui): synchronize image inspection`
- [ ] **[SOLO]** Add positive and negative proposal views. — commit: `feat(ui): add intervention inspector`

### Reliability layers

- [ ] **[SOLO]** Add separate Proposal-Benefit, Measurement-Consistency, Stability, Familiarity, Decision, and Unresolved layers. — commit: `feat(ui): add reliability layers`
- [ ] **[SOLO]** Add legends that state exact meaning, calibration, and limitations. — commit: `feat(ui): add semantic legends`
- [ ] **[SOLO]** Avoid a default universal trust color map. — commit: `style(ui): use non-authoritative diagnostic encoding`

### Inspectors and dashboards

- [ ] **[SOLO]** Add pixel and patch inspector. — commit: `feat(ui): add local value inspector`
- [ ] **[SOLO]** Add policy threshold explorer against recorded evaluation data. — commit: `feat(ui): add policy explorer`
- [ ] **[SOLO]** Add ranking, calibration, selective-risk, worst-group, and downstream charts as separate views. — commit: `feat(ui): add reliability dashboard`
- [ ] **[SOLO]** Add ambiguity, natural failure, and stress-test browser. — commit: `feat(ui): add failure review workspace`
- [ ] **[SOLO]** Add report export and provenance panel. — commit: `feat(ui): add review package export`

### UI testing

- [ ] **[SOLO]** Add component tests for legends, missing diagnostics, and action states. — commit: `test(ui): cover reliability display states`
- [ ] **[SOLO]** Add end-to-end test from upload to report. — commit: `test(ui): add full review workflow`
- [ ] **[SOLO]** Test large-image navigation and aligned overlays. — commit: `test(ui): verify spatial alignment`
- [ ] **[SOLO]** Audit keyboard use, contrast, and non-color encodings. — commit: `fix(ui): improve review accessibility`

### 🔁 Phase 13 checkpoint

- [ ] The UI displays only backend-computed scientific values.
- [ ] Every diagnostic can be viewed separately.
- [ ] Missing or inapplicable calibration is visible.
- [ ] The Base view is always available for comparison.

**Exit Phase 13:** the product communicates the system without collapsing diagnostics into false certainty.

---

## ☐ PHASE 14 — Human Interpretation and Review Workflow

Because the developer is solo, formal participant recruitment and domain feedback are external dependencies. The code and protocol can still be prepared independently.

- [ ] **[SOLO]** Write a user-study protocol focused on interpretation, over-trust, and decision quality. — commit: `docs(research): add interpretation study protocol`
- [ ] **[SOLO]** Prepare success, ambiguity, failure, unfamiliarity, and abstention cases. — commit: `feat(research): prepare review study fixtures`
- [ ] **[SOLO]** Add consent and data-handling documentation where required. — commit: `docs(research): add study data handling`
- [ ] **[SOLO]** Implement anonymous review-event capture without raw-tensor logging by default. — commit: `feat(review): add human review records`
- [ ] **[SOLO]** Run a pilot on the protocol and fix confusing wording. — commit: `docs(research): refine study protocol`
- [ ] **[SOLO]** Conduct available expert or user evaluations and report limitations honestly. — commit: `docs(research): report interpretation findings`
- [ ] **[SOLO]** Rename, redesign, or remove layers that are systematically misunderstood. — commit: `fix(ui): address interpretation failures`

### 🟠 Research Gate 10 — Can users interpret the outputs correctly?

- [ ] Users can explain that proposal benefit is not physical proof.
- [ ] Users distinguish compatibility, stability, familiarity, and benefit.
- [ ] Abstention and unresolved Base output are understood.
- [ ] Any unavailable participant population is documented as a limitation, not simulated.

**Exit Phase 14:** product language has evidence from actual use or an explicit unresolved validation gap.

---

## ☐ PHASE 15 — Deployment, Optimization, and Decision Parity

### Packaging

- [ ] **[SOLO]** Add container build for the reference service. — commit: `feat(deploy): containerize reference service`
- [ ] **[SOLO]** Add local deployment configuration with explicit volumes and secrets. — commit: `feat(deploy): add local service stack`
- [ ] **[SOLO]** Document GPU and CPU compatibility. — commit: `docs(deploy): document runtime requirements`

### Export

- [ ] **[SOLO]** Export only promoted model components to ONNX. — commit: `feat(deploy): export promoted models to onnx`
- [ ] **[SOLO]** Compare tensor, spatial, ranking, calibration, action, and abstention outputs. — commit: `test(deploy): validate onnx decision parity`
- [ ] **[SOLO]** Add TensorRT only if deployment requirements justify it. — commit: `feat(deploy): add optional tensorrt runtime`
- [ ] **[SOLO]** Repeat full decision-parity validation after optimization. — commit: `test(deploy): validate optimized decision parity`

### Performance

- [ ] **[SOLO]** Benchmark model size, memory, throughput, and latency at declared resolutions. — commit: `perf: benchmark promoted pipeline`
- [ ] **[SOLO]** Validate tiled inference at boundaries and diagnostic alignment. — commit: `test(inference): validate tiled decision parity`
- [ ] **[SOLO]** Document conservative-only or reduced-diagnostic modes only if scientifically valid. — commit: `docs(inference): document supported runtime modes`

### 📦 Release Gate

- [ ] Tag `v0.7-deployment-candidate` after parity checks pass.

**Exit Phase 15:** optimization does not silently change scientific decisions.

---

## ☐ PHASE 16 — Security, Privacy, and Integrity Hardening

- [ ] **[SOLO]** Enforce local-first and retention configuration where required. — commit: `feat(security): add retention controls`
- [ ] **[SOLO]** Confirm raw tensors are excluded from standard logs. — commit: `test(security): prevent tensor logging`
- [ ] **[SOLO]** Add authenticated and role-aware access for shared deployments if used. — commit: `feat(security): add access control`
- [ ] **[SOLO]** Validate filename sanitization and decompression limits. — commit: `test(security): harden upload handling`
- [ ] **[SOLO]** Use controlled model-loading formats and trusted paths. — commit: `fix(security): restrict model loading`
- [ ] **[SOLO]** Pin production dependencies and run dependency and container scans. — commit: `chore(security): add dependency scanning`
- [ ] **[SOLO]** Add artifact and policy integrity verification. — commit: `feat(security): verify provenance hashes`
- [ ] **[SOLO]** Scan current Git history for secrets and large sensitive artifacts. — commit: `chore(security): audit repository history`
- [ ] **[SOLO]** Write a security and privacy operations note. — commit: `docs(security): add operations guidance`

**Exit Phase 16:** the product preserves the provenance and confidentiality commitments in the final idea.

---

## ☐ PHASE 17 — Monitoring and Post-Deployment Review

- [ ] **[SOLO]** Add service metrics for requests, failures, latency, memory, and artifact writes. — commit: `feat(monitoring): add service telemetry`
- [ ] **[SOLO]** Add data-health summaries for range and acquisition-statistic drift. — commit: `feat(monitoring): add input drift summaries`
- [ ] **[SOLO]** Add action and unresolved-area distributions. — commit: `feat(monitoring): track decision distributions`
- [ ] **[SOLO]** Track active semantic versions of support definitions, calibration, and policies. — commit: `feat(monitoring): track semantic versions`
- [ ] **[SOLO]** Add reviewed-case calibration and selective-risk reports. — commit: `feat(monitoring): add reliability review reports`
- [ ] **[SOLO]** Add deployment decision-parity monitoring. — commit: `feat(monitoring): monitor runtime parity`
- [ ] **[SOLO]** Document alert meanings and prohibit unsupported interpretations of score drift. — commit: `docs(monitoring): define operational signals`

**Exit Phase 17:** monitoring observes both software health and the validity boundary of the model.

---

## ☐ PHASE 18 — Final Validation and Release

### Final untouched evaluation

- [ ] **[SOLO]** Freeze code, model, semantic contracts, calibration, and policy for evaluation. — commit: `chore(release): freeze validation candidate`
- [ ] **[SOLO]** Run the frozen pipeline on every input in `Test_NoisyLR/` once under the final protocol, preserving original relative names in the submission/output manifest and making no post-run model or policy changes. — commit: `exp(release): run final test noisylr inference`
- [ ] **[SOLO]** Generate grouped confidence intervals and all primary endpoints. — commit: `docs(release): report final primary results`
- [ ] **[SOLO]** Publish failures, negative results, and limitations beside successes. — commit: `docs(release): publish limitations and failures`
- [ ] **[SOLO]** Confirm no post-test tuning was performed. — commit: `docs(release): record evaluation integrity`
- [ ] **[SOLO]** Verify one output exists for every supported `Test_NoisyLR/` input and that no unexpected extra files are present. — commit: `test(release): verify test noisylr output coverage`
- [ ] **[SOLO]** Validate output dimensions, data type, range, filename mapping, and required packaging against the documented evaluation contract. — commit: `test(release): validate evaluation output contract`
- [ ] **[SOLO]** Freeze hashes for the `Test_NoisyLR/` source manifest, final outputs, model, calibration, and decision policy in the release report. — commit: `docs(release): record final evaluation provenance`

### Documentation

- [ ] **[SOLO]** Finalize README from clone to inference and review. — commit: `docs: finalize project runbook`
- [ ] **[SOLO]** Finalize model, data, calibration, forward-model, support-definition, and decision-policy cards. — commit: `docs: finalize scientific cards`
- [ ] **[SOLO]** Finalize architecture and API documentation. — commit: `docs: finalize technical documentation`
- [ ] **[SOLO]** Finalize demo script with success, ambiguity, failure, and downstream cases. — commit: `docs: finalize demonstration narrative`
- [ ] **[SOLO]** Update CHANGELOG and migration notes. — commit: `docs: prepare release changelog`

### Release checks

- [ ] **[SOLO]** Run clean-clone installation and full smoke test. — commit: `test(release): pass clean environment smoke`
- [ ] **[SOLO]** Run unit, numerical, integration, calibration, regression, UI, security, and parity suites. — commit: `test(release): pass complete verification suite`
- [ ] **[SOLO]** Verify required artifacts and semantic versions appear in every run. — commit: `test(release): verify output completeness`
- [ ] **[SOLO]** Verify no secrets or prohibited data are in repository history or release assets. — commit: `chore(release): complete security review`
- [ ] **[SOLO]** Tag the validated release. — commit: `chore(release): prepare validated release`

---

# PART 4 — Definition of Done

EVIDENCE-Net is not complete merely because it produces visually attractive reconstructions. A validated release requires all of the following.

## Scientific foundation

- [ ] Dataset provenance, pairing, alignment, grouping, and target uncertainty are documented.
- [ ] `train/` and `Test_NoisyLR/` are registered as separate official local sources with immutable manifests.
- [ ] `Test_NoisyLR/` never enters training, validation, calibration, model selection, or policy tuning.
- [ ] Train, validation, calibration, and untouched test sets are separate at the source-group level.
- [ ] The Base Reconstruction is independently competitive.
- [ ] Any lower-intervention claim is measured rather than assumed.
- [ ] Oracle gating demonstrates meaningful headroom.
- [ ] The Proposal-Benefit Predictor beats declared simple heuristics on held-out data.
- [ ] Ranking and calibration are measured separately.
- [ ] Every displayed probability names its exact event and calibration domain.

## Selective restoration

- [ ] Accept, attenuate, reject, and abstain have versioned meanings.
- [ ] Rejecting detail does not imply that the Base output is resolved.
- [ ] Abstention reduces measured risk at a declared coverage.
- [ ] The final policy improves a pre-declared endpoint without unacceptable regressions.

## Diagnostics

- [ ] Measurement consistency is labeled compatibility, not truth.
- [ ] Model agreement is labeled stability, not correctness.
- [ ] Distribution familiarity has a defined representation and validation suite.
- [ ] Every optional diagnostic demonstrates incremental held-out value or is removed.

## Structural and downstream validation

- [ ] Candidate manipulation, observation ambiguity, acquisition artifact, natural failure, and downstream consequence suites are distinct.
- [ ] Hidden final stress parameters are isolated from training.
- [ ] Structural false positives and false negatives are reported.
- [ ] At least one available downstream evaluation is completed, or its absence is clearly stated as a release limitation.

## Product integrity

- [ ] The full output contract is generated in one run.
- [ ] Separate reliability layers are available.
- [ ] No context-free global trust score is presented.
- [ ] Calibration applicability and unresolved regions are visible.
- [ ] A complete provenance and review package is exportable.
- [ ] Human interpretation has been tested where participants are available, with limitations recorded.

## Engineering quality

- [ ] Clean clone to smoke test works from the README.
- [ ] CI and the full test suite pass.
- [ ] Runs contain config, manifest, environment, metrics, logs, artifacts, and conclusions.
- [ ] Checkpoint resume and inference are reproducible under the declared policy.
- [ ] Deployment runtimes preserve numerical, spatial, ranking, decision, and abstention behavior.
- [ ] No secrets, restricted data, or untrusted checkpoints are committed.

## Honest release claim

The release may claim only what the final evidence supports. If industrial modality validation, downstream labels, or expert user studies are unavailable, the release remains a research platform for semiconductor-like structural imagery and states those gaps explicitly.

---

# PART 5 — Solo Developer Dependency Map

```text
Phase 0  Repository and contracts
   |
Phase 1  Data validity
   |
Phase 2  Evaluation harness
   |
Phase 3  Base Reconstruction
   |
Phase 4  Detail Proposal + oracle headroom
   |
Phase 5  Benefit prediction + calibration
   |
Phase 6  Decision + abstention
   |
   +--> Phase 7  Measurement consistency
   +--> Phase 8  Model stability
   +--> Phase 9  Distribution familiarity
              |
Phase 10 Structural and downstream validation
   |
Phase 11 Unified inference and reporting
   |
Phase 12 API and persistence
   |
Phase 13 Review UI
   |
Phase 14 Human interpretation
   |
Phase 15 Deployment and optimization
   |
Phase 16 Security hardening
   |
Phase 17 Monitoring
   |
Phase 18 Final validation and release
```

Phases 7, 8, and 9 are conceptually optional diagnostics, but a solo developer should still implement and judge them **one at a time**, not in parallel. Each must pass its own incremental-value gate before being integrated into the decision policy.

---

# PART 6 — Current-Work Rule

At any moment, maintain only:

- **One active implementation objective.**
- **One active experiment question.**
- **One pending evidence gate.**
- **One runnable promoted path.**

When a new idea appears, place it in `BACKLOG.md` or an ADR proposal. Do not interrupt the current gate unless the new information invalidates its assumptions.

The project's guiding execution rule is:

> Build the smallest reproducible layer that can falsify the next assumption. Preserve it as a working vertical slice. Add complexity only after simpler alternatives fail and held-out evidence shows that the new component earns its place.
