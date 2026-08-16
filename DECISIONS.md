# Decision Log (DECISIONS.md)

Every major choice gets a short entry. Entries follow the format defined in
`EXECUTION.md` (Part 1, Section 8). Statuses: `proposed | accepted | rejected |
superseded`.

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

---

## ADR-001 — Initial modality scope
- Status: accepted
- Context: The final idea targets semiconductor-like structural imagery but
  requires one concrete imaging modality, acquisition process, and real
  validation dataset before any industrial positioning. The official local
  datasets (`train/`, `Test_NoisyLR/`) provide the development and isolated
  evaluation inputs for Phase 1.
- Decision: Position the first validated release as a research platform for
  semiconductor-like structural imagery. Do not market it as an industrial
  semiconductor solution until a concrete modality contract and real validation
  dataset are established.
- Evidence: Product definition (docs/product-definition.md), Section 3
  "Scope and Domain Positioning"; EXECUTION.md Research Gate 1.
- Alternatives rejected: Claiming a universal restoration solution without a
  registered modality contract; restricting to benchmark-only positioning
  before the official datasets are audited.
- Consequences: The modality contract is versioned and can be revised after
  Phase 1 data intake; claim boundaries stay explicit in docs.
- Contracts or experiments affected: docs/modality-contract.md.

## ADR-002 — Storage strategy
- Status: accepted
- Context: The project must stay reproducible while never committing datasets,
  secrets, checkpoints, or generated artifacts. The execution plan separates
  governed run bundles from repository content.
- Decision: Keep code, contracts, governance ledgers, and frozen manifest
  files in Git. Keep official datasets (`train/`, `Test_NoisyLR/`) outside the
  repository. Never commit generated runs, artifacts, checkpoints, or secrets
  (see `.gitignore`). Generated experiment state lives in `runs/<run_id>/`
  bundles.
- Evidence: EXECUTION.md Part 1 Sections 3, 5; Part 2 repository layout.
- Alternatives rejected: Committing datasets for convenience; storing generated
  artifacts in Git; external-only experiment tracking with no local run bundle.
- Consequences: A clean clone is small and reproducible; dataset access is
  documented in the data card and environment template.
- Contracts or experiments affected: docs/run-and-artifact-contract.md,
  docs/dataset-manifest-contract.md.

## ADR-003 — Experiment tracking
- Status: accepted
- Context: A result counts only if its configuration, data manifest, seed
  policy, code commit, and output artifacts can be recovered. The project must
  not depend on an external tracking service in Phase 0.
- Decision: Use file-based run bundles under `runs/<run_id>/` containing
  `config.yaml`, `manifest.json`, `environment.txt`, `metrics.json`,
  `summary.md`, `artifacts/`, `logs/`, and `checkpoint-or-reference.txt`.
  Each governed experiment is registered in `EXPERIMENTS.md` before it runs.
- Evidence: EXECUTION.md Part 1 Section 3 (required experiment bundle).
- Alternatives rejected: Notebook-only results; relying on an MLflow deployment
  before the science is proven; tracking only metrics without config and
  manifest.
- Consequences: Experiment infrastructure is available from Phase 0 onward and
  extended in later phases.
- Contracts or experiments affected: docs/run-and-artifact-contract.md.

## ADR-004 — Non-goals
- Status: accepted
- Context: The final idea explicitly limits what the system may claim. Without
  recorded non-goals, scope creeps and unearned claims enter releases.
- Decision: Record as non-goals: perfect recovery of destroyed information;
  zero hallucination; exact identification of hidden degradation order;
  physical proof that every accepted detail existed; a universally calibrated
  score across sensors or datasets; replacement of expert review for
  high-impact decisions.
- Evidence: Product definition, Section 3.3 "Non-goals".
- Alternatives rejected: Softening the non-goals to widen marketing claims.
- Consequences: Releases must state these gaps explicitly; diagnostics are
  labeled as compatibility/stability/familiarity, never truth.
- Contracts or experiments affected: docs/modality-contract.md, Phase 10 and 18
  gates in EXECUTION.md.

## ADR-005 — Phase 1 dataset decision
- Status: accepted
- Context: Research Gate 1 (EXECUTION.md) requires one decision — continue,
  repair, change scope, or use benchmark-only positioning — before learned
  model comparison begins.
- Decision: **continue** with supervised restoration on the official
  `train/` dataset. Begin the evaluation harness and learned-model work on
  the approved splits.
- Evidence: EXP-001. Pairing clean (3200/3200); 0 exact and 0 near
  duplicates; 100% readable; train/test input compatibility confirmed;
  Test_NoisyLR isolation enforced; target meaning and alignment uncertainty
  recorded in the source manifest and data card.
- Alternatives rejected: repair (no corrupted or unreadable files exist);
  change scope (degradation labels are absent, but none are required to
  proceed with supervised restoration); benchmark-only positioning (the
  official paired data is suitable for supervised development with the
  documented limitations).
- Consequences: Phase 2/3 model work may start on the frozen splits;
  degradation-held-out remains reserved (0 members) until degradation labels
  exist; alignment uncertainty is carried in `target_uncertainty` records and
  must be reflected in any alignment-sensitive metric claims.
- Contracts or experiments affected: dataset-manifest-v1, docs/data-card.md,
  docs/grouping-and-splits.md, EXP-001.

## ADR-006 — Phase 3 Base Reconstruction decision (Research Gate 2)
- Status: accepted
- Context: Research Gate 2 (EXECUTION.md Phase 3) requires the candidate
  Base Reconstruction to be independently useful against declared baselines,
  understood by structural group, and any lower-intervention claim supported
  or removed — before proposal research begins.
- Decision: **continue** with the Base Reconstruction
  (`b = U(y) + h_b(f(y))`, deterministic anchor + learned refinement) as the
  frozen Phase 3 model; promote `checkpoints/train-base-gate2/best.pt` and
  tag `v0.2-base-reconstruction`.
- Evidence: EXP-003, `runs/compare-gate2/`, `runs/catalogue-gate2/`,
  `docs/decision-base-reconstruction.md`. Base improves on the deterministic
  anchor (PSNR 25.21 vs 25.08 dB, SSIM 0.639 vs 0.599, MAE 0.0399 vs 0.0430)
  and beats classical (24.46 dB); equal-capacity direct CNN is much weaker
  (22.60 dB). Failure catalogue: periodic MAE 0.096, edge-band 0.084, flat
  0.030; worst samples reported individually.
- Alternatives rejected: redesigning the objective (the anchor+refinement
  earns its structure over an equal-capacity direct model); abandoning the
  lower-intervention claim (the deterministic-anchor construction bounds the
  refinement by design; the claim is stated with those bounds, not as
  automatic safety).
- Consequences: Phase 4's Bounded Detail Proposal must target the catalogue's
  worst regions (periodic, edge bands) to add value; the Base model remains
  the comparison floor and is re-evaluated through the same trusted harness
  in every later phase.
- Contracts or experiments affected: EXP-003, docs/base-failures.md,
  docs/decision-base-reconstruction.md, evaluation-protocol v1.

## ADR-007 — Phase 4 proposal-oracle decision (Research Gate 3)
- Status: accepted
- Context: Research Gate 3 (EXECUTION.md Phase 4) requires the gated
  decomposition to have measurable value before building a benefit
  predictor: the oracle must find meaningful headroom beyond the frozen Base
  and an equal-capacity direct model on declared outcomes.
- Decision: **continue** with the Base + bounded Detail Proposal + selective
  acceptance decomposition. EXP-004 conditions met: oracle patch MAE -6.3%
  vs Base (>= 5% bar), oracle patch PSNR 25.66 vs direct 22.60 dB, patch
  coverage 86.8% in [10%, 90%], edge displacement not detectably increased.
  Promote `checkpoints/train-proposal-gate3v2/best.pt` and tag
  `v0.3-proposal-oracle`.
- Evidence: EXP-004, `runs/oracle-gate3-20260815-205601/`,
  `runs/proposal-effects-20260815-205856/`,
  `docs/decision-proposal-oracle.md`. The first trained proposal was nearly
  inert; adding residual fidelity to the objective (per product definition
  10.3) produced the headroom, so the objective was redesigned, not the
  decomposition.
- Alternatives rejected: redesigning the proposal (passing gate), changing
  the spatial unit (patch unit matches the Phase 5 benefit event), and
  abandoning the gated decomposition (direct model is 3 dB below oracle).
- Consequences: Phase 5 builds the proposal-benefit predictor, starting with
  simple baselines and requiring calibration/selective-risk reports; the
  decision policy must be regional (periodic regions are high-risk, archived
  as FAIL-001); the oracle stays a study tool, never used at inference.
- Contracts or experiments affected: EXP-004, docs/proposal-contract.md,
  docs/proposal-failures.md, FAIL-001.

## ADR-008 — Accept the Phase 4 handoff and the four-developer integration protocol
- Status: accepted
- Context: Research Gate 3 recorded **continue** (ADR-007) and the Phase 4
  vertical slice is reproducible. The execution plan
  (`EVIDENCE_NET_EXECUTION_WITH_DATASET_4_DEVELOPERS.md`, Part 3) requires a
  frozen set of shared contracts, pinned checkpoints, published fixtures,
  lane ownership, and branch/PR/contract-change rules before four developers
  may work in parallel after Phase 4.
- Decision: Freeze the nine handoff contracts in `docs/contracts/`
  (`dataset-v1`, `tensor-v1`, `metrics-v1`, `artifacts-v1`, `base-output-v1`,
  `proposal-output-v1`, `structural-summary-v1`, `oracle-report-v1`,
  `error-and-optional-fields-v1`); pin the promoted Base
  (`checkpoints/train-base-gate2/best.pt`, sha256
  `3e5d2f94...`, tag `v0.2-base-reconstruction`) and Proposal
  (`checkpoints/train-proposal-gate3v2/best.pt`, sha256 `524156ed...`, tag
  `v0.3-proposal-oracle`) in `docs/handoff/checkpoint-registry.md`; register
  fixtures with schema and producer versions in
  `data/fixtures/manifest-v1.json`; add `CODEOWNERS` for lanes A, B, C, D;
  adopt the workflow and kill switches in `docs/four-developer-workflow.md`
  and `docs/kill-switches.md`; enforce the invariants mechanically with
  `scripts/verify_handoff.py` (run in CI) and `tests/unit/test_handoff.py`.
- Evidence: Phase 0-4 code, tests, manifests, EXP-001..004, ADR-005..007,
  tags `v0.1-data-eval`/`v0.2-base-reconstruction`/`v0.3-proposal-oracle`;
  handoff verification passes (contracts frozen, isolation and fixture
  checks green).
- Alternatives rejected: starting parallel lanes without a frozen contract
  set (breaks the no-silent-cross-lane-change rule); committing checkpoints
  or datasets to Git (violates ADR-002 storage strategy — hashes are pinned
  instead); using synthetic fixtures in scientific reports (prohibited by
  the fixture rules).
- Consequences: four lanes A/B/C/D may begin their phase sets in parallel;
  every PR names consumed contract versions and passes the handoff kill
  switch; contract changes require an ADR, version increment, migration
  note, affected-owner review, and rerun decision; `Test_NoisyLR/` remains
  isolated; Integration checkpoints I-V and Phase 18 remain joint gates.
- Contracts or experiments affected: all nine contracts in
  `docs/contracts/`; fixture registry `data/fixtures/manifest-v1.json`;
  checkpoint registry `docs/handoff/checkpoint-registry.md`; future EXP-005+.

## ADR-009 — Phase 5 benefit-prediction decision (Research Gate 4)

- Status: accepted
- Context: Research Gate 4 (EXECUTION.md Phase 5) requires the learned
  benefit predictor to beat declared simple heuristics on held-out data,
  provide useful selective-risk ordering, and calibrate within a stated
  domain. The governed real run
  (`runs/benefit-eval-gate4-real-v4/` on the frozen
  `train-base-gate2`/`train-proposal-gate3v2` checkpoints, 128 calibration
  + 128 validation samples, split isolation enforced) found that the
  benefit event is real but the norm: 79.4% of validation patches are
  beneficial (candidate patch MAE 0.0382 vs Base 0.0408, delta 0.0026,
  matching the Phase 4 oracle headroom), and no predictor discriminates it
  (pooled AUC: attention-gate 0.5910, local-signal 0.5513,
  minimal-predictor 0.5165, residual-magnitude 0.4940; gated error at 0.5
  coverage is worse than the ungated floor for every predictor).
- Decision: **simplify** — the per-patch benefit-prediction claim is not
  promoted. The event, labels (`labels-v1`), and calibration remain (ECE
  0.013 in-domain); the decision policy defaults to accept the candidate
  (better than Base on 79.4% of patches) with the unresolved mask as the
  only abstention channel. A harness bug (the proposal checkpoint's
  `forward` returns the candidate, double-adding `b + d`) was found and
  fixed during the run; documented in FAILURES.md.
- Evidence: EXP-009, `runs/benefit-predictor-gate4-real-v3/`,
  `runs/benefit-eval-gate4-real-v4/`.
- Alternatives rejected: redesigning the proposal head without new evidence
  (Gate 3 already recorded continue); promoting the attention gate as the
  benefit signal (still near chance, 0.591 pooled AUC, and gating adds no
  value); stopping the support-aware claim entirely (the event and
  calibration remain scientifically valid).
- Consequences: `support-definition-v1` event + `calibration-version-v1`
  remain; the predictor is retained as a reported diagnostic, not a gating
  input; Integration I consumes the event, labels, and calibration but not
  per-patch gating probabilities.
- Contracts or experiments affected: `support-definition-v1`,
  `calibration-version-v1`, EXP-009, EXP-010.

## ADR-010 — Phase 6 decision-policy decision (Research Gate 5)

- Status: accepted
- Context: Research Gate 5 requires selective action to improve a
  predeclared endpoint, abstention to lower risk at usable coverage, and
  the policy to never disguise unresolved Base errors. The governed real
  run (`runs/policy-eval-gate5-real/`, 128 validation samples, thresholds
  fit on the calibration split) found the declared accept/attenuate/reject
  bands are degenerate on real data: with benefit prediction at chance
  (EXP-009 / ADR-009) the calibrated probability is near-constant and the
  accept band is empty.
- Decision: **continue with simplified policy** — the promoted policy is
  default-accept + unresolved abstention: the candidate is emitted
  everywhere (it improves the endpoint over the frozen Base: PSNR 25.376
  vs 24.888 dB, MAE 0.0382 vs 0.0408, edge displacement 6.179 vs 6.655 px)
  and unresolved (high edge-density) patches fall back to the Base without
  certifying it. Abstention lowers measured edge displacement further
  (6.1767 px) at unresolved area 0.001 (below the declared cap). The
  probability-band variant is not promoted.
- Evidence: EXP-010, `runs/policy-eval-gate5-real/`.
- Alternatives rejected: promoting the probability-band policy on real data
  (degenerate fit); rejecting more aggressively (no evidence the proposal
  harms on a predictable subset — 79.4% of patches benefit).
- Consequences: `decision-policy-v1` retains the accept/attenuate/reject
  action semantics and unresolved mask; the frozen default-accept +
  unresolved-abstention configuration is the promoted v1; kill-switch rule
  (rejection never certifies the Base) stays regression-tested.
- Contracts or experiments affected: `decision-policy-v1`, EXP-010.

## ADR-011 — Phase 7 measurement-consistency decision (Research Gate 6)

- Status: accepted
- Context: Research Gate 6 requires the consistency diagnostic to add
  held-out value or independently useful review information, labeled
  compatibility never truth. The governed real run
  (`runs/consistency-gate6-real/`, 32 validation groups) reports
  per-operator residual MAE 0.034-0.041 with tight group CIs and a tiny
  stochastic spread (noisy-blur std 0.00006).
- Decision: **keep** the measurement-consistency diagnostic as a review
  layer reporting the operator-family residual distribution; it stays
  labeled compatibility (never truth) and is not a benefit-prediction
  input.
- Evidence: EXP-005, `runs/consistency-gate6-real/`.
- Alternatives rejected: removing it (provides the operator-family
  distribution the review workflow needs); promoting it as a correctness
  signal (prohibited by the forward-model contract).
- Consequences: `forward-model-v1` freezes at v1; the consistency layer is
  available to C (UI legend) and D (monitoring) as a separate diagnostic.
- Contracts or experiments affected: `forward-model-v1`, EXP-005.

## ADR-012 — Phase 8 model-stability decision (Research Gate 7)

- Status: accepted
- Context: Research Gate 7 requires stability to be measured agreement,
  never correctness. The governed real run (`runs/stability-gate7-real/`,
  32 validation groups; promoted Base best/last + direct checkpoints)
  reports max mean perturbation drift 0.01485 (flip-v), checkpoint
  agreement best-vs-last 0.0089 MAE, and measured error diversity with
  complementarity 0.9998+ across the diverse comparison.
- Decision: **keep** the stability diagnostic (perturbation + checkpoint +
  diversity) as a reported review layer; agreement is labeled stability,
  never probability of truth.
- Evidence: EXP-006, `runs/stability-gate7-real/`.
- Alternatives rejected: removing it (adds the review information arm);
  treating agreement as correctness (prohibited by `stability-v1`).
- Consequences: `stability-v1` freezes; stability is a separate UI layer
  and monitoring signal, not a gating input.
- Contracts or experiments affected: `stability-v1`, EXP-006.

## ADR-013 — Phase 9 familiarity decision (Research Gate 8)

- Status: accepted
- Context: Research Gate 8 requires the familiarity diagnostic to detect
  declared shifts without systematically suppressing rare valid
  structures. The governed real run (`runs/familiarity-gate8-real/`;
  reference = 64 calibration inputs) detected **none** of the declared
  shift groups (acquisition/severity/source detection 0.000) and flagged
  **100% of rare-valid structures** unfamiliar (predeclared false-warning
  cap 0.50 exceeded).
- Decision: **do not promote** the familiarity diagnostic as a gating or
  warning input. It fails the predeclared Gate 8 acceptance rule; it is
  retained only as a disabled-by-default reported diagnostic pending a
  redesigned representation/threshold, and it never gates actions or
  certifies outputs.
- Evidence: EXP-007, `runs/familiarity-gate8-real/`.
- Alternatives rejected: lowering the threshold or cap post-hoc to pass
  (post-hoc tuning is prohibited); promoting it despite the cap breach
  (systematically suppresses rare valid structures).
- Consequences: `familiarity-v1` remains a draft contract; C must show the
  layer only when enabled and always with the exact legend; D must not
  monitor it as a production signal; Integration II is not triggered for
  familiarity.
- Contracts or experiments affected: `familiarity-v1`, EXP-007.

## ADR-014 — Phase 10 structural-risk decision (Research Gate 9)

- Status: accepted
- Context: Research Gate 9 requires distinct candidate, ambiguity,
  acquisition, natural-failure, and downstream evidence with frozen hidden
  stress definitions. The governed real run (`runs/structural-gate9-real/`,
  32 validation groups) shows each category producing evidence on its own
  terms (e.g. false-line +4.25 px edge displacement in the candidate
  suite; ambiguity cases; acquisition input deltas; 5-case natural failure
  bank; downstream evaluated without co-training) and the hidden stress
  hash is frozen (`087d6c13...`).
- Decision: **continue** with the structural-risk program as the
  validated-release threat model; no hallucination-resistance claim follows
  from any single suite.
- Evidence: EXP-008, `runs/structural-gate9-real/`.
- Alternatives rejected: removing suites (each category is a distinct
  threat model); merging categories into one claim (prohibited by the
  contract).
- Consequences: `structural-risk-v1` freezes; Integration III publishes
  the reports; hidden tests remain isolated from training.
- Contracts or experiments affected: `structural-risk-v1`, EXP-008.
