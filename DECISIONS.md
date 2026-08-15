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
