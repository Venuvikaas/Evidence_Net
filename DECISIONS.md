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
