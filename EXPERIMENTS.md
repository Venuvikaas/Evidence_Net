# Experiment Ledger (EXPERIMENTS.md)

Governed experiments are registered here **before** they run. Acceptance rules
are written before examining final test results. The format is defined in
`EXECUTION.md` (Part 1, Section 9). Every experiment produces a run bundle in
`runs/<run_id>/`.

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

---

_No governed experiments registered yet. Phase 1 registers the dataset
validity conditions before any learned-model experiment begins._
