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

## EXP-001 — Official dataset validity for supervised restoration
- Question: Is the official `train/` dataset fit for supervised restoration
  (pairing, alignment, leakage, target meaning, rights)?
- Primary metric: pair integrity (unmatched + duplicated + ambiguous = 0),
  readable fraction = 1.0, exact and near-duplicate groups = 0.
- Secondary diagnostics: alignment phase distribution; target range;
  resolution ratio; train/test input compatibility; Test_NoisyLR isolation.
- Baselines: n/a (data-validity gate before model comparison).
- Dataset manifest: `official-train-source-v1.json`
  (hash c504b2dded0f3a04...) and `official-test-noisylr-source-v1.json`
  (hash aab75186e9a46982...).
- Configs: `scripts/audit_dataset.py` (align-sample 200, near-sample 0,
  fixed seeds 0/1/2/3); `scripts/build_splits.py --seed 0`.
- Acceptance rule (predeclared): **continue** if (1) 100% of files readable;
  (2) pairing clean (0 unmatched/duplicated/ambiguous); (3) 0 exact and 0
  near duplicates; (4) train and test inputs compatible in extension, shape,
  channels, dtype, and range family; (5) no Test_NoisyLR path in any
  development manifest; (6) target meaning and alignment uncertainty
  documented and recorded. **Repair / change scope** if any of (1)-(5)
  fails; **benchmark-only** if the pairing or target meaning cannot be
  trusted for supervised learning.
- Result: _pending — recorded after Research Gate 1 review._
