# Backlog

Ideas that are not part of the currently active evidence gate live here, per
the Current-Work Rule in `EXECUTION.md` (Part 6): one active implementation
objective, one experiment question, one pending evidence gate, and one runnable
promoted path at a time.

New ideas are added here or as an ADR proposal before work starts. A gate is
not interrupted unless new information invalidates its assumptions.

## Integration dependency (Lane B, Phase 7 -> Integration I)

- EXP-005 (measurement-consistency incremental value, Gate 6) needs Lane A's
  simple Benefit features (Phase 5) for the governed with/without comparison.
  The consistency feature set (`per_image_residuals` in
  `src/evidence_net/stress_tests/consistency.py`) is ready to consume.
