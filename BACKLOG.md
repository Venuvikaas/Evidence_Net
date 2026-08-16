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

## Integration dependency (Lane B, Phase 8 -> Integration II)

- EXP-006 (model-stability incremental value, Gate 7) needs Lane A's simple
  Benefit features (Phase 5) for the governed with/without comparison.
  The stability feature set (perturbation deviation, checkpoint agreement,
  error diversity in `src/evidence_net/stress_tests/stability.py`) is ready
  to consume; the diversity guard's included-model set is part of the
  review information.

## Integration dependency (Lane B, Phase 9 -> Integration II)

- EXP-007 (familiarity shift detection and rare-structure behavior, Gate 8)
  needs the real-mode run on the frozen calibration reference to decide the
  feature representation; the synthetic smoke suite reports the rare-valid
  false-warning cap mechanism. Real rare-valid cases (curated from frozen
  natural failures, Phase 10) will feed the gate. Familiarity integrates
  into warnings/abstention only after Gate 8 (Lane A policy).

## Lane A, Phase 6 -> Gate 5 (Integration I)

- EXP-010 (selective action, Gate 5) machinery is complete: the
  decision-policy-v1 contract, threshold fit on calibration/validation only
  with a frozen config, accept/attenuate/reject actions, the orthogonal
  unresolved mask (input edge density — the rejected-proposal-never-certifies-
  Base rule), action-map and coverage-risk reports, and the evaluation
  script (`scripts/measure_policy.py`, CI synthetic smoke). The governed
  real run at Integration I (with the trained predictor) decides the gate;
  the synthetic smoke validates the path end-to-end.

## Lane A, Phase 5 -> Gate 4 (Integration I)

- EXP-009 (benefit predictability, Gate 4) machinery is complete: deterministic
  labels (`support-definition-v1`), baselines (residual-magnitude,
  local-signal, attention gate), minimal two-stage predictor, calibration
  (`calibration-version-v1`, calibration-split-only fit), and the evaluation
  suite. The governed real run on the frozen calibration/validation splits
  (real Base/Proposal checkpoints) decides the gate; CI exercises the full
  path in synthetic smoke. Integration I (Benefit and policy promotion) is
  the first unblocked cross-lane checkpoint once Gate 4 passes.

## Integration dependency (Lane B, Phase 10 -> Integration III)

- EXP-008 (five structural-risk evidence categories, Gate 9) needs the
  governed real run on the frozen validation split (Base/Proposal
  checkpoints + oracle-patch proxy) to decide the gate. The candidate
  suite, ambiguity suite, acquisition artifacts, frozen hidden stress
  (`data/stress/hidden-stress-v1.json`), natural failure bank
  (`data/failures/natural-failures-v1.json`), and downstream task
  (`src/evidence_net/stress_tests/downstream.py`) are ready to consume;
  the downstream task must never be co-trained on the stress suite.
  Real-mode candidate/ambiguity/acquisition probes are labeled synthetic
  where applicable; no hallucination-resistance claim follows from any
  single suite.
