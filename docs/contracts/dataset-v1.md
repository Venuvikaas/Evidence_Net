# Contract: dataset-v1

- **Name:** `dataset-v1`
- **Version:** v1
- **Status:** frozen
- **Frozen at:** Phase 4 handoff (Research Gate 3: continue, ADR-007)
- **Owner:** ALL lanes
- **Governed by:** `docs/dataset-manifest-contract.md`, `docs/data-card.md`,
  `docs/grouping-and-splits.md`

## Purpose

Fix the identity, provenance, and isolation rules of the official datasets so
every lane consumes the same immutable data.

## Frozen fields

1. **Sources.** `train/` is the only official development source
   (training, validation, calibration, policy selection, stress design, demo
   selection). `Test_NoisyLR/` is isolated final-evaluation input.
2. **Manifests.** Immutable, hashed, versioned source manifests live in
   `data/manifests/`:
   - `official-train-source-v1.json` — 3200 pairs (6400 files), sha256 per
     file, target-alignment uncertainty recorded.
   - `official-test-noisylr-source-v1.json` — 400 inputs, no development
     labels or metrics.
3. **Derived manifests.** `dataset-splits-v1.json` and
   `dataset-manifest-v1.json` are immutable and hashed; development splits
   (train 2551 / validation 328 / calibration 164 / heldout-source 157 /
   heldout-degradation 0 reserved) are grouped by source group, seed 0.
4. **Isolation.** No `Test_NoisyLR/` path may appear in any development
   manifest. `test-final` labels exist only on `Test_NoisyLR/`-derived
   manifests. Final `Test_NoisyLR/` inference begins only after the
   four-lane controlled freeze (Integration V / Phase 18).
5. **Never in Git.** Both dataset directories, extracted copies, caches, and
   previews stay outside the repository (see `.gitignore`).
6. **Compatibility.** Train and test inputs are compatible in extension,
   shape, channels, dtype, and range family (verified in EXP-001).

## Implementation references

- Manifests: `data/manifests/official-train-source-v1.json`,
  `data/manifests/official-test-noisylr-source-v1.json`,
  `data/manifests/dataset-splits-v1.json`,
  `data/manifests/dataset-manifest-v1.json`
- Code: `src/evidence_net/data/` (paths, inventory, pairing, audit, splits,
  manifests, loaders)
- Scripts: `scripts/inventory_dataset.py`, `scripts/audit_dataset.py`,
  `scripts/build_splits.py`, `scripts/verify_manifests.py`,
  `scripts/resolve_dataset_paths.py`, `scripts/validate_dataset_paths.py`
- Tests: `tests/unit/test_isolation.py`, `tests/unit/test_manifests.py`,
  `tests/unit/test_splits.py`

## Change procedure

A new schema or split policy requires a new version (`dataset-v2`), an ADR,
a migration note, a rerun decision for affected experiments, and affected-owner
review. Freezing a new manifest version never edits a frozen one.
