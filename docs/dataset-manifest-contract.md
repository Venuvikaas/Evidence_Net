# Dataset Manifest Contract v1 (schema)

- **Status:** Draft contract — frozen manifests are produced in Phase 1.
- **Version:** v1 (schema version; a new schema version requires a decision
  entry and migration note).
- **Governed by:** `EXECUTION.md` Phase 1.

This document defines the dataset manifest schema, source grouping, split
labels, target uncertainty, and file hashing rules. All manifest files live in
`data/manifests/` and are committed to Git as immutable, versioned contracts.

## 1. Manifest versions and naming

- Source manifests are frozen and immutable once created:
  - `official-train-source-v1.json` — the `train/` directory (development data).
  - `official-test-noisylr-source-v1.json` — the `Test_NoisyLR/` directory
    (isolated evaluation input; no development labels or metrics).
- Derived development split manifests (training / validation / calibration /
  held-out groups) are also immutable and hashed once produced.
- Any change to a frozen manifest requires a decision-log entry, a version
  increment, a migration note, and a rerun decision for affected experiments.

## 2. Schema

```json
{
  "manifest_version": "1",
  "dataset_id": "official-train-source-v1",
  "created_at": "ISO-8601 UTC timestamp",
  "source_root": "resolved absolute path of the dataset directory",
  "hash_algorithm": "sha256",
  "dataset_hash": "sha256 over the canonical manifest content",
  "provenance": {
    "source": "official local directory",
    "permitted_uses": ["development", "training", "validation", "calibration"],
    "restrictions": ["never commit to git", "isolated from Test_NoisyLR"]
  },
  "grouping": {
    "source_group_field": "highest-level acquisition/session identifier",
    "hierarchy": ["source_group", "acquisition", "sample"]
  },
  "files": [
    {
      "relative_path": "path relative to source_root, forward slashes",
      "extension": ".npy",
      "byte_size": 12345,
      "sha256": "hex digest",
      "readable": true,
      "dimensions": [64, 64],
      "channels": 1,
      "dtype": "float32",
      "range": [0.0, 1.0],
      "source_group": "group-id",
      "split_label": "train",
      "role": "input | target | input_and_target | unknown",
      "target_uncertainty": {
        "method": "not yet assessed",
        "estimate": null,
        "notes": ""
      }
    }
  ]
}
```

### Field rules

- `dimensions`, `channels`, `dtype`, and `range` are recorded only when the
  file is readable; otherwise `readable` is `false` and those fields are
  omitted.
- `split_label` is one of: `train`, `validation`, `calibration`,
  `heldout-source`, `heldout-degradation`, `test-final`.
- `test-final` labels may **only** appear on manifests derived from
  `Test_NoisyLR/`; automated checks fail if any `Test_NoisyLR/` path appears in
  a training, validation, calibration, hyperparameter-search, or
  threshold-selection manifest.

## 3. Source grouping

- The source-group hierarchy is the **highest** level that can leak repeated
  structures or correlated degradation across samples (for example repeated
  structures or acquisition sessions).
- `source_group` values are assigned from observed structure in `train/`
  during Phase 1 and documented; no grouping is assumed in Phase 0.
- Development splits are performed **by source group**, never by random pixel
  or file split that could leak across groups.

## 4. Split labels and separation

- `train`, `validation`, `calibration` are development-only and derived from
  `train/`.
- `heldout-source` and `heldout-degradation` are reserved development-time
  robustness groups, also from `train/`.
- `test-final` is the untouched final evaluation set derived from
  `Test_NoisyLR/`; it must never influence model choice, loss design, feature
  selection, calibration, thresholds, or abstention policy.

## 5. Target uncertainty

- Target uncertainty records how much confidence exists in the alignment and
  semantic meaning of the clean target for a sample.
- Method values: `not-yet-assessed`, `reported-by-source`, `estimated`,
  `expert-reviewed`.
- Uncertainty is stored per manifest (dataset level) and per file where
  measurable (Phase 1 alignment audit).

## 6. File hashing

- Hash algorithm: SHA-256 over raw file bytes.
- `dataset_hash`: SHA-256 over the canonical JSON serialization of the frozen
  manifest (sorted keys, no trailing whitespace) — this makes the manifest
  itself verifiable.
- Duplicate detection uses file hashes; near-duplicate detection is a Phase 1
  audit output, not part of this schema.

## 7. Isolation guarantees

- `Test_NoisyLR/` is inventoried read-only and must not influence any
  development decision.
- The automated isolation test (`test(data): enforce test noisylr isolation`)
  fails if any `Test_NoisyLR/` path enters a development manifest.
