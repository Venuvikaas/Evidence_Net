# Fixtures

Deterministic fixtures used by the smoke pipeline, unit tests, and the
four-developer handoff. All fixtures are registered in
[`manifest-v1.json`](manifest-v1.json).

## Fixture rules (four-developer handoff)

- Every fixture names its **schema version** (e.g. `tensor-v1`,
  `error-and-optional-fields-v1`) and **producer version** (the contract
  version and code commit that produced it) in `manifest-v1.json`.
- **Real fixtures** come from frozen Phase 4 outputs (frozen Base/Proposal
  checkpoints pinned in `docs/handoff/checkpoint-registry.md`) and are
  allowed for tests and reproduction.
- **Synthetic software-only fixtures** exist for future optional fields and
  error cases. **No scientific report may use synthetic software-only
  fixtures.**
- Regenerating a fixture with a different producer requires a new fixture
  version entry; never overwrite a frozen fixture silently.

## Registered fixtures

### `sample_8x8.npy` (synthetic, `tensor-v1`)

8x8 float32 array in `[0, 1)`, seed 0. Loaded by `scripts/smoke.py` to
exercise the manifest -> sample -> preprocess -> infer -> evaluate -> save
artifacts -> report path. Regenerate with:

```python
import numpy as np

rng = np.random.default_rng(0)
np.save("sample_8x8.npy", rng.random((8, 8), dtype=np.float32))
```

### `error-and-optional-fields-v1-example.json` (synthetic, `error-and-optional-fields-v1`)

Example payload for the `error-and-optional-fields-v1` contract: optional
fields present, absent (`not-defined`), and a structured error payload.
Exercised by `tests/unit/test_handoff.py`.
