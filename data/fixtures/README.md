# Fixtures

Deterministic fixtures used by the smoke pipeline and unit tests. Generated
with a fixed seed; regenerate with:

```python
import numpy as np

rng = np.random.default_rng(0)
np.save("sample_8x8.npy", rng.random((8, 8), dtype=np.float32))
```

- `sample_8x8.npy` — 8x8 float32 array in `[0, 1)`, seed 0. Loaded by
  `scripts/smoke.py` to exercise the manifest → sample → preprocess → infer →
  evaluate → save artifacts → report path.
