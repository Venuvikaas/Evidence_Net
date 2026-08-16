# Checkpoint Registry (Phase 4 handoff)

The two promoted Phase 4 checkpoints are **not** committed to Git
(see `.gitignore`: `*.pt`). They live under `checkpoints/` in a working
clone, and this registry pins them by sha256 so every lane reproduces the
same Phase 4 vertical slice. A lane may only use a checkpoint whose hash
matches this registry.

## Promoted checkpoints

| Checkpoint | Contract | Tag | sha256 |
| --- | --- | --- | --- |
| `checkpoints/train-base-gate2/best.pt` | `base-output-v1` | `v0.2-base-reconstruction` | `3e5d2f943448a0e763746f28cf277434df0422a134e623bb0940bc6b0170be33` |
| `checkpoints/train-proposal-gate3v2/best.pt` | `proposal-output-v1` | `v0.3-proposal-oracle` | `524156ed6ea71b60ffc361be8ec1efc88554903008e96349d63b062cab7978d2` |

Verify locally:

```bash
sha256sum checkpoints/train-base-gate2/best.pt \
          checkpoints/train-proposal-gate3v2/best.pt
```

## Reproduction commands

Requirements: `pip install -e ".[dev]"`, `pre-commit install`, datasets in
the project parent (`train/`, `Test_NoisyLR/`), and a GPU or CPU with enough
memory for the config.

### Base Reconstruction (`v0.2-base-reconstruction`, EXP-003)

```bash
# 1. Train (12 epochs, gate-2 config)
python scripts/train_base.py --config configs/model/base-gate2.yaml \
    --run-id train-base-gate2

# 2. Compare Base vs deterministic anchor vs classical vs direct
python scripts/compare_restoration.py --n-samples 12

# 3. Catalogue structural failures by region
python scripts/catalogue_failures.py --n-samples 12
```

Expected outcome (EXP-003): Base PSNR 25.21 dB vs deterministic anchor
25.08 dB, SSIM 0.639 vs 0.599, MAE 0.0399 vs 0.0430; direct CNN 22.60 dB;
failure catalogue periodic 0.096 / edge 0.084 / flat 0.030 MAE.

### Detail Proposal (`v0.3-proposal-oracle`, EXP-004)

```bash
# 1. Train the proposer against the frozen Base (12 epochs, gate-3 config)
python scripts/train_proposal.py --config configs/model/proposal-gate3.yaml \
    --base-checkpoint checkpoints/train-base-gate2/best.pt \
    --run-id train-proposal-gate3v2

# 2. Measure oracle headroom (pixel + patch gates)
python scripts/measure_oracle.py --n-samples 12

# 3. Analyze proposal effects by structural region
python scripts/analyze_proposal_effects.py --n-samples 12
```

Expected outcome (EXP-004): oracle patch MAE 0.0373 vs Base 0.0399 (-6.3%),
oracle patch PSNR 25.66 dB vs direct 22.60 dB, patch coverage 86.8%, edge
displacement not detectably increased; harm concentrates in periodic regions
(FAIL-001).

### Vertical slice without the dataset (CI smoke, synthetic)

```bash
python scripts/smoke.py
python scripts/train_proposal.py --config configs/model/proposal-smoke.yaml \
    --synthetic --run-id smoke-ci-proposal
```

## Registry change procedure

Changing a promoted checkpoint (retraining, re-export) requires a new
registry entry with a new hash, an ADR, and the `base-output-v1` /
`proposal-output-v1` contract-change procedure. Old hashes stay valid for
reproduction of past experiments.
