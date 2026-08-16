# EVIDENCE-Net Security & Privacy Operations (Phase 16)

- **Version:** v1
- **Status:** frozen
- **Owner:** Developer D

---

## 1. Scope

This note records the security and privacy controls implemented for the
EVIDENCE-Net reference service, the repository audit findings, and the
operational rules operators must follow. It covers Phase 16 of the
execution plan: retention, tensor-logging prevention, upload hardening,
model-loading restrictions, dependency scanning, provenance integrity, and
repository auditing.

---

## 2. Implemented controls

### 2.1 Retention and local-first

- Raw uploaded inputs and generated artifacts live under `scratch/`
  (runtime uploads) and `runs/` (run bundles); both are gitignored.
- `data/metadata.db` stores metadata and artifact references only — large
  tensors and images are never stored as database blobs.
- Retention configuration is local-first: no telemetry leaves the host by
  default; operators who enable remote export must scope it to metadata
  (never raw tensors).

### 2.2 Tensor-logging prevention

- Standard logs never contain raw tensors or image arrays. The
  `security.hardening.redact_tensor_repr` helper truncates tensor
  representations before any logging path; regression-tested in
  `tests/decision_parity/test_security_controls.py`.
- Review events are captured anonymously with no raw-tensor payloads by
  default (Phase 14 protocol).

### 2.3 Upload and path hardening

- Filenames are sanitized (`sanitize_filename`), file types and sizes are
  validated (`validate_upload_meta`), and all reads are confined to the
  configured base directory (`validate_file_path`) — path traversal is
  rejected and regression-tested.

### 2.4 Model loading

- Model loading is restricted to trusted `.pt` checkpoint paths under the
  repository's `checkpoints/` tree with `weights_only=False` only for
  internal promoted checkpoints; the ONNX export path loads only files
  produced by `deploy/export_onnx.py`.

### 2.5 Dependency and container scanning

- Development dependencies are pinned in `pyproject.toml`; CI installs
  from the lockfile-equivalent constraints.
- The reference container builds from the pinned environment; a container
  scan (e.g. `docker scout` or Trivy) is recommended before any
  deployment beyond local use — no container image is published by CI.

### 2.6 Provenance integrity

- `security.integrity` computes and verifies SHA256 digests for model
  checkpoints, policies, contract manifests, and artifacts
  (`compute_sha256`, `verify_file_integrity`).
- The checkpoint registry (`docs/handoff/checkpoint-registry.md`) pins
  promoted checkpoint hashes; `scripts/verify_handoff.py` enforces this in
  CI.

### 2.7 Repository audit

- `train/` and `Test_NoisyLR/` are outside Git and gitignored; a
  detect-secrets pre-commit hook scans all committed content; content
  hashes in frozen data manifests are excluded as documented false
  positives.
- History audit: `git log --all --name-only` scans confirmed no datasets,
  secrets, or `.env` values in history; the pre-commit hook blocks future
  secret commits.

---

## 3. Operational rules

1. Never log raw tensors, full images, or `.env` values — use the
   redaction helper.
2. Keep `TRAIN_DATA_DIR` / `TEST_NOISY_LR_DIR` secrets out of committed
   files; use `.env` (gitignored) or container secrets.
3. Run the container scan and the secret scan before any deployment.
4. Treat version mismatches or action-distribution anomalies as review
   triggers (see `docs/operational-signals-and-monitoring.md`), never as
   proof of correctness.
5. Raw uploads are temporary by default; define a retention window before
   enabling remote persistence.

---

## 4. Known limitations

- Access control is role-aware only when shared deployment requires it;
  the default reference service is local-first without authentication.
- Dependency scanning runs in the repository/CI (pinned deps + secret
  scan); container image scanning is documented but not executed by CI
  because no image is published.
