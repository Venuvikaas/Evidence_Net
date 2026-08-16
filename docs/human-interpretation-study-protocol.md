# EVIDENCE-Net Human Interpretation Study Protocol (Phase 14)

- **Version:** v1
- **Status:** frozen
- **Owner:** Developer C (in collaboration with Developer A and Developer B)
- **Governed by:** Research Gate 10 criteria

---

## 1. Objective

This protocol governs expert user studies evaluating human interpretation of EVIDENCE-Net outputs. The goal is to ensure technical reviewers accurately distinguish between:
1. **Base Reconstruction ($b$):** Deterministic, lower-intervention baseline.
2. **Bounded Detail Proposal ($d$):** High-frequency candidate detail.
3. **Proposal Benefit Predictor & Diagnostic Layers:** Expected support, measurement consistency, model stability, and distribution familiarity.
4. **Final Gated Output ($o$):** Selective reconstruction under policy $G$.

---

## 2. Research Gate 10 Criteria

> [!IMPORTANT]
> **Research Gate 10 Requirement**
> Reviewers must distinguish Proposal Benefit, compatibility, stability, familiarity, rejection, and unresolved output without treating them as physical proof of unobserved features.

### Success Criteria:
- Reviewers achieve $\ge 90\%$ accuracy distinguishing Base Reconstruction from Bounded Detail Proposal interventions.
- Reviewers correctly identify `"not-defined"` or unconfident regions instead of assuming physical reality.
- Zero confusion between diagnostic confidence scores and physical ground truth.

---

## 3. Study Procedure

1. **Orientation & Consent:** Reviewers review study documentation and provide informed consent.
2. **Synchronized Inspection:** Reviewers inspect synchronized panes (Input, Base, Proposal, Ungated Candidate, Final Gated Reconstruction).
3. **Intervention Analysis:** Reviewers inspect positive ($d^+$) and negative ($d^-$) proposals.
4. **Diagnostic Verification:** Reviewers toggle reliability layers (Benefit, Consistency, Stability, Familiarity).
5. **Decision Capture:** Every review action (accept, reject, flag patch) is recorded via `POST /api/v1/review/events`.

---

## 4. Review Event Schema

Each captured event records:
- `run_id`: Unique run identifier
- `reviewer_id`: Anonymized reviewer identifier
- `action_type`: `accept_proposal` | `reject_proposal` | `inspect_patch` | `override_policy`
- `decision_reason`: Structured rationale
- `timestamp`: UTC ISO-8601 timestamp
