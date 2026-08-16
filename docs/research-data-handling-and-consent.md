# EVIDENCE-Net Research Data Handling & Consent Governance (Phase 14)

- **Version:** v1
- **Status:** frozen
- **Owner:** Developer C

---

## 1. Overview

This document specifies data handling, privacy protection, and participant consent procedures for human interpretation studies evaluating EVIDENCE-Net.

---

## 2. Participant Privacy & Anonymization

1. **Reviewer IDs:** Reviewers are assigned pseudo-random identifiers (e.g. `expert-01`, `expert-02`). No PII is logged in the database.
2. **Review Event Data:** Action logs (`review_events` table) store only coordinate bounds, timing, action types, and structured notes.
3. **Data Retention:** Event logs are retained only within governed local database stores (`data/metadata.db`) and are excluded from public git commits.

---

## 3. Informed Consent Checklist

Before participating in a review session, experts verify:
- [x] Purpose of the evaluation has been explained.
- [x] Understanding that diagnostic overlays are statistical confidence bounds, not physical proof.
- [x] Agreement to logging of interaction events (zooms, toggles, approvals, rejections).
- [x] Right to terminate participation at any point.
