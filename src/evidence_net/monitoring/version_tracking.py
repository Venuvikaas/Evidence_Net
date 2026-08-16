"""Active semantic-version and decision-distribution tracking (Phase 17).

Tracks which support definitions, calibration mappings, policies, and
diagnostic versions are active, plus the running accept/attenuate/reject
and unresolved-area distributions. This is an operational signal: version
mismatches and unusual action distributions are review triggers, never
interpreted as proof of correctness.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VersionTracker:
    """Records the active semantic versions and action distributions."""

    active_versions: dict[str, str] = field(default_factory=dict)
    actions: Counter[str] = field(default_factory=Counter)
    n_patches: int = 0
    unresolved_patches: int = 0

    def register_version(self, component: str, version: str) -> None:
        """Register the active semantic version of a component."""
        self.active_versions[component] = version

    def record_action_map(self, actions: list[str], unresolved: list[bool] | None = None) -> None:
        """Record one sample's per-patch actions and unresolved flags."""
        self.actions.update(actions)
        self.n_patches += len(actions)
        if unresolved is not None:
            self.unresolved_patches += int(sum(unresolved))

    def get_summary(self) -> dict[str, Any]:
        total = max(self.n_patches, 1)
        return {
            "active_versions": dict(self.active_versions),
            "action_fractions": {
                action: round(count / total, 4) for action, count in self.actions.items()
            },
            "unresolved_fraction": round(self.unresolved_patches / total, 4),
            "n_patches": self.n_patches,
        }


# Global singleton instance
version_tracker = VersionTracker()
