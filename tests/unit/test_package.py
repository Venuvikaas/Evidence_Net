"""Package-level import checks."""

import evidence_net


def test_version_is_semver_like() -> None:
    parts = evidence_net.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_subpackages_importable() -> None:
    import evidence_net.api  # noqa: F401
    import evidence_net.data  # noqa: F401
    import evidence_net.decision  # noqa: F401
    import evidence_net.evaluation  # noqa: F401
    import evidence_net.inference  # noqa: F401
    import evidence_net.losses  # noqa: F401
    import evidence_net.models  # noqa: F401
    import evidence_net.reporting  # noqa: F401
    import evidence_net.stress_tests  # noqa: F401
    import evidence_net.training  # noqa: F401
