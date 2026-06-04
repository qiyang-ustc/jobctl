"""Smoke tests: package importable and version is set."""
import jobctl


def test_version_is_string():
    assert isinstance(jobctl.__version__, str)


def test_version_is_semver_like():
    parts = jobctl.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
