"""Tests for package metadata consistency."""

from pathlib import Path

import veska

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_package_version_matches_pyproject():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert veska.__version__ == pyproject["project"]["version"]
