"""Tests for package metadata consistency."""

import tomllib
from pathlib import Path

import veska


def test_package_version_matches_pyproject():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert veska.__version__ == pyproject["project"]["version"]
