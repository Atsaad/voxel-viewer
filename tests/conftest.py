"""Shared pytest fixtures for voxel-viewer tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_csv() -> Path:
    """Path to the standard test CSV (pipeline v6.1 format)."""
    return FIXTURES_DIR / "sample.csv"


@pytest.fixture()
def sample_legacy_csv() -> Path:
    """Path to a CSV using the old ``tag_value`` column name."""
    return FIXTURES_DIR / "sample_legacy.csv"


@pytest.fixture(autouse=True)
def _reset_colors():
    """Ensure the global colour scheme is pristine for every test."""
    from voxel_viewer.colors import reset_colors

    reset_colors()
    yield
    reset_colors()
