"""Tests for voxel_viewer.binvox module."""

import struct
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from voxel_viewer.binvox import (
    BinvoxHeader,
    BinvoxModel,
    read_binvox,
    read_binvox_directory,
    binvox_summary,
    _parse_header,
    _decode_rle,
)


SAMPLE_DIR = Path(__file__).parent.parent / "tt"


def _make_binvox(tmp_path, filename="test.binvox", dim=4, translate=(0, 0, 0), scale=2.0, rle_data=None):
    """Helper to create a small binvox file for testing."""
    header = (
        f"#binvox 1\n"
        f"dim {dim} {dim} {dim}\n"
        f"translate {translate[0]} {translate[1]} {translate[2]}\n"
        f"scale {scale}\n"
        f"data\n"
    )
    if rle_data is None:
        # Create a simple pattern: first 8 empty, then 4 occupied, rest empty
        total = dim ** 3
        rle_data = bytes([0, 8, 1, 4, 0, total - 12])

    path = tmp_path / filename
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rle_data)
    return path


class TestBinvoxHeader:
    def test_voxel_size(self):
        h = BinvoxHeader(version=1, dims=(140, 140, 140), translate=(0, 0, 0), scale=70.0)
        assert abs(h.voxel_size - 0.5) < 0.01

    def test_voxel_size_asymmetric(self):
        h = BinvoxHeader(version=1, dims=(100, 200, 100), translate=(0, 0, 0), scale=100.0)
        assert h.voxel_size == 0.5  # scale / max(dims) = 100/200


class TestDecodeRLE:
    def test_simple(self):
        # 2 empty, 3 occupied, 3 empty = 8 total
        data = bytes([0, 2, 1, 3, 0, 3])
        result = _decode_rle(data, 8)
        expected = np.array([False, False, True, True, True, False, False, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_empty(self):
        data = bytes([0, 10])
        result = _decode_rle(data, 10)
        assert not result.any()

    def test_all_occupied(self):
        data = bytes([1, 8])
        result = _decode_rle(data, 8)
        assert result.all()


class TestReadBinvox:
    def test_synthetic(self, tmp_path):
        path = _make_binvox(tmp_path, dim=4)
        model = read_binvox(path)
        assert model.header.version == 1
        assert model.header.dims == (4, 4, 4)
        assert model.header.scale == 2.0
        assert model.n_voxels == 4

    def test_world_coords(self, tmp_path):
        path = _make_binvox(tmp_path, dim=4, translate=(100, 200, 300), scale=4.0)
        model = read_binvox(path)
        world = model.to_world_coords()
        # All coords should be within translate + scale range
        assert world[:, 0].min() >= 100
        assert world[:, 0].max() <= 104
        assert world[:, 1].min() >= 200
        assert world[:, 2].min() >= 300

    def test_to_dataframe(self, tmp_path):
        path = _make_binvox(tmp_path, dim=4)
        model = read_binvox(path)
        df = model.to_dataframe()
        assert "X" in df.columns
        assert "Y" in df.columns
        assert "Z" in df.columns
        assert "source_file" in df.columns
        assert len(df) == 4

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_binvox("/nonexistent/path.binvox")

    def test_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.binvox"
        bad.write_text("not a binvox file")
        with pytest.raises(ValueError, match="Not a valid binvox"):
            read_binvox(bad)


class TestReadBinvoxDirectory:
    def test_multiple_files(self, tmp_path):
        _make_binvox(tmp_path, "a.binvox", dim=4, translate=(0, 0, 0))
        _make_binvox(tmp_path, "b.binvox", dim=4, translate=(10, 0, 0))
        df, models = read_binvox_directory(tmp_path)
        assert len(models) == 2
        assert len(df) == 8  # 4 + 4, no overlap expected
        assert "source_file" in df.columns

    def test_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No binvox files"):
            read_binvox_directory(tmp_path)

    def test_deduplication(self, tmp_path):
        # Same translate = overlapping voxels
        _make_binvox(tmp_path, "a.binvox", dim=4, translate=(0, 0, 0))
        _make_binvox(tmp_path, "b.binvox", dim=4, translate=(0, 0, 0))
        df, models = read_binvox_directory(tmp_path, deduplicate=True)
        assert len(df) == 4  # Deduplicated: only 4 unique positions


class TestBinvoxSummary:
    def test_summary(self, tmp_path):
        _make_binvox(tmp_path, "a.binvox", dim=4)
        _make_binvox(tmp_path, "b.binvox", dim=4)
        _, models = read_binvox_directory(tmp_path)
        s = binvox_summary(models)
        assert s["file_count"] == 2
        assert s["total_voxels"] == 8
        assert len(s["files"]) == 2


@pytest.mark.skipif(not SAMPLE_DIR.exists(), reason="tt/ sample directory not present")
class TestRealSamples:
    """Integration tests using the real sample .binvox files in tt/."""

    def test_read_single(self):
        files = list(SAMPLE_DIR.glob("*.binvox"))
        assert len(files) > 0
        model = read_binvox(files[0])
        assert model.n_voxels > 0
        assert model.header.dims[0] > 0

    def test_read_all(self):
        df, models = read_binvox_directory(SAMPLE_DIR)
        assert len(df) > 0
        assert len(models) > 0
        # All models should have reasonable voxel sizes (~0.5m)
        for m in models:
            assert 0.1 < m.header.voxel_size < 2.0

    def test_combined_has_xyz(self):
        df, _ = read_binvox_directory(SAMPLE_DIR)
        assert "X" in df.columns
        assert "Y" in df.columns
        assert "Z" in df.columns
