"""Tests for voxel_viewer.converter module."""

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from voxel_viewer.converter import (
    _build_quad_faces,
    _build_triangle_faces,
    _build_vertices,
    compute_statistics,
    csv_to_gltf,
    csv_to_obj,
    csv_to_ply,
    read_csv_smart,
    validate_csv,
)


# ── validate_csv ─────────────────────────────────────────────────────

class TestValidateCsv:
    def test_valid(self):
        df = pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]})
        validate_csv(df)  # should not raise

    def test_missing_column(self):
        df = pd.DataFrame({"X": [1.0], "Y": [2.0]})
        with pytest.raises(ValueError, match="missing required"):
            validate_csv(df)

    def test_non_numeric(self):
        df = pd.DataFrame({"X": ["a"], "Y": [2.0], "Z": [3.0]})
        with pytest.raises(ValueError, match="must be numeric"):
            validate_csv(df)


# ── read_csv_smart ───────────────────────────────────────────────────

class TestReadCsv:
    def test_from_path(self, sample_csv):
        df = read_csv_smart(sample_csv)
        assert len(df) == 10
        assert "object_type" in df.columns

    def test_from_dataframe_passthrough(self):
        df = pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]})
        assert read_csv_smart(df) is df

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_csv_smart("/nonexistent.csv")

    def test_legacy_format(self, sample_legacy_csv):
        df = read_csv_smart(sample_legacy_csv)
        assert "tag_value" in df.columns

    def test_chunked(self, sample_csv):
        df = read_csv_smart(sample_csv, chunk_size=3)
        assert len(df) == 10


# ── Geometry helpers ─────────────────────────────────────────────────

class TestGeometry:
    def test_build_vertices_shape(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        verts = _build_vertices(coords, scale=1.0)
        assert verts.shape == (16, 3)  # 2 voxels × 8 corners

    def test_build_vertices_scale(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        verts = _build_vertices(coords, scale=2.0)
        assert verts.min() == pytest.approx(-1.0)
        assert verts.max() == pytest.approx(1.0)

    def test_quad_faces_shape(self):
        faces = _build_quad_faces(5)
        assert faces.shape == (30, 4)  # 5 × 6 quads

    def test_quad_faces_one_based(self):
        faces = _build_quad_faces(1)
        assert faces.min() >= 1

    def test_triangle_faces_shape(self):
        faces = _build_triangle_faces(3)
        assert faces.shape == (36, 3)  # 3 × 12 triangles

    def test_triangle_faces_zero_based(self):
        faces = _build_triangle_faces(1)
        assert faces.min() == 0
        assert faces.max() == 7


# ── compute_statistics ───────────────────────────────────────────────

class TestStatistics:
    def test_basic(self, sample_csv):
        df = read_csv_smart(sample_csv)
        stats = compute_statistics(df, "object_type")
        assert stats["total_voxels"] == 10
        assert "WallSurface" in stats["surface_counts"]
        assert stats["unique_surfaces"] >= 2

    def test_without_color_column(self):
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [3.0, 4.0], "Z": [5.0, 6.0]})
        stats = compute_statistics(df)
        assert stats["total_voxels"] == 2
        assert "surface_counts" not in stats


# ── OBJ export ───────────────────────────────────────────────────────

class TestObjExport:
    def test_creates_files(self, sample_csv, tmp_path):
        out = tmp_path / "test.obj"
        csv_to_obj(sample_csv, out, scale=0.5, use_colors=True)
        assert out.exists()
        assert out.with_suffix(".mtl").exists()

    def test_vertex_count(self, sample_csv, tmp_path):
        out = tmp_path / "test.obj"
        csv_to_obj(sample_csv, out, scale=0.5)
        content = out.read_text()
        n_vertices = content.count("\nv ") + (1 if content.startswith("v ") else 0)
        assert n_vertices == 10 * 8  # 10 voxels × 8 corners

    def test_no_color(self, sample_csv, tmp_path):
        out = tmp_path / "test.obj"
        csv_to_obj(sample_csv, out, use_colors=False)
        assert not out.with_suffix(".mtl").exists()

    def test_returns_stats(self, sample_csv, tmp_path):
        out = tmp_path / "test.obj"
        stats = csv_to_obj(sample_csv, out)
        assert stats["total_voxels"] == 10


# ── PLY export ───────────────────────────────────────────────────────

class TestPlyExport:
    def test_ascii(self, sample_csv, tmp_path):
        out = tmp_path / "test.ply"
        csv_to_ply(sample_csv, out)
        content = out.read_text()
        assert "ply" in content
        assert "element vertex 10" in content

    def test_binary(self, sample_csv, tmp_path):
        out = tmp_path / "test.ply"
        csv_to_ply(sample_csv, out, binary=True)
        header = out.read_bytes()[:200].decode("ascii", errors="ignore")
        assert "binary_little_endian" in header

    def test_no_color(self, sample_csv, tmp_path):
        out = tmp_path / "test.ply"
        csv_to_ply(sample_csv, out, use_colors=False)
        content = out.read_text()
        assert "property uchar red" not in content

    def test_legacy_format(self, sample_legacy_csv, tmp_path):
        out = tmp_path / "legacy.ply"
        csv_to_ply(sample_legacy_csv, out, use_colors=True)
        content = out.read_text()
        assert "element vertex 5" in content
        assert "property uchar red" in content


# ── glTF export ──────────────────────────────────────────────────────

class TestGltfExport:
    def test_creates_valid_glb(self, sample_csv, tmp_path):
        out = tmp_path / "test.glb"
        csv_to_gltf(sample_csv, out, scale=0.5)
        assert out.exists()
        data = out.read_bytes()
        # GLB magic = "glTF"
        magic = struct.unpack("<I", data[:4])[0]
        assert magic == 0x46546C67

    def test_returns_stats(self, sample_csv, tmp_path):
        out = tmp_path / "test.glb"
        stats = csv_to_gltf(sample_csv, out)
        assert stats["total_voxels"] == 10
