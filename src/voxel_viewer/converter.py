"""Core conversion logic — voxel CSV → OBJ / PLY / glTF.

All heavy geometry is built with NumPy broadcasting and written with
``np.savetxt`` for maximum throughput.  A 687 K-voxel dataset that took
minutes with ``df.iterrows()`` now completes in seconds.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple, Union

import numpy as np
import pandas as pd

from .colors import SURFACE_COLORS, build_dynamic_palette, detect_color_column, get_color_for_tag

# ── Constants ────────────────────────────────────────────────────────

REQUIRED_COLUMNS: set[str] = {"X", "Y", "Z"}
DEFAULT_CHUNK_SIZE: int = 100_000

# 8 corner offsets for a unit cube centred at the origin
_CUBE_OFFSETS = np.array(
    [
        [-0.5, -0.5, -0.5],  # 0  left  bottom back
        [+0.5, -0.5, -0.5],  # 1  right bottom back
        [+0.5, +0.5, -0.5],  # 2  right top    back
        [-0.5, +0.5, -0.5],  # 3  left  top    back
        [-0.5, -0.5, +0.5],  # 4  left  bottom front
        [+0.5, -0.5, +0.5],  # 5  right bottom front
        [+0.5, +0.5, +0.5],  # 6  right top    front
        [-0.5, +0.5, +0.5],  # 7  left  top    front
    ],
    dtype=np.float64,
)

# 6 quad faces (vertex indices within one cube)
_CUBE_QUADS = np.array(
    [
        [0, 1, 2, 3],  # back
        [4, 7, 6, 5],  # front
        [0, 3, 7, 4],  # left
        [1, 5, 6, 2],  # right
        [3, 2, 6, 7],  # top
        [0, 4, 5, 1],  # bottom
    ],
    dtype=np.int64,
)

# 12 triangles (for glTF — 6 faces × 2 triangles)
_CUBE_TRIANGLES = np.array(
    [
        [0, 1, 2], [0, 2, 3],  # back
        [4, 7, 6], [4, 6, 5],  # front
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
        [3, 2, 6], [3, 6, 7],  # top
        [0, 4, 5], [0, 5, 1],  # bottom
    ],
    dtype=np.int64,
)


# ── Validation / I/O ────────────────────────────────────────────────

def validate_csv(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if the DataFrame lacks X / Y / Z columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    for col in ("X", "Y", "Z"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"Column '{col}' must be numeric, got {df[col].dtype}"
            )


def read_csv_smart(
    source: Union[str, Path, TextIO, pd.DataFrame],
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Read CSV from a file path, open file handle, or pass-through DataFrame.

    When *chunk_size* is set the file is parsed in chunks and concatenated
    (helps with CSV-parser memory peaks on very large files).
    """
    if isinstance(source, pd.DataFrame):
        validate_csv(source)
        return source

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"CSV file is empty: {path}")

    if chunk_size:
        chunks = list(pd.read_csv(source, chunksize=chunk_size))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(source)

    validate_csv(df)
    return df


def _count_csv_lines(path: Union[str, Path]) -> int:
    """Fast line count (excludes header)."""
    count = -1
    with open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return max(count, 0)


# ── Statistics ───────────────────────────────────────────────────────

def compute_statistics(
    df: pd.DataFrame,
    color_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a dict of QA statistics for a voxel dataset."""
    coords = df[["X", "Y", "Z"]]
    stats: Dict[str, Any] = {
        "total_voxels": len(df),
        "bbox_min": coords.min().to_dict(),
        "bbox_max": coords.max().to_dict(),
        "centroid": coords.mean().to_dict(),
        "std_dev": coords.std().to_dict(),
    }
    if color_column and color_column in df.columns:
        stats["surface_counts"] = df[color_column].value_counts().to_dict()
        stats["unique_surfaces"] = int(df[color_column].nunique())
    return stats


# ── Geometry helpers (vectorised) ────────────────────────────────────

def _build_vertices(coords: np.ndarray, scale: float) -> np.ndarray:
    """(N,3) centres → (N×8, 3) cube-corner vertices via broadcasting."""
    offsets = _CUBE_OFFSETS * scale  # (8, 3)
    # (N,1,3) + (1,8,3) → (N,8,3)
    verts = coords[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    return verts.reshape(-1, 3)


def _build_colors(
    df: pd.DataFrame,
    color_column: Optional[str],
    use_colors: bool,
    vertices_per_voxel: int = 1,
) -> Optional[np.ndarray]:
    """Build a ``(N × vertices_per_voxel, 3)`` uint8 colour array."""
    if not use_colors or not color_column or color_column not in df.columns:
        return None
    tags = df[color_column].values
    colors = np.array([get_color_for_tag(t) for t in tags], dtype=np.uint8)
    if vertices_per_voxel > 1:
        colors = np.repeat(colors, vertices_per_voxel, axis=0)
    return colors


def _build_quad_faces(n_voxels: int) -> np.ndarray:
    """(N×6, 4) 1-based quad-face indices for OBJ."""
    bases = np.arange(n_voxels, dtype=np.int64) * 8 + 1  # OBJ is 1-based
    faces = bases[:, np.newaxis, np.newaxis] + _CUBE_QUADS[np.newaxis, :, :]
    return faces.reshape(-1, 4)


def _build_triangle_faces(n_voxels: int) -> np.ndarray:
    """(N×12, 3) 0-based triangle indices for glTF."""
    bases = np.arange(n_voxels, dtype=np.int64) * 8
    faces = bases[:, np.newaxis, np.newaxis] + _CUBE_TRIANGLES[np.newaxis, :, :]
    return faces.reshape(-1, 3)


# ═════════════════════════════════════════════════════════════════════
#  OBJ  export
# ═════════════════════════════════════════════════════════════════════

def csv_to_obj(
    source: Union[str, Path, TextIO, pd.DataFrame],
    output_path: Union[str, Path],
    scale: float = 1.0,
    use_colors: bool = True,
    color_column: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert voxel CSV → OBJ mesh (+ MTL material file when coloured)."""
    print(f"Reading voxels…")
    df = read_csv_smart(source, chunk_size)

    col = detect_color_column(list(df.columns), color_column)
    has_tags = col is not None and use_colors
    n_voxels = len(df)

    # Build dynamic palette from actual CSV classes
    if has_tags:
        unique_classes = [str(v) for v in df[col].dropna().unique()]
        build_dynamic_palette(unique_classes)

    print(f"Total voxels: {n_voxels:,}")
    if has_tags:
        print(f"Surface types found: {list(df[col].unique())}")

    output_path = Path(output_path)

    # --- MTL ---
    mtl_path: Optional[Path] = None
    if has_tags:
        mtl_path = output_path.with_suffix(".mtl")
        _create_mtl_file(mtl_path, df[col].unique())

    # --- Geometry ---
    coords = df[["X", "Y", "Z"]].values.astype(np.float64)
    vertices = _build_vertices(coords, scale)
    colors = _build_colors(df, col, use_colors, vertices_per_voxel=8)

    # --- Write OBJ ---
    with open(output_path, "w") as fh:
        fh.write("# Combined voxel visualisation\n")
        fh.write(f"# Total voxels: {n_voxels}\n")
        fh.write("# Generated by voxel-viewer\n")
        if mtl_path:
            fh.write(f"mtllib {output_path.stem}.mtl\n")
        fh.write("\n")

        # Vertices (vectorised via np.savetxt)
        if colors is not None:
            combined = np.hstack([vertices, colors.astype(np.float64) / 255.0])
            np.savetxt(fh, combined, fmt="v %.6f %.6f %.6f %.4f %.4f %.4f")
        else:
            np.savetxt(fh, vertices, fmt="v %.6f %.6f %.6f")

        fh.write("\n")

        # Faces — grouped by material when coloured
        if has_tags:
            all_faces = _build_quad_faces(n_voxels)  # (N*6, 4)
            materials = np.repeat(df[col].fillna("default").values, 6)
            unique_mats, inverse = np.unique(materials, return_inverse=True)
            for mat_idx, mat_name in enumerate(unique_mats):
                fh.write(f"\nusemtl {str(mat_name).replace(' ', '_')}\n")
                mask = inverse == mat_idx
                np.savetxt(fh, all_faces[mask], fmt="f %d %d %d %d")
        else:
            all_faces = _build_quad_faces(n_voxels)
            fh.write("\n")
            np.savetxt(fh, all_faces, fmt="f %d %d %d %d")

    stats = compute_statistics(df, col)
    print(f"✓ OBJ file created: {output_path}")
    if mtl_path:
        print(f"✓ MTL file created: {mtl_path}")
    return stats


def _create_mtl_file(mtl_path: Path, tag_values: np.ndarray) -> None:
    """Write an OBJ material library for the given surface types."""
    with open(mtl_path, "w") as fh:
        fh.write("# Material file for voxel visualisation\n")
        fh.write("# Generated by voxel-viewer\n\n")
        for tag in tag_values:
            mat_name = (
                str(tag).replace(" ", "_")
                if tag is not None and not (isinstance(tag, float) and pd.isna(tag))
                else "default"
            )
            r, g, b = get_color_for_tag(tag)
            rn, gn, bn = r / 255, g / 255, b / 255
            fh.write(f"newmtl {mat_name}\n")
            fh.write(f"Kd {rn:.4f} {gn:.4f} {bn:.4f}\n")
            fh.write(f"Ka {rn * 0.3:.4f} {gn * 0.3:.4f} {bn * 0.3:.4f}\n")
            fh.write("Ks 0.1 0.1 0.1\n")
            fh.write("Ns 10\n")
            fh.write("d 1.0\n\n")
    print(f"  Material file: {len(tag_values)} materials")


# ═════════════════════════════════════════════════════════════════════
#  PLY  export
# ═════════════════════════════════════════════════════════════════════

def csv_to_ply(
    source: Union[str, Path, TextIO, pd.DataFrame],
    output_path: Union[str, Path],
    use_colors: bool = True,
    color_column: Optional[str] = None,
    binary: bool = False,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert voxel CSV → PLY point cloud (ASCII or binary)."""
    print(f"Reading voxels…")
    df = read_csv_smart(source, chunk_size)

    col = detect_color_column(list(df.columns), color_column)
    has_tags = col is not None and use_colors
    n_voxels = len(df)

    # Build dynamic palette from actual CSV classes
    if has_tags:
        unique_classes = [str(v) for v in df[col].dropna().unique()]
        build_dynamic_palette(unique_classes)

    print(f"Total voxels: {n_voxels:,}")
    if has_tags:
        print(f"Surface types found: {list(df[col].unique())}")

    output_path = Path(output_path)
    coords = df[["X", "Y", "Z"]].values.astype(np.float64)
    colors = _build_colors(df, col, use_colors, vertices_per_voxel=1)

    if binary:
        _write_ply_binary(output_path, coords, colors)
    else:
        _write_ply_ascii(output_path, coords, colors)

    stats = compute_statistics(df, col)
    print(f"✓ PLY file created: {output_path}")
    if has_tags:
        print("  Vertex colours applied by surface type")
    return stats


def _write_ply_ascii(
    path: Path,
    coords: np.ndarray,
    colors: Optional[np.ndarray],
) -> None:
    n = len(coords)
    has_colors = colors is not None
    with open(path, "w") as fh:
        # Header
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write("comment Generated by voxel-viewer\n")
        fh.write(f"element vertex {n}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        if has_colors:
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        # Data
        if has_colors:
            combined = np.hstack([coords, colors.astype(np.float64)])
            np.savetxt(fh, combined, fmt="%.6f %.6f %.6f %d %d %d")
        else:
            np.savetxt(fh, coords, fmt="%.6f %.6f %.6f")


def _write_ply_binary(
    path: Path,
    coords: np.ndarray,
    colors: Optional[np.ndarray],
) -> None:
    n = len(coords)
    has_colors = colors is not None
    with open(path, "wb") as fh:
        lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment Generated by voxel-viewer",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_colors:
            lines += ["property uchar red", "property uchar green", "property uchar blue"]
        lines.append("end_header")
        fh.write(("\n".join(lines) + "\n").encode("ascii"))

        coords_f32 = coords.astype(np.float32)
        if has_colors:
            dtype = np.dtype(
                [
                    ("x", np.float32),
                    ("y", np.float32),
                    ("z", np.float32),
                    ("r", np.uint8),
                    ("g", np.uint8),
                    ("b", np.uint8),
                ]
            )
            data = np.empty(n, dtype=dtype)
            data["x"] = coords_f32[:, 0]
            data["y"] = coords_f32[:, 1]
            data["z"] = coords_f32[:, 2]
            data["r"] = colors[:, 0]
            data["g"] = colors[:, 1]
            data["b"] = colors[:, 2]
            data.tofile(fh)
        else:
            coords_f32.tofile(fh)


# ═════════════════════════════════════════════════════════════════════
#  glTF / GLB  export
# ═════════════════════════════════════════════════════════════════════

def csv_to_gltf(
    source: Union[str, Path, TextIO, pd.DataFrame],
    output_path: Union[str, Path],
    scale: float = 1.0,
    use_colors: bool = True,
    color_column: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert voxel CSV → binary glTF (.glb) for web viewers."""
    print("Reading voxels…")
    df = read_csv_smart(source, chunk_size)

    col = detect_color_column(list(df.columns), color_column)
    has_tags = col is not None and use_colors
    n_voxels = len(df)

    # Build dynamic palette from actual CSV classes
    if has_tags:
        unique_classes = [str(v) for v in df[col].dropna().unique()]
        build_dynamic_palette(unique_classes)

    print(f"Total voxels: {n_voxels:,}")

    coords = df[["X", "Y", "Z"]].values.astype(np.float64)
    vertices = _build_vertices(coords, scale).astype(np.float32)
    indices = _build_triangle_faces(n_voxels).astype(np.uint32).flatten()

    vertex_colors: Optional[np.ndarray] = None
    if has_tags:
        c8 = _build_colors(df, col, use_colors, vertices_per_voxel=8)
        if c8 is not None:
            vc = np.ones((len(c8), 4), dtype=np.float32)
            vc[:, :3] = c8.astype(np.float32) / 255.0
            vertex_colors = vc

    output_path = Path(output_path)
    _write_glb(output_path, vertices, indices, vertex_colors)

    stats = compute_statistics(df, col)
    print(f"✓ GLB file created: {output_path}")
    return stats


def _write_glb(
    path: Path,
    vertices: np.ndarray,
    indices: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Construct a minimal GLB (binary glTF 2.0) from raw arrays."""
    n_vertices = len(vertices)
    n_indices = len(indices)

    # Binary buffers
    vbytes = vertices.astype(np.float32).tobytes()
    ibytes = indices.astype(np.uint32).tobytes()
    cbytes = colors.astype(np.float32).tobytes() if colors is not None else b""

    vlen, ilen, clen = len(vbytes), len(ibytes), len(cbytes)
    total_buf = vlen + ilen + clen

    # Accessors & buffer-views
    v_min = vertices.min(axis=0).tolist()
    v_max = vertices.max(axis=0).tolist()

    buffer_views = [
        {"buffer": 0, "byteOffset": 0, "byteLength": vlen, "target": 34962},
        {"buffer": 0, "byteOffset": vlen, "byteLength": ilen, "target": 34963},
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": 5126,
            "count": n_vertices,
            "type": "VEC3",
            "min": v_min,
            "max": v_max,
        },
        {
            "bufferView": 1,
            "componentType": 5125,
            "count": n_indices,
            "type": "SCALAR",
            "min": [0],
            "max": [int(n_vertices - 1)],
        },
    ]

    mesh_attrs: Dict[str, int] = {"POSITION": 0}

    if colors is not None:
        buffer_views.append(
            {"buffer": 0, "byteOffset": vlen + ilen, "byteLength": clen, "target": 34962}
        )
        accessors.append(
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": n_vertices,
                "type": "VEC4",
            }
        )
        mesh_attrs["COLOR_0"] = 2

    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "voxel-viewer"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {"primitives": [{"attributes": mesh_attrs, "indices": 1, "mode": 4}]}
        ],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": total_buf}],
    }

    # Serialise JSON chunk (pad to 4-byte alignment with spaces)
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - len(json_bytes) % 4) % 4
    json_bytes += b" " * json_pad

    # Binary data chunk (pad with null bytes)
    bin_data = vbytes + ibytes + cbytes
    bin_pad = (4 - len(bin_data) % 4) % 4
    bin_data += b"\x00" * bin_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

    with open(path, "wb") as fh:
        # GLB header
        fh.write(struct.pack("<III", 0x46546C67, 2, total_length))
        # JSON chunk
        fh.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        fh.write(json_bytes)
        # BIN chunk
        fh.write(struct.pack("<II", len(bin_data), 0x004E4942))
        fh.write(bin_data)
