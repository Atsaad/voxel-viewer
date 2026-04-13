"""Binvox file reader — parse ``.binvox`` voxel grids (v1 RLE format).

This module provides a zero-dependency reader for the binvox format
produced by the ``binvox`` / ``cuda_voxelizer`` tools.  It can:

* Read a single ``.binvox`` file into a DataFrame ``(X, Y, Z)``.
* Read an entire directory of ``.binvox`` files and merge them (with
  deduplication) into a single combined DataFrame.

The voxel coordinates are reconstructed in *world space* using the
``translate`` and ``scale`` fields from each file's header.

File format reference: https://www.patrickmin.com/binvox/binvox.html
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class BinvoxHeader:
    """Parsed header of a ``.binvox`` file."""
    version: int
    dims: Tuple[int, int, int]       # (depth, height, width) — X, Y, Z grid sizes
    translate: Tuple[float, float, float]
    scale: float

    @property
    def voxel_size(self) -> float:
        """Size of one voxel in world units (metres)."""
        return self.scale / max(self.dims)


@dataclass
class BinvoxModel:
    """A parsed binvox file: header + occupied voxel coordinates."""
    header: BinvoxHeader
    voxels: np.ndarray               # (N, 3) int32 grid coordinates
    source_file: Optional[str] = None

    @property
    def n_voxels(self) -> int:
        return len(self.voxels)

    def to_world_coords(self) -> np.ndarray:
        """Convert grid coordinates → world coordinates (float64).

        Uses the formula from the binvox spec::

            world_x = translate_x + grid_x * (scale / dim_x)
            world_y = translate_y + grid_y * (scale / dim_y)
            world_z = translate_z + grid_z * (scale / dim_z)
        """
        h = self.header
        dims = np.array(h.dims, dtype=np.float64)
        translate = np.array(h.translate, dtype=np.float64)
        step = h.scale / dims
        return translate + self.voxels.astype(np.float64) * step

    def to_dataframe(self, include_source: bool = True) -> pd.DataFrame:
        """Convert to a DataFrame with ``X, Y, Z`` world coordinates.

        Binvox / cuda_voxelizer uses **Y-up** (graphics convention) where
        the axes are ``(X, Y_up, Z_north)``.  The CSV pipeline expects
        **Z-up** (GIS convention): ``(X_east, Y_north, Z_up)``.

        This method swaps Y ↔ Z so the output is consistent with the
        rest of the voxel-viewer pipeline.
        """
        world = self.to_world_coords()
        # Swap Y and Z: binvox (X, Y_up, Z_north) → GIS (X, Y_north, Z_up)
        df = pd.DataFrame({
            "X": world[:, 0],
            "Y": world[:, 2],   # binvox Z (northing) → CSV Y
            "Z": world[:, 1],   # binvox Y (elevation) → CSV Z
        })
        if include_source and self.source_file:
            df["source_file"] = Path(self.source_file).name
        return df


# ── Parser ───────────────────────────────────────────────────────────

def _parse_header(fp) -> BinvoxHeader:
    """Read the text header from an open binary file handle."""
    # Line 1: #binvox <version>
    line = fp.readline().decode("ascii").strip()
    if not line.startswith("#binvox"):
        raise ValueError(f"Not a valid binvox file: expected '#binvox', got '{line}'")
    version = int(line.split()[1])

    dims = None
    translate = None
    scale = None

    while True:
        line = fp.readline().decode("ascii").strip()
        if line.startswith("dim"):
            parts = line.split()
            dims = (int(parts[1]), int(parts[2]), int(parts[3]))
        elif line.startswith("translate"):
            parts = line.split()
            translate = (float(parts[1]), float(parts[2]), float(parts[3]))
        elif line.startswith("scale"):
            scale = float(line.split()[1])
        elif line.startswith("data"):
            break
        elif line == "":
            raise ValueError("Unexpected end of binvox header before 'data' line")

    if dims is None or translate is None or scale is None:
        raise ValueError("Incomplete binvox header (missing dim, translate, or scale)")

    return BinvoxHeader(version=version, dims=dims, translate=translate, scale=scale)


def _decode_rle(raw_bytes: bytes, total_voxels: int) -> np.ndarray:
    """Decode run-length encoded voxel data → boolean flat array."""
    grid = np.zeros(total_voxels, dtype=np.bool_)
    idx = 0
    pos = 0

    while pos < len(raw_bytes) - 1 and idx < total_voxels:
        value = raw_bytes[pos]
        count = raw_bytes[pos + 1]
        pos += 2
        if value:
            end = min(idx + count, total_voxels)
            grid[idx:end] = True
        idx += count

    return grid


def read_binvox(path: Union[str, Path]) -> BinvoxModel:
    """Read a single ``.binvox`` file and return a :class:`BinvoxModel`.

    Parameters
    ----------
    path : str or Path
        Path to the ``.binvox`` file.

    Returns
    -------
    BinvoxModel
        Parsed model with grid coordinates of occupied voxels.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Binvox file not found: {path}")

    with open(path, "rb") as fp:
        header = _parse_header(fp)
        raw = fp.read()

    total = header.dims[0] * header.dims[1] * header.dims[2]
    grid_flat = _decode_rle(raw, total)

    # Convert flat boolean array → (N, 3) grid coordinates
    # binvox stores in x-major order: index = x * (dim_y * dim_z) + y * dim_z + z
    occupied_indices = np.where(grid_flat)[0]

    dx, dy, dz = header.dims
    x = occupied_indices // (dy * dz)
    remainder = occupied_indices % (dy * dz)
    # binvox actually stores depth-major: z changes fastest
    # index = x * (height * width) + z * width + y  (binvox quirk)
    z = remainder // dy
    y = remainder % dy

    voxels = np.column_stack([x, y, z]).astype(np.int32)

    return BinvoxModel(header=header, voxels=voxels, source_file=str(path))


def read_binvox_directory(
    directory: Union[str, Path],
    pattern: str = "*.binvox",
    deduplicate: bool = True,
) -> Tuple[pd.DataFrame, List[BinvoxModel]]:
    """Read all ``.binvox`` files in a directory and merge into one DataFrame.

    Parameters
    ----------
    directory : str or Path
        Directory containing ``.binvox`` files.
    pattern : str
        Glob pattern for matching files (default: ``*.binvox``).
    deduplicate : bool
        If True, remove duplicate voxel positions across files.

    Returns
    -------
    tuple of (DataFrame, list of BinvoxModel)
        Combined DataFrame with ``X, Y, Z`` (and ``source_file``) plus
        the individual parsed models for inspection.
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No binvox files matching '{pattern}' found in {directory}"
        )

    models: List[BinvoxModel] = []
    frames: List[pd.DataFrame] = []

    for f in files:
        print(f"  Reading: {f.name}")
        model = read_binvox(f)
        models.append(model)
        frames.append(model.to_dataframe(include_source=True))
        print(f"    → {model.n_voxels:,} voxels, "
              f"grid {model.header.dims}, "
              f"voxel_size={model.header.voxel_size:.4f}m")

    df = pd.concat(frames, ignore_index=True)

    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset=["X", "Y", "Z"], keep="first")
        removed = before - len(df)
        if removed > 0:
            print(f"  Deduplicated: removed {removed:,} overlapping voxels")

    print(f"  Combined: {len(df):,} voxels from {len(models)} files")
    return df, models


def binvox_summary(models: Sequence[BinvoxModel]) -> Dict:
    """Return a summary dict for a list of parsed binvox models."""
    return {
        "file_count": len(models),
        "total_voxels": sum(m.n_voxels for m in models),
        "grid_sizes": [m.header.dims for m in models],
        "voxel_sizes_m": [round(m.header.voxel_size, 4) for m in models],
        "translates": [m.header.translate for m in models],
        "scales": [m.header.scale for m in models],
        "files": [Path(m.source_file).name if m.source_file else "?" for m in models],
    }
