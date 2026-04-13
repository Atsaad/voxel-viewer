"""Voxel Viewer — Convert voxel CSV / binvox to 3D visualization formats.

Supports OBJ (mesh), PLY (point cloud), and glTF/GLB (web) with
automatic surface-type coloring.  Reads both CSV and binvox inputs.
"""

__version__ = "2.0.0"

from .colors import (
    SURFACE_COLORS,
    build_dynamic_palette,
    get_active_palette,
    get_color_for_tag,
    detect_color_column,
    load_color_scheme,
    save_color_scheme,
    reset_colors,
)
from .converter import (
    csv_to_obj,
    csv_to_ply,
    csv_to_gltf,
    compute_statistics,
    read_csv_smart,
    validate_csv,
)
from .binvox import (
    read_binvox,
    read_binvox_directory,
    binvox_summary,
    BinvoxHeader,
    BinvoxModel,
)

__all__ = [
    "SURFACE_COLORS",
    "build_dynamic_palette",
    "get_active_palette",
    "get_color_for_tag",
    "detect_color_column",
    "load_color_scheme",
    "save_color_scheme",
    "reset_colors",
    "csv_to_obj",
    "csv_to_ply",
    "csv_to_gltf",
    "compute_statistics",
    "read_csv_smart",
    "validate_csv",
    "read_binvox",
    "read_binvox_directory",
    "binvox_summary",
    "BinvoxHeader",
    "BinvoxModel",
]
