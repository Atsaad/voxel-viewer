"""Color management for voxel surface types.

Provides a default CityGML color scheme and supports custom schemes
loaded from JSON files.  Colors can be specified as RGB arrays [r,g,b]
(0-255) or hex strings ("#8B7765").

**Dynamic palette**: when the CSV contains classes not present in any
predefined or user-supplied scheme, the module auto-generates visually
distinct colours so that every class gets its own colour automatically.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

# ── Default CityGML color scheme (RGB 0-255) ────────────────────────

DEFAULT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "WallSurface":         (139, 119, 101),
    "RoofSurface":         (178,  34,  34),
    "GroundSurface":       ( 34, 139,  34),
    "OuterFloorSurface":   (128, 128, 128),
    "OuterCeilingSurface": (169, 169, 169),
    "ClosureSurface":      (255, 165,   0),
    "Window":              (135, 206, 235),
    "Door":                (139,  69,  19),
    "default":             (200, 200, 200),
}

# Known column names that hold the surface type (priority order)
SURFACE_COLUMN_CANDIDATES: list[str] = [
    "tag_value",
    "object_type",
    "surface_type",
    "type",
]

# Active (mutable) color scheme — updated by load_color_scheme()
SURFACE_COLORS: Dict[str, Tuple[int, int, int]] = dict(DEFAULT_COLORS)

# Dynamic palette built from CSV classes (populated by build_dynamic_palette)
_DYNAMIC_COLORS: Dict[str, Tuple[int, int, int]] = {}


# ── Colour helpers ───────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert ``#RRGGBB`` hex string to an ``(R, G, B)`` tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert ``(R, G, B)`` tuple to ``#rrggbb`` hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Dynamic palette generation ──────────────────────────────────────

def _generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """Generate *n* visually distinct RGB colours using the golden-angle
    hue spacing technique.  Saturation and lightness are varied slightly
    to improve discrimination when *n* is large.

    Returns a list of ``(R, G, B)`` tuples (0-255).
    """
    if n <= 0:
        return []

    colors: List[Tuple[int, int, int]] = []
    golden_ratio = 0.618033988749895

    for i in range(n):
        hue = (i * golden_ratio) % 1.0
        # Alternate between two saturation / lightness bands
        saturation = 0.65 + 0.20 * (i % 3) / 2
        lightness = 0.45 + 0.15 * ((i // 3) % 3) / 2
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    return colors


def build_dynamic_palette(
    class_values: Sequence[str],
    user_scheme: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Dict[str, Tuple[int, int, int]]:
    """Build a colour palette for an arbitrary set of class values.

    Colour assignment priority:

    1. ``user_scheme`` (loaded JSON or programmatic overrides)
    2. ``SURFACE_COLORS`` (CityGML defaults + anything previously loaded)
    3. Auto-generated visually distinct colours for everything else

    The result is cached in the module-level ``_DYNAMIC_COLORS`` dict and
    is also used by :func:`get_color_for_tag`.

    Parameters
    ----------
    class_values : sequence of str
        Unique class labels found in the CSV data.
    user_scheme : dict, optional
        Per-class colour overrides supplied by the user.

    Returns
    -------
    dict
        Mapping of *every* class value to an ``(R, G, B)`` tuple.
    """
    global _DYNAMIC_COLORS
    palette: Dict[str, Tuple[int, int, int]] = {}
    unmapped: List[str] = []

    merged_scheme: Dict[str, Tuple[int, int, int]] = dict(SURFACE_COLORS)
    if user_scheme:
        merged_scheme.update(user_scheme)

    for cls in class_values:
        cls_str = str(cls)
        if cls_str in merged_scheme:
            # Direct hit in the existing scheme
            palette[cls_str] = merged_scheme[cls_str]
        else:
            # Try case-insensitive / partial match against existing scheme
            matched = _fuzzy_lookup(cls_str, merged_scheme)
            if matched is not None:
                palette[cls_str] = matched
            else:
                unmapped.append(cls_str)

    # Generate colours for unmapped classes
    if unmapped:
        generated = _generate_distinct_colors(len(unmapped))
        for cls_str, color in zip(unmapped, generated):
            palette[cls_str] = color

    _DYNAMIC_COLORS = palette
    return palette


def _fuzzy_lookup(
    tag: str,
    scheme: Dict[str, Tuple[int, int, int]],
) -> Optional[Tuple[int, int, int]]:
    """Case-insensitive / partial match against a colour scheme."""
    tag_lower = tag.lower()
    for key, color in scheme.items():
        if key == "default":
            continue
        if key.lower() in tag_lower or tag_lower in key.lower():
            return color
    return None


def get_color_for_tag(tag_value: Any) -> Tuple[int, int, int]:
    """Return the ``(R, G, B)`` colour for a surface-type tag.

    Lookup order:

    1. Dynamic palette (built by :func:`build_dynamic_palette`).
    2. Exact match in ``SURFACE_COLORS``.
    3. Case-insensitive partial match in ``SURFACE_COLORS``.
    4. Falls back to ``SURFACE_COLORS['default']``.
    """
    if tag_value is None or (isinstance(tag_value, float) and pd.isna(tag_value)):
        return SURFACE_COLORS.get("default", DEFAULT_COLORS["default"])

    tag_str = str(tag_value)

    # 1. Dynamic palette takes priority (most specific)
    if tag_str in _DYNAMIC_COLORS:
        return _DYNAMIC_COLORS[tag_str]

    # 2. Exact match in static scheme
    if tag_str in SURFACE_COLORS:
        return SURFACE_COLORS[tag_str]

    # 3. Case-insensitive partial match
    matched = _fuzzy_lookup(tag_str, SURFACE_COLORS)
    if matched is not None:
        return matched

    return SURFACE_COLORS.get("default", DEFAULT_COLORS["default"])


# ── Column auto-detection ────────────────────────────────────────────

def detect_color_column(
    columns: Sequence[str],
    user_column: Optional[str] = None,
) -> Optional[str]:
    """Auto-detect which CSV column holds the surface type.

    Parameters
    ----------
    columns : sequence of str
        Column names from the DataFrame / CSV header.
    user_column : str, optional
        Explicit column name supplied by the user (takes priority).

    Returns
    -------
    str or None
        Detected column name, or *None* if not found.

    Raises
    ------
    ValueError
        If *user_column* is given but not present in *columns*.
    """
    col_list = list(columns)

    if user_column:
        if user_column in col_list:
            return user_column
        raise ValueError(
            f"Specified column '{user_column}' not found. "
            f"Available columns: {col_list}"
        )

    for candidate in SURFACE_COLUMN_CANDIDATES:
        if candidate in col_list:
            return candidate

    return None


# ── Custom colour-scheme I/O ────────────────────────────────────────

def load_color_scheme(path: Path | str) -> Dict[str, Tuple[int, int, int]]:
    """Load a custom color scheme from a JSON file.

    The JSON may use RGB arrays ``[r, g, b]`` or hex strings ``"#RRGGBB"``.
    Loaded colours are merged into the active ``SURFACE_COLORS`` dict.
    """
    path = Path(path)
    with open(path) as fh:
        data: dict = json.load(fh)

    for key, value in data.items():
        if isinstance(value, list) and len(value) == 3:
            SURFACE_COLORS[key] = tuple(int(v) for v in value)  # type: ignore[arg-type]
        elif isinstance(value, str) and value.startswith("#"):
            SURFACE_COLORS[key] = hex_to_rgb(value)
        else:
            raise ValueError(
                f"Invalid colour for '{key}': expected [R,G,B] or '#RRGGBB', got {value!r}"
            )

    return dict(SURFACE_COLORS)


def save_color_scheme(
    path: Path | str,
    scheme: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> None:
    """Persist a colour scheme to a JSON file (RGB arrays)."""
    scheme = scheme or SURFACE_COLORS
    data = {k: list(v) for k, v in scheme.items()}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def reset_colors() -> None:
    """Restore the built-in default colour scheme and clear dynamic palette."""
    global _DYNAMIC_COLORS
    SURFACE_COLORS.clear()
    SURFACE_COLORS.update(DEFAULT_COLORS)
    _DYNAMIC_COLORS = {}


def get_active_palette() -> Dict[str, Tuple[int, int, int]]:
    """Return the currently active palette (dynamic if built, else static).

    This is the palette that should be used for legends and UI display.
    """
    if _DYNAMIC_COLORS:
        return dict(_DYNAMIC_COLORS)
    return {k: v for k, v in SURFACE_COLORS.items() if k != "default"}


def print_color_legend() -> None:
    """Print the colour legend to stdout."""
    palette = get_active_palette()
    print("\n🎨 Color Legend:")
    print("-" * 45)
    for surface, (r, g, b) in palette.items():
        print(f"  {surface:25} RGB({r:3}, {g:3}, {b:3})")
    print("-" * 45)
