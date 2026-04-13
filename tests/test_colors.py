"""Tests for voxel_viewer.colors module."""

import json
import pytest

from voxel_viewer.colors import (
    DEFAULT_COLORS,
    SURFACE_COLORS,
    SURFACE_COLUMN_CANDIDATES,
    build_dynamic_palette,
    detect_color_column,
    get_active_palette,
    get_color_for_tag,
    hex_to_rgb,
    load_color_scheme,
    reset_colors,
    rgb_to_hex,
    save_color_scheme,
)


# ── hex / rgb helpers ────────────────────────────────────────────────

class TestHexRgb:
    def test_hex_to_rgb(self):
        assert hex_to_rgb("#8B7765") == (139, 119, 101)

    def test_hex_to_rgb_no_hash(self):
        assert hex_to_rgb("B22222") == (178, 34, 34)

    def test_hex_to_rgb_invalid(self):
        with pytest.raises(ValueError):
            hex_to_rgb("#ZZ")

    def test_rgb_to_hex(self):
        assert rgb_to_hex(139, 119, 101) == "#8b7765"

    def test_roundtrip(self):
        r, g, b = 42, 128, 255
        assert hex_to_rgb(rgb_to_hex(r, g, b)) == (r, g, b)


# ── get_color_for_tag ────────────────────────────────────────────────

class TestGetColorForTag:
    def test_exact_match(self):
        assert get_color_for_tag("WallSurface") == DEFAULT_COLORS["WallSurface"]

    def test_case_insensitive(self):
        assert get_color_for_tag("wallsurface") == DEFAULT_COLORS["WallSurface"]

    def test_partial_match(self):
        assert get_color_for_tag("Wall") == DEFAULT_COLORS["WallSurface"]

    def test_none_returns_default(self):
        assert get_color_for_tag(None) == DEFAULT_COLORS["default"]

    def test_nan_returns_default(self):
        assert get_color_for_tag(float("nan")) == DEFAULT_COLORS["default"]

    def test_unknown_returns_default(self):
        assert get_color_for_tag("UnknownThing") == DEFAULT_COLORS["default"]

    def test_dynamic_palette_takes_priority(self):
        """After building a dynamic palette, tags should resolve from it."""
        build_dynamic_palette(["IfcBeam", "IfcSlab"])
        color = get_color_for_tag("IfcBeam")
        # Should NOT be the default grey
        assert color != DEFAULT_COLORS["default"]
        # Should be consistent
        assert color == get_color_for_tag("IfcBeam")


# ── detect_color_column ──────────────────────────────────────────────

class TestDetectColorColumn:
    def test_detects_object_type(self):
        cols = ["X", "Y", "Z", "object_type"]
        assert detect_color_column(cols) == "object_type"

    def test_detects_tag_value(self):
        cols = ["X", "Y", "Z", "tag_value"]
        assert detect_color_column(cols) == "tag_value"

    def test_prefers_tag_value_over_object_type(self):
        cols = ["X", "Y", "Z", "tag_value", "object_type"]
        assert detect_color_column(cols) == "tag_value"

    def test_user_column_overrides(self):
        cols = ["X", "Y", "Z", "tag_value", "my_col"]
        assert detect_color_column(cols, user_column="my_col") == "my_col"

    def test_user_column_missing_raises(self):
        cols = ["X", "Y", "Z"]
        with pytest.raises(ValueError, match="not found"):
            detect_color_column(cols, user_column="missing")

    def test_returns_none_when_absent(self):
        cols = ["X", "Y", "Z"]
        assert detect_color_column(cols) is None


# ── build_dynamic_palette ────────────────────────────────────────────

class TestBuildDynamicPalette:
    def test_generates_colors_for_unknown_classes(self):
        """Classes not in any predefined scheme get auto-generated colours."""
        classes = ["IfcBeam", "IfcSlab", "IfcColumn", "IfcRailing"]
        palette = build_dynamic_palette(classes)
        # Every class should have a colour
        for cls in classes:
            assert cls in palette
            assert len(palette[cls]) == 3  # RGB tuple

    def test_all_colors_are_distinct(self):
        """Each class should get a unique colour."""
        classes = ["ClassA", "ClassB", "ClassC", "ClassD", "ClassE"]
        palette = build_dynamic_palette(classes)
        colors = list(palette.values())
        assert len(set(colors)) == len(colors)

    def test_predefined_classes_use_existing_colors(self):
        """CityGML classes should use the hardcoded colours."""
        classes = ["WallSurface", "RoofSurface", "IfcBeam"]
        palette = build_dynamic_palette(classes)
        assert palette["WallSurface"] == DEFAULT_COLORS["WallSurface"]
        assert palette["RoofSurface"] == DEFAULT_COLORS["RoofSurface"]
        # IfcBeam has no predefined colour so gets auto-generated
        assert "IfcBeam" in palette
        assert palette["IfcBeam"] != DEFAULT_COLORS["default"]

    def test_user_scheme_overrides(self):
        """User-supplied colours should take priority."""
        user = {"IfcBeam": (255, 0, 0)}
        palette = build_dynamic_palette(["IfcBeam", "IfcSlab"], user_scheme=user)
        assert palette["IfcBeam"] == (255, 0, 0)
        assert "IfcSlab" in palette

    def test_empty_classes(self):
        palette = build_dynamic_palette([])
        assert palette == {}

    def test_get_color_for_tag_uses_dynamic(self):
        """get_color_for_tag should use the dynamic palette once built."""
        build_dynamic_palette(["IfcBeam"])
        color = get_color_for_tag("IfcBeam")
        assert color != DEFAULT_COLORS["default"]

    def test_reset_clears_dynamic(self):
        """reset_colors should clear the dynamic palette."""
        build_dynamic_palette(["IfcBeam"])
        assert get_color_for_tag("IfcBeam") != DEFAULT_COLORS["default"]
        reset_colors()
        # After reset, IfcBeam has no match → default
        assert get_color_for_tag("IfcBeam") == DEFAULT_COLORS["default"]

    def test_large_palette(self):
        """Should handle many classes without errors."""
        classes = [f"Class_{i}" for i in range(50)]
        palette = build_dynamic_palette(classes)
        assert len(palette) == 50
        # All should be unique
        colors = list(palette.values())
        assert len(set(colors)) == 50


# ── get_active_palette ───────────────────────────────────────────────

class TestGetActivePalette:
    def test_returns_static_when_no_dynamic(self):
        """Without building a dynamic palette, returns SURFACE_COLORS."""
        palette = get_active_palette()
        assert "WallSurface" in palette
        assert "default" not in palette  # 'default' is excluded

    def test_returns_dynamic_when_built(self):
        build_dynamic_palette(["IfcBeam", "IfcSlab"])
        palette = get_active_palette()
        assert "IfcBeam" in palette
        assert "IfcSlab" in palette

    def test_reset_goes_back_to_static(self):
        build_dynamic_palette(["IfcBeam"])
        reset_colors()
        palette = get_active_palette()
        assert "IfcBeam" not in palette
        assert "WallSurface" in palette


# ── load / save / reset ─────────────────────────────────────────────

class TestColorSchemeIO:
    def test_load_rgb_array(self, tmp_path):
        p = tmp_path / "scheme.json"
        p.write_text(json.dumps({"WallSurface": [255, 0, 0]}))
        load_color_scheme(p)
        assert SURFACE_COLORS["WallSurface"] == (255, 0, 0)

    def test_load_hex_string(self, tmp_path):
        p = tmp_path / "scheme.json"
        p.write_text(json.dumps({"RoofSurface": "#00FF00"}))
        load_color_scheme(p)
        assert SURFACE_COLORS["RoofSurface"] == (0, 255, 0)

    def test_save_and_reload(self, tmp_path):
        p = tmp_path / "out.json"
        save_color_scheme(p)
        data = json.loads(p.read_text())
        assert "WallSurface" in data
        assert len(data["WallSurface"]) == 3

    def test_reset(self):
        SURFACE_COLORS["WallSurface"] = (0, 0, 0)
        reset_colors()
        assert SURFACE_COLORS["WallSurface"] == DEFAULT_COLORS["WallSurface"]

    def test_load_custom_classes_and_build(self, tmp_path):
        """Custom JSON classes should be respected by build_dynamic_palette."""
        p = tmp_path / "scheme.json"
        p.write_text(json.dumps({"IfcBeam": [0, 128, 255]}))
        load_color_scheme(p)
        palette = build_dynamic_palette(["IfcBeam", "IfcSlab"])
        assert palette["IfcBeam"] == (0, 128, 255)
