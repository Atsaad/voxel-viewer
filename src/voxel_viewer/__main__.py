"""Command-line interface for voxel-viewer.

Usage examples::

    # Convert with defaults (all formats, coloured)
    voxel-viewer data/voxels_output.csv

    # PLY only, binary, custom voxel size
    voxel-viewer data/voxels_output.csv -f ply --binary-ply -s 0.25

    # Pipe from stdin
    cat data/voxels_output.csv | voxel-viewer - -f obj -o my_output

    # Just print statistics
    voxel-viewer data/voxels_output.csv -f none --stats

    # Launch GUI
    voxel-viewer --gui
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from .colors import build_dynamic_palette, load_color_scheme, print_color_legend
from .binvox import read_binvox, read_binvox_directory, binvox_summary
from .converter import (
    compute_statistics,
    csv_to_gltf,
    csv_to_obj,
    csv_to_ply,
    detect_color_column,
    read_csv_smart,
    validate_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxel-viewer",
        description=(
            "Convert voxel CSV or binvox files to 3D visualisation formats "
            "(OBJ mesh, PLY point cloud, glTF/GLB) with automatic "
            "surface-type colouring."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  voxel-viewer data/voxels_output.csv\n"
            "  voxel-viewer data.csv -f ply --binary-ply\n"
            "  voxel-viewer data.csv -f obj -s 0.25 --column object_type\n"
            "  voxel-viewer data.csv -f none --stats\n"
            "  cat data.csv | voxel-viewer - -f ply -o output\n"
            "  voxel-viewer --gui\n"
        ),
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Path to voxels CSV, .binvox file, or directory of .binvox files, or '-' for stdin.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output base path (extensions added automatically). "
             "Default: derived from input filename.",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["obj", "ply", "gltf", "glb", "all", "none"],
        default="all",
        help="Output format (default: all).",
    )
    parser.add_argument(
        "-s", "--scale", "--voxel-size",
        type=float,
        default=0.5,
        dest="scale",
        help="Voxel cube size in metres (default: 0.5).",
    )
    parser.add_argument(
        "-c", "--column",
        help="Column name for surface-type colouring (auto-detected by default).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable surface-type colouring.",
    )
    parser.add_argument(
        "--binary-ply",
        action="store_true",
        help="Write PLY in binary format (smaller file, faster I/O).",
    )
    parser.add_argument(
        "--color-scheme",
        type=Path,
        metavar="JSON",
        help="Path to custom colour-scheme JSON file.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        metavar="N",
        help="Read CSV in chunks of N rows (reduces parser memory peaks).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical interface.",
    )

    return parser


def _print_stats(stats: dict) -> None:
    """Pretty-print dataset statistics."""
    print("\n📊 Dataset Statistics:")
    print(f"  Total voxels: {stats['total_voxels']:,}")
    print("  Bounding box:")
    for axis in ("X", "Y", "Z"):
        lo = stats["bbox_min"][axis]
        hi = stats["bbox_max"][axis]
        print(f"    {axis}: {lo:,.2f} → {hi:,.2f}")
    c = stats["centroid"]
    print(f"  Centroid: ({c['X']:,.2f}, {c['Y']:,.2f}, {c['Z']:,.2f})")
    if "surface_counts" in stats:
        total = stats["total_voxels"]
        print(f"  Surface types ({stats['unique_surfaces']}):")
        for surface, count in sorted(
            stats["surface_counts"].items(), key=lambda x: -x[1]
        ):
            pct = count / total * 100
            print(f"    {surface:25} {count:>10,} ({pct:5.1f}%)")
    print()


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── GUI shortcut ─────────────────────────────────────────────
    if args.gui:
        from .gui import main as gui_main

        gui_main()
        return

    # ── Validate input ───────────────────────────────────────────
    if not args.input:
        parser.print_help()
        sys.exit(1)

    # ── Load custom colour scheme ────────────────────────────────
    if args.color_scheme:
        load_color_scheme(args.color_scheme)

    # ── Read data once ───────────────────────────────────────
    input_path = Path(args.input) if args.input != "-" else None

    if args.input == "-":
        print("Reading from stdin…")
        df = pd.read_csv(sys.stdin)
        validate_csv(df)
        base_name = "stdin_voxels"
    elif input_path and input_path.is_dir():
        # Directory of binvox files
        print(f"Reading binvox directory: {input_path}")
        df, models = read_binvox_directory(str(input_path))
        base_name = input_path.name + "_combined"
        summary = binvox_summary(models)
        print(f"\nℹ  {summary['file_count']} files, {len(df):,} voxels")
        # Use source_file for coloring if no class column exists
        if "source_file" in df.columns and not args.no_color:
            from .colors import build_dynamic_palette
            unique_sources = [str(v) for v in df["source_file"].unique()]
            build_dynamic_palette(unique_sources)
    elif input_path and input_path.suffix.lower() == ".binvox":
        # Single binvox file
        print(f"Reading binvox: {input_path}")
        model = read_binvox(str(input_path))
        df = model.to_dataframe(include_source=True)
        base_name = input_path.stem
        print(f"  → {model.n_voxels:,} voxels, grid {model.header.dims}, "
              f"voxel_size={model.header.voxel_size:.4f}m")
    else:
        df = read_csv_smart(args.input, args.chunk_size)
        base_name = Path(args.input).stem

    use_colors = not args.no_color
    fmt = args.format

    # ── Output base path ─────────────────────────────────────────
    if args.output:
        output_base = str(Path(args.output).with_suffix(""))
    else:
        output_base = base_name

    # ── Build dynamic palette from CSV classes ───────────────────
    col = detect_color_column(list(df.columns), args.column)
    if use_colors and col and col in df.columns:
        unique_classes = [str(v) for v in df[col].dropna().unique()]
        build_dynamic_palette(unique_classes)
        print(f"\nℹ  Found {len(unique_classes)} classes in '{col}': {unique_classes}")

    # ── Print colour legend ──────────────────────────────────────
    if use_colors:
        print_color_legend()
        print()

    # ── Convert ──────────────────────────────────────────────────
    if fmt in ("ply", "all"):
        ply_path = f"{output_base}_pointcloud.ply"
        csv_to_ply(df, ply_path, use_colors, args.column, args.binary_ply)
        print()

    if fmt in ("obj", "all"):
        obj_path = f"{output_base}_voxels.obj"
        csv_to_obj(df, obj_path, args.scale, use_colors, args.column)
        print()

    if fmt in ("gltf", "glb", "all"):
        glb_path = f"{output_base}_voxels.glb"
        csv_to_gltf(df, glb_path, args.scale, use_colors, args.column)
        print()

    # ── Statistics ───────────────────────────────────────────────
    if args.stats or fmt == "none":
        col = detect_color_column(list(df.columns), args.column)
        stats = compute_statistics(df, col)
        _print_stats(stats)

    if fmt == "none" and not args.stats:
        # Nothing requested — show stats anyway as feedback
        col = detect_color_column(list(df.columns), args.column)
        stats = compute_statistics(df, col)
        _print_stats(stats)

    print("Done! ✨")


if __name__ == "__main__":
    main()
