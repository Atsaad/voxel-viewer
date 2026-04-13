"""Voxel Viewer GUI — Modern CustomTkinter interface.

Features beyond the original:
* Auto-detects surface-type column (tag_value / object_type / …)
* Statistics panel (voxel count, bounding box, centroid, surface breakdown)
* Embedded matplotlib 3D scatter preview
* glTF / GLB export option
* Binary PLY toggle
* Load custom colour scheme from JSON
* Proper error handling & type hints
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional

import customtkinter as ctk

# Relative imports (works when launched via  python -m voxel_viewer --gui)
from .colors import (
    SURFACE_COLORS,
    build_dynamic_palette,
    detect_color_column,
    get_active_palette,
    load_color_scheme,
    reset_colors,
)
from .converter import (
    compute_statistics,
    csv_to_gltf,
    csv_to_obj,
    csv_to_ply,
    read_csv_smart,
)
from .binvox import read_binvox, read_binvox_directory, binvox_summary

# Optional matplotlib support
try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Appearance ───────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ═════════════════════════════════════════════════════════════════════

class VoxelViewerApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title("🎲 Voxel Viewer")
        self.geometry("780x900")
        self.minsize(680, 700)

        # Paths
        self.app_dir = Path(__file__).parent.absolute()
        self.input_dir = Path.cwd() / "input"
        self.output_dir = Path.cwd() / "output"
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        # State
        self.selected_file = ctk.StringVar(value="")
        self.voxel_size = ctk.DoubleVar(value=0.5)
        self.generate_obj = ctk.BooleanVar(value=True)
        self.generate_ply = ctk.BooleanVar(value=True)
        self.generate_gltf = ctk.BooleanVar(value=True)
        self.binary_ply = ctk.BooleanVar(value=False)
        self.use_colors = ctk.BooleanVar(value=True)

        self._last_stats: Optional[Dict[str, Any]] = None
        self._last_df: Optional[Any] = None  # pd.DataFrame kept for preview

        self._create_widgets()

    # ── Widget layout ────────────────────────────────────────────

    def _create_widgets(self) -> None:
        main = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # ── Header ───────────────────────────────────────────────
        hdr = ctk.CTkFrame(main, fg_color="transparent")
        hdr.pack(fill="x", pady=(0, 20))
        ctk.CTkLabel(hdr, text="🎲 Voxel Viewer",
                      font=ctk.CTkFont(size=28, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(hdr, text="Convert voxel CSV / binvox to coloured 3D formats",
                      font=ctk.CTkFont(size=14), text_color="gray").pack(anchor="w")

        # ── Input ────────────────────────────────────────────────
        inp = ctk.CTkFrame(main)
        inp.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(inp, text="📁 Input File",
                      font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=15, pady=(15, 10))

        row = ctk.CTkFrame(inp, fg_color="transparent")
        row.pack(fill="x", padx=15, pady=(0, 10))
        self.file_entry = ctk.CTkEntry(
            row, textvariable=self.selected_file,
            placeholder_text="Select a CSV or binvox file / folder…", height=40,
            font=ctk.CTkFont(size=13))
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(row, text="Browse CSV", command=self._browse_file,
                       width=100, height=40).pack(side="right")

        # Binvox browse buttons
        bvox_row = ctk.CTkFrame(inp, fg_color="transparent")
        bvox_row.pack(fill="x", padx=15, pady=(0, 10))
        ctk.CTkButton(
            bvox_row, text="📦 Browse Binvox File",
            command=self._browse_binvox,
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"), height=32, width=180,
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            bvox_row, text="📂 Browse Binvox Folder",
            command=self._browse_binvox_folder,
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"), height=32, width=200,
        ).pack(side="left", padx=(0, 10))
        ctk.CTkButton(
            bvox_row, text="📂 Open Input Folder",
            command=lambda: self._open_folder(self.input_dir),
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"), height=32,
        ).pack(side="right")

        # ── Settings ─────────────────────────────────────────────
        sett = ctk.CTkFrame(main)
        sett.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(sett, text="⚙️ Settings",
                      font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=15, pady=(15, 10))

        # Voxel size
        sz = ctk.CTkFrame(sett, fg_color="transparent")
        sz.pack(fill="x", padx=15, pady=(0, 5))
        ctk.CTkLabel(sz, text="Voxel Size:", font=ctk.CTkFont(size=13)).pack(side="left")
        self._size_lbl = ctk.CTkLabel(
            sz, text=f"{self.voxel_size.get():.2f} m",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#3B8ED0")
        self._size_lbl.pack(side="right")
        ctk.CTkSlider(sett, from_=0.1, to=2.0, number_of_steps=38,
                       variable=self.voxel_size,
                       command=self._on_size_change).pack(fill="x", padx=15, pady=(0, 12))

        # Formats
        fmt = ctk.CTkFrame(sett, fg_color="transparent")
        fmt.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkLabel(fmt, text="Output Formats:", font=ctk.CTkFont(size=13)).pack(
            side="left", padx=(0, 12))
        ctk.CTkCheckBox(fmt, text="OBJ", variable=self.generate_obj,
                         font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 14))
        ctk.CTkCheckBox(fmt, text="PLY", variable=self.generate_ply,
                         font=ctk.CTkFont(size=13)).pack(side="left", padx=(0, 14))
        ctk.CTkCheckBox(fmt, text="glTF/GLB", variable=self.generate_gltf,
                         font=ctk.CTkFont(size=13)).pack(side="left")

        opt = ctk.CTkFrame(sett, fg_color="transparent")
        opt.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkCheckBox(opt, text="Binary PLY (smaller file)",
                         variable=self.binary_ply,
                         font=ctk.CTkFont(size=13)).pack(side="left")

        # Colour options
        clr = ctk.CTkFrame(sett, fg_color="transparent")
        clr.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkCheckBox(clr, text="🎨 Colour by surface type",
                         variable=self.use_colors, font=ctk.CTkFont(size=13),
                         command=self._toggle_legend).pack(side="left")

        ctk.CTkButton(
            sett, text="📄 Load Colour Scheme (JSON)",
            command=self._load_color_scheme,
            fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"), height=32,
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # ── Colour legend ────────────────────────────────────────
        self._legend_frame = ctk.CTkFrame(main)
        self._legend_frame.pack(fill="x", pady=(0, 15))
        self._build_legend()

        # ── Convert button ───────────────────────────────────────
        self.convert_btn = ctk.CTkButton(
            main, text="🚀 Convert to 3D", command=self._start_conversion,
            height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.convert_btn.pack(fill="x", pady=(0, 15))

        # ── Progress ─────────────────────────────────────────────
        prog = ctk.CTkFrame(main)
        prog.pack(fill="x", pady=(0, 15))
        self._progress_lbl = ctk.CTkLabel(
            prog, text="Ready", font=ctk.CTkFont(size=13))
        self._progress_lbl.pack(anchor="w", padx=15, pady=(15, 10))
        self._progress_bar = ctk.CTkProgressBar(prog)
        self._progress_bar.pack(fill="x", padx=15, pady=(0, 15))
        self._progress_bar.set(0)

        # ── Statistics panel (hidden until first conversion) ─────
        self._stats_frame = ctk.CTkFrame(main)
        # Not packed yet — shown after first successful conversion
        self._stats_text: Optional[ctk.CTkTextbox] = None
        self._preview_btn: Optional[ctk.CTkButton] = None

        # ── Log ──────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(main)
        log_frame.pack(fill="both", expand=True)

        lh = ctk.CTkFrame(log_frame, fg_color="transparent")
        lh.pack(fill="x", padx=15, pady=(15, 10))
        ctk.CTkLabel(lh, text="📋 Log",
                      font=ctk.CTkFont(size=16, weight="bold")).pack(side="left")
        ctk.CTkButton(lh, text="📂 Output Folder",
                       command=lambda: self._open_folder(self.output_dir),
                       fg_color="transparent", border_width=1,
                       text_color=("gray10", "gray90"),
                       height=28, width=140).pack(side="right")

        self._log = ctk.CTkTextbox(
            log_frame, font=ctk.CTkFont(family="Consolas", size=12), height=150)
        self._log.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self._log_msg("Welcome to Voxel Viewer!")
        self._log_msg(f"Input folder : {self.input_dir}")
        self._log_msg(f"Output folder: {self.output_dir}")
        self._log_msg("")

    # ── Legend builder ────────────────────────────────────────────

    def _build_legend(self) -> None:
        """(Re)build the colour-legend grid inside ``_legend_frame``."""
        for w in self._legend_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(self._legend_frame, text="🎨 Colour Legend",
                      font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=15, pady=(15, 10))

        # Use the active palette (dynamic if built from CSV classes)
        palette = get_active_palette()

        # If no dynamic palette yet, show a placeholder hint
        if not palette or palette == {k: v for k, v in SURFACE_COLORS.items() if k != "default"}:
            # Check if the palette is just the static defaults (no CSV loaded)
            from .colors import _DYNAMIC_COLORS
            if not _DYNAMIC_COLORS:
                ctk.CTkLabel(
                    self._legend_frame,
                    text="Colours will be auto-generated from your CSV classes.\n"
                         "Select a CSV file to see the legend.",
                    font=ctk.CTkFont(size=13),
                    text_color="gray",
                    justify="left",
                ).pack(anchor="w", padx=15, pady=(0, 15))
                return

        grid = ctk.CTkFrame(self._legend_frame, fg_color="transparent")
        grid.pack(fill="x", padx=15, pady=(0, 15))

        items = list(palette.items())
        # Append the default/other entry
        default_color = SURFACE_COLORS.get("default", (200, 200, 200))
        items.append(("Other / Unknown", default_color))

        for idx, (name, (r, g, b)) in enumerate(items):
            frm = ctk.CTkFrame(grid, fg_color="transparent")
            frm.grid(row=idx // 3, column=idx % 3, padx=10, pady=5, sticky="w")
            ctk.CTkLabel(frm, text="  ", fg_color=f"#{r:02x}{g:02x}{b:02x}",
                          corner_radius=4, width=24, height=24).pack(
                side="left", padx=(0, 8))
            ctk.CTkLabel(frm, text=name, font=ctk.CTkFont(size=12)).pack(side="left")

    # ── Statistics panel ─────────────────────────────────────────

    def _show_statistics(self, stats: Dict[str, Any]) -> None:
        """Build (or update) the statistics panel."""
        if self._stats_text is None:
            # First time — build the panel
            ctk.CTkLabel(self._stats_frame, text="📊 Statistics",
                          font=ctk.CTkFont(size=16, weight="bold")).pack(
                anchor="w", padx=15, pady=(15, 5))
            self._stats_text = ctk.CTkTextbox(
                self._stats_frame,
                font=ctk.CTkFont(family="Consolas", size=12), height=160)
            self._stats_text.pack(fill="x", padx=15, pady=(0, 5))

            btn_row = ctk.CTkFrame(self._stats_frame, fg_color="transparent")
            btn_row.pack(fill="x", padx=15, pady=(0, 15))
            self._preview_btn = ctk.CTkButton(
                btn_row, text="🔍 Preview 3D",
                command=self._show_preview,
                height=36, width=160)
            self._preview_btn.pack(side="left")
            if not HAS_MATPLOTLIB:
                self._preview_btn.configure(state="disabled")
                ctk.CTkLabel(btn_row, text="(install matplotlib for preview)",
                              font=ctk.CTkFont(size=11),
                              text_color="gray").pack(side="left", padx=10)

            # Insert the frame before the log frame
            self._stats_frame.pack(fill="x", pady=(0, 15),
                                    before=self._log.master)

        # Populate
        self._stats_text.configure(state="normal")
        self._stats_text.delete("1.0", "end")
        lines = [
            f"Total voxels : {stats['total_voxels']:,}",
            "",
            "Bounding box :",
        ]
        for ax in ("X", "Y", "Z"):
            lo = stats["bbox_min"][ax]
            hi = stats["bbox_max"][ax]
            lines.append(f"  {ax}: {lo:,.2f}  →  {hi:,.2f}")
        c = stats["centroid"]
        lines.append(f"\nCentroid     : ({c['X']:,.2f}, {c['Y']:,.2f}, {c['Z']:,.2f})")
        if "surface_counts" in stats:
            total = stats["total_voxels"]
            lines.append(f"\nSurface types ({stats['unique_surfaces']}):")
            for stype, cnt in sorted(
                stats["surface_counts"].items(), key=lambda x: -x[1]
            ):
                pct = cnt / total * 100
                lines.append(f"  {stype:25} {cnt:>10,}  ({pct:5.1f}%)")
        self._stats_text.insert("1.0", "\n".join(lines))
        self._stats_text.configure(state="disabled")

    # ── 3-D Preview (matplotlib) ─────────────────────────────────

    def _show_preview(self) -> None:
        if not HAS_MATPLOTLIB or self._last_df is None:
            return

        import pandas as pd

        df = self._last_df
        col = detect_color_column(list(df.columns))

        # Sample for performance
        max_pts = 10_000
        if len(df) > max_pts:
            sample = df.sample(max_pts, random_state=42)
        else:
            sample = df

        x = sample["X"].values
        y = sample["Y"].values
        z = sample["Z"].values

        # Colours
        from .colors import get_color_for_tag

        if col and col in sample.columns:
            rgb = np.array(
                [get_color_for_tag(t) for t in sample[col].values],
                dtype=np.float64,
            ) / 255.0
        else:
            rgb = np.full((len(sample), 3), 0.6)

        # Build figure
        fig = Figure(figsize=(9, 7), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=rgb, s=1, alpha=0.7)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Voxel Preview ({len(sample):,} / {len(df):,} points)")

        # Pop-up window
        win = ctk.CTkToplevel(self)
        win.title("Voxel Preview")
        win.geometry("920x720")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        toolbar.pack(fill="x")

    # ── Colour scheme loader ─────────────────────────────────────

    def _load_color_scheme(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Colour Scheme JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if not path:
            return
        try:
            load_color_scheme(path)
            self._build_legend()
            self._log_msg(f"Loaded colour scheme: {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load colour scheme:\n{exc}")

    # ── Helpers ──────────────────────────────────────────────────

    def _toggle_legend(self) -> None:
        if self.use_colors.get():
            self._legend_frame.pack(fill="x", pady=(0, 15),
                                     before=self.convert_btn)
        else:
            self._legend_frame.pack_forget()

    def _on_size_change(self, value: float) -> None:
        self._size_lbl.configure(text=f"{float(value):.2f} m")

    def _log_msg(self, msg: str) -> None:
        self._log.insert("end", f"{msg}\n")
        self._log.see("end")

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Voxels CSV",
            initialdir=str(self.input_dir),
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.selected_file.set(path)
            self._log_msg(f"Selected CSV: {path}")
            # Peek at CSV to build the dynamic palette immediately
            self._peek_csv_classes(path)

    def _browse_binvox(self) -> None:
        """Browse for a single .binvox file."""
        path = filedialog.askopenfilename(
            title="Select Binvox File",
            initialdir=str(self.input_dir),
            filetypes=[("Binvox Files", "*.binvox"), ("All Files", "*.*")],
        )
        if path:
            self.selected_file.set(path)
            self._log_msg(f"Selected binvox: {path}")
            try:
                model = read_binvox(path)
                self._log_msg(
                    f"  → {model.n_voxels:,} voxels, "
                    f"grid {model.header.dims}, "
                    f"voxel_size={model.header.voxel_size:.4f}m"
                )
            except Exception as exc:
                self._log_msg(f"⚠ Could not read binvox: {exc}")

    def _browse_binvox_folder(self) -> None:
        """Browse for a folder containing .binvox files."""
        path = filedialog.askdirectory(
            title="Select Folder with Binvox Files",
            initialdir=str(self.input_dir),
        )
        if path:
            from pathlib import Path as P
            n_files = len(list(P(path).glob("*.binvox")))
            if n_files == 0:
                messagebox.showerror("Error", f"No .binvox files found in:\n{path}")
                return
            self.selected_file.set(path)
            self._log_msg(f"Selected binvox folder: {path} ({n_files} files)")

    def _peek_csv_classes(self, csv_path: str) -> None:
        """Read just enough of the CSV to discover classes and build the legend."""
        try:
            import pandas as pd
            # Read only a handful of rows to detect columns, then unique values
            # For large files, nrows speeds this up dramatically
            df_peek = pd.read_csv(csv_path, nrows=0)  # just headers
            col = detect_color_column(list(df_peek.columns))
            if col:
                # Read only the class column for all rows (much faster than full CSV)
                df_col = pd.read_csv(csv_path, usecols=[col])
                unique_classes = [str(v) for v in df_col[col].dropna().unique()]
                build_dynamic_palette(unique_classes)
                self._build_legend()
                self._log_msg(
                    f"🎨 Detected {len(unique_classes)} classes in '{col}': "
                    f"{unique_classes}"
                )
            else:
                self._log_msg("ℹ No surface-type column detected in CSV headers")
        except Exception as exc:
            self._log_msg(f"⚠ Could not peek at CSV classes: {exc}")

    def _open_folder(self, folder: Path) -> None:
        try:
            folder.mkdir(exist_ok=True)
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(folder)],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform == "win32":
                os.startfile(str(folder))  # type: ignore[attr-defined]
        except Exception as exc:
            self._log_msg(f"Could not open folder: {exc}")

    def _set_progress(self, msg: str, value: float) -> None:
        self.after(0, lambda: self._progress_lbl.configure(text=msg))
        self.after(0, lambda: self._progress_bar.set(value))
        self._log_msg(msg)

    # ── Conversion ───────────────────────────────────────────────

    def _start_conversion(self) -> None:
        input_path = self.selected_file.get()
        if not input_path:
            messagebox.showerror("Error", "Please select a CSV file, binvox file, or binvox folder first!")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("Error", f"Path not found:\n{input_path}")
            return
        if not (self.generate_obj.get() or self.generate_ply.get()
                or self.generate_gltf.get()):
            messagebox.showerror("Error", "Select at least one output format!")
            return

        self.convert_btn.configure(state="disabled", text="⏳ Converting…")
        t = threading.Thread(target=self._run_conversion, args=(input_path,),
                              daemon=True)
        t.start()

    def _detect_input_type(self, path: str) -> str:
        """Detect whether the input is a CSV, a single binvox, or a binvox folder."""
        p = Path(path)
        if p.is_dir():
            return "binvox_folder"
        elif p.suffix.lower() == ".binvox":
            return "binvox_single"
        else:
            return "csv"

    def _run_conversion(self, input_path: str) -> None:
        try:
            input_type = self._detect_input_type(input_path)

            # ── Read input ───────────────────────────────────────
            if input_type == "binvox_single":
                self._set_progress("Reading binvox file…", 0.05)
                model = read_binvox(input_path)
                df = model.to_dataframe(include_source=True)
                name = Path(input_path).stem
                self._log_msg(
                    f"📦 Binvox: {model.n_voxels:,} voxels, "
                    f"grid {model.header.dims}, "
                    f"voxel_size={model.header.voxel_size:.4f}m"
                )

            elif input_type == "binvox_folder":
                self._set_progress("Reading binvox folder…", 0.05)
                df, models = read_binvox_directory(input_path)
                name = Path(input_path).name + "_combined"
                summary = binvox_summary(models)
                self._log_msg(
                    f"📂 Binvox folder: {summary['file_count']} files, "
                    f"{len(df):,} voxels (after dedup)"
                )
                # If source_file column exists, we can use it for coloring
                if "source_file" in df.columns and self.use_colors.get():
                    unique_sources = [str(v) for v in df["source_file"].unique()]
                    build_dynamic_palette(unique_sources)
                    self.after(0, self._build_legend)
                    self._log_msg(f"🎨 Coloring by source file ({len(unique_sources)} files)")

            else:  # CSV
                self._set_progress("Reading CSV…", 0.05)
                df = read_csv_smart(input_path)
                name = Path(input_path).stem

            self._last_df = df

            col = detect_color_column(list(df.columns))
            use_clr = self.use_colors.get()
            step = 0.1

            # Build dynamic palette from actual CSV classes
            if use_clr and col and col in df.columns:
                unique_classes = [str(v) for v in df[col].dropna().unique()]
                build_dynamic_palette(unique_classes)
                self._log_msg(f"🎨 Built palette for {len(unique_classes)} classes: {unique_classes}")
                # Rebuild legend to show actual data classes
                self.after(0, self._build_legend)
            elif use_clr and not col and input_type == "csv":
                self._log_msg("⚠ No surface-type column found — using default grey")

            # PLY
            if self.generate_ply.get():
                self._set_progress("Generating PLY…", step)
                ply = self.output_dir / f"{name}_pointcloud.ply"
                csv_to_ply(df, str(ply), use_clr, col, self.binary_ply.get())
                self._log_msg(f"✓ {ply}")
                step += 0.25

            # OBJ
            if self.generate_obj.get():
                self._set_progress("Generating OBJ…", step)
                obj = self.output_dir / f"{name}_voxels.obj"
                voxel_size = self.voxel_size.get()
                stats = csv_to_obj(df, str(obj), voxel_size, use_clr, col)
                self._log_msg(f"✓ {obj}")
                mtl = obj.with_suffix(".mtl")
                if mtl.exists():
                    self._log_msg(f"✓ {mtl}")
                step += 0.25

            # glTF
            if self.generate_gltf.get():
                self._set_progress("Generating GLB…", step)
                glb = self.output_dir / f"{name}_voxels.glb"
                csv_to_gltf(df, str(glb), self.voxel_size.get(), use_clr, col)
                self._log_msg(f"✓ {glb}")
                step += 0.25

            # Statistics
            self._set_progress("Computing statistics…", 0.95)
            stats = compute_statistics(df, col)
            self._last_stats = stats
            self.after(0, lambda: self._show_statistics(stats))

            self._set_progress("Done!", 1.0)
            self._log_msg("\n🎉 Conversion complete!\n")

            self.after(100, lambda: messagebox.showinfo(
                "Success",
                f"Files saved to:\n{self.output_dir}\n\n"
                f"Total voxels: {stats['total_voxels']:,}",
            ))

        except FileNotFoundError as exc:
            self._log_msg(f"❌ File not found: {exc}")
            self.after(100, lambda: messagebox.showerror("File Error", str(exc)))
        except ValueError as exc:
            self._log_msg(f"❌ Invalid data: {exc}")
            self.after(100, lambda: messagebox.showerror("Data Error", str(exc)))
        except Exception as exc:
            self._log_msg(f"❌ Error: {exc}")
            self._log_msg(traceback.format_exc())
            self.after(100, lambda: messagebox.showerror("Error", str(exc)))
        finally:
            self.after(100, lambda: self.convert_btn.configure(
                state="normal", text="🚀 Convert to 3D"))


# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    app = VoxelViewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
