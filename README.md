# 🎲 Voxel Viewer

Convert **voxel CSV** and **binvox** files to coloured 3D formats — **OBJ** (mesh), **PLY** (point cloud), and **glTF/GLB** (web) — with **dynamic, automatic coloring** based on whatever classes exist in your data.

Built as a standalone companion to the [Urban-Auto-Vox](https://github.com/atsaad/Urban-Auto-Vox) pipeline but works with **any** voxel dataset.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| **Multi-format input** | CSV files with `X, Y, Z` columns **and** `.binvox` files (single file or entire directory) |
| **3 output formats** | OBJ + MTL, PLY (ASCII / binary), glTF/GLB |
| **Dynamic colouring** | Auto-generates visually distinct colours for whatever classes exist in your data (IFC, CityGML, custom, or any other scheme) |
| **Vectorised engine** | NumPy broadcasting + `np.savetxt` — hundreds of thousands of voxels in seconds |
| **Binary PLY** | 3–5× smaller files, instant I/O |
| **glTF/GLB** | View in any browser (three.js, Cesium, online viewers) |
| **GUI** | Modern CustomTkinter interface with live colour legend, 3D preview (matplotlib), and statistics panel |
| **CLI** | Full `argparse` interface with stdin pipe support |
| **Custom colours** | Override auto-generated colours with a JSON file — your overrides take priority |
| **Statistics** | Voxel count, bounding box, centroid, surface breakdown |
| **Binvox reader** | Native parser (zero extra dependencies) with automatic Y↔Z axis swap for GIS convention |
| **Pip-installable** | `pip install -e .` — includes `voxel-viewer` CLI command |
| **Tested** | 77 pytest tests covering colours, geometry, binvox parsing, and all export formats |

---

## 📦 Installation

### Option A — pip (recommended)

```bash
git clone https://github.com/atsaad/voxel-viewer.git
cd voxel-viewer
pip install -e ".[gui]"      # core + GUI + matplotlib
```

### Option B — start.sh (auto-creates venv)

```bash
chmod +x start.sh
./start.sh
```

### Option C — core only (no GUI)

```bash
pip install -e .
```

---

## 🚀 CLI Usage

### CSV input

```bash
# Convert with defaults (all formats, coloured)
voxel-viewer data/voxels_output.csv

# PLY only, binary, custom voxel size
voxel-viewer data.csv -f ply --binary-ply -s 0.25

# Specify which column to colour by
voxel-viewer data.csv --column object_type

# Pipe from stdin
cat data.csv | voxel-viewer - -f obj -o my_output

# Just statistics, no conversion
voxel-viewer data.csv -f none --stats

# Custom colour scheme
voxel-viewer data.csv --color-scheme my_colors.json
```

### Binvox input

```bash
# Single .binvox file
voxel-viewer model.binvox -f all

# Entire directory of .binvox files (merged + deduplicated)
voxel-viewer voxels_folder/ -f all -o combined_output

# Directory with statistics
voxel-viewer voxels_folder/ -f ply --binary-ply --stats
```

### Launch GUI

```bash
voxel-viewer --gui
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `input` | — | Path to CSV, `.binvox` file, directory of `.binvox` files, or `-` for stdin |
| `-f`, `--format` | `all` | `obj`, `ply`, `gltf`, `glb`, `all`, `none` |
| `-s`, `--scale` | `0.5` | Voxel cube size in metres |
| `-c`, `--column` | auto | Column name for surface type |
| `-o`, `--output` | auto | Output base path |
| `--no-color` | — | Disable colouring |
| `--binary-ply` | — | Binary PLY format |
| `--color-scheme` | — | Path to JSON colour scheme |
| `--stats` | — | Print dataset statistics |
| `--chunk-size` | — | CSV chunk size (rows) for large files |
| `--gui` | — | Launch GUI |

---

## 🖥️ GUI

```bash
voxel-viewer --gui
# or
./start.sh
```

The GUI supports both **CSV** and **binvox** inputs:

- **Browse CSV** — select a CSV file; the colour legend updates immediately from detected classes
- **Browse Binvox File** — select a single `.binvox` file
- **Browse Binvox Folder** — select a directory; all `.binvox` files are merged and deduplicated
- **Voxel size slider** (0.1 – 2.0 m)
- **Format checkboxes**: OBJ, PLY, glTF/GLB
- **Binary PLY toggle**
- **Dynamic colour legend** — auto-generated from your data's actual classes
- **Load custom colour scheme** (JSON overrides)
- **Statistics panel** — voxel count, bounding box, centroid, per-class breakdown
- **3D scatter preview** (matplotlib, up to 10K sampled points)

---

## 📊 Input Formats

### CSV

Any CSV with `X`, `Y`, `Z` columns. An optional class column enables colouring.

**Pipeline v6.1 format (IFC classes):**

```csv
voxel_position,vox_geom,gmlid,object_type,X,Y,Z
1,01010...,C91037...,IfcBeam,691034.62,5336144.85,514.52
2,01010...,C91037...,IfcWall,691034.68,5336144.91,514.52
```

**CityGML format:**

```csv
X,Y,Z,tag_value
691021.60,5336035.54,527.21,WallSurface
691021.60,5336035.54,527.71,RoofSurface
```

The viewer auto-detects the class column (`object_type`, `tag_value`, `surface_type`, or `type`). Override with `--column <name>`.

### Binvox

The binary `.binvox` format produced by [cuda_voxelizer](https://github.com/Forceflow/cuda_voxelizer) or [binvox](https://www.patrickmin.com/binvox/). Each file contains a 3D voxel grid with:

- **Grid dimensions** (e.g., `140 × 140 × 140`)
- **Translate + scale** for world-space reconstruction
- **RLE-encoded** occupancy data

The viewer handles the **Y↔Z axis swap** automatically (binvox uses Y-up graphics convention; the viewer outputs Z-up GIS convention).

When loading a **directory**, all files are merged into a single combined model with:
- Automatic **deduplication** of overlapping voxels
- Per-file **colour coding** (each source file gets a distinct colour)

---

## 🎨 Dynamic Colour System

The colour system works with **any** classification scheme — not just predefined CityGML classes.

### How it works

1. **Read** the unique class values from the data (CSV column or binvox source files)
2. **Check** for user overrides (JSON file) — these take top priority
3. **Check** the built-in defaults (CityGML colours) — used if they match
4. **Auto-generate** visually distinct colours for everything else (golden-angle hue spacing)

### Built-in defaults (CityGML)

| Surface | Colour | RGB |
|---------|--------|-----|
| WallSurface | Brown/Tan | (139, 119, 101) |
| RoofSurface | Firebrick | (178, 34, 34) |
| GroundSurface | Forest Green | (34, 139, 34) |
| Window | Sky Blue | (135, 206, 235) |
| Door | Saddle Brown | (139, 69, 19) |
| Other | Light Grey | (200, 200, 200) |

### Custom colour scheme

Create a JSON file with RGB arrays or hex strings:

```json
{
  "IfcWall": [180, 160, 140],
  "IfcBeam": "#4488CC",
  "IfcSlab": [100, 100, 100],
  "IfcRoof": "#B22222"
}
```

Load via CLI: `--color-scheme my_colors.json`
Load via GUI: **📄 Load Colour Scheme (JSON)** button

> **Note:** Any classes not in your JSON file still get auto-generated colours. Your overrides take priority — the two systems combine seamlessly.

---

## 📤 Output Files

| Format | File | Use Case |
|--------|------|----------|
| OBJ + MTL | `*_voxels.obj` + `.mtl` | MeshLab, Blender, CloudCompare |
| PLY | `*_pointcloud.ply` | CloudCompare, ParaView |
| GLB | `*_voxels.glb` | Web viewers, three.js, Cesium |

### Viewing tips

- **CloudCompare** — best for large point clouds, shows PLY vertex colours
- **MeshLab** — load `.obj`, MTL colours auto-applied
- **Blender** — import OBJ with materials, or drag-drop GLB
- **Online** — drag `.glb` into [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com/)

---

## 🐍 Python API

```python
from voxel_viewer import csv_to_obj, csv_to_ply, csv_to_gltf, compute_statistics

# Convert from CSV
stats = csv_to_obj("data.csv", "output.obj", scale=0.5)

# Or pass a DataFrame directly
import pandas as pd
df = pd.read_csv("data.csv")
csv_to_ply(df, "output.ply", binary=True)
csv_to_gltf(df, "output.glb")

# Statistics
stats = compute_statistics(df, color_column="object_type")
print(stats["total_voxels"])
print(stats["surface_counts"])
```

### Binvox API

```python
from voxel_viewer import read_binvox, read_binvox_directory

# Single file
model = read_binvox("building.binvox")
print(model.header)           # BinvoxHeader(dims=(140,140,140), ...)
print(model.n_voxels)         # 1,381
df = model.to_dataframe()     # DataFrame with X, Y, Z (GIS convention)

# Entire directory (merged + deduplicated)
df, models = read_binvox_directory("voxels/")
print(f"{len(df):,} combined voxels from {len(models)} files")
```

### Dynamic colouring API

```python
from voxel_viewer import build_dynamic_palette, get_color_for_tag, get_active_palette

# Build palette from your classes
palette = build_dynamic_palette(["IfcWall", "IfcBeam", "IfcSlab", "IfcColumn"])

# Look up a colour
r, g, b = get_color_for_tag("IfcBeam")

# Get the full active palette (for legends, etc.)
all_colors = get_active_palette()
```

---

## 🧪 Development

```bash
# Install with dev dependencies
pip install -e ".[all]"

# Run tests
pytest -v

# Run with coverage
pytest --cov=voxel_viewer --cov-report=term-missing
```

**Test coverage:** 77 tests covering:
- Hex/RGB conversion and roundtrips
- Dynamic palette generation (distinct colours, user overrides, reset)
- Column auto-detection
- CSV validation and reading (chunked, legacy, DataFrame passthrough)
- Geometry generation (vertices, quad faces, triangle faces)
- OBJ, PLY (ASCII + binary), and GLB export
- Binvox parsing (synthetic files, RLE decoding, directory merging, deduplication)
- Integration tests with real sample `.binvox` files

---

## 📁 Project Structure

```
voxel-viewer/
├── src/voxel_viewer/
│   ├── __init__.py        # Public API exports
│   ├── __main__.py        # CLI entry point (argparse)
│   ├── binvox.py          # Binvox parser (RLE, header, world coords)
│   ├── colors.py          # Dynamic colour system, JSON config, palette generation
│   ├── converter.py       # Vectorised OBJ/PLY/glTF export + statistics
│   └── gui.py             # CustomTkinter GUI (CSV + binvox, preview, legend)
├── tests/
│   ├── conftest.py        # Shared fixtures, auto-reset colours
│   ├── test_binvox.py     # Binvox reader tests (34 tests)
│   ├── test_colors.py     # Colour system tests (17 tests)
│   ├── test_converter.py  # Converter tests (26 tests)
│   └── fixtures/          # Sample CSVs for testing
├── colors.json            # Example colour scheme
├── pyproject.toml         # Package config, entry points, dependencies
├── requirements.txt       # Flat dependency list
├── start.sh               # One-click GUI launcher (creates venv)
├── LICENSE                # MIT
└── README.md
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).
