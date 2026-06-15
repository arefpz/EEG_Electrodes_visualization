# Interactive 3D EEG 10-10 Cap Comparison Viewer

A Python Dash application for visualizing EEG electrode locations on a rotatable 3D head surface. The tool is intended for quick review of EEG caps, electrode labels, and approximate cortical regions under 10-10 montage electrodes.

This project was designed with assistance from ChatGPT. The main goal was to create a practical viewer for fast inspection of EEG cap layouts, not a full anatomical source-localization pipeline.

Current app version: `2.0`

---

## What this tool does

This viewer lets you:

- Display a default EEG 10-10 reference cap.
- Upload your own EEG coordinate CSV as the reference cap.
- Upload a second EEG coordinate CSV as a comparison cap.
- Compare two cap layouts electrode by electrode.
- See matched-electrode distance values in millimeters.
- Rotate, zoom, and inspect a 3D head model in the browser.
- Click or hover on electrodes to show electrode information.
- Adjust electrode size, label size, opacity, colors, mesh detail, and projection offset interactively.
- Select standard camera views: default, left, right, top, front, and back.
- View electrodes on either:
  - an adaptive phantom shell, or
  - a realistic public template head mesh using MNE fsaverage when available.

The app preserves the original input coordinates but projects electrodes outward for display, so electrodes remain visible and easier to select on the outer head surface.

---

## Example use cases

- Quick review of an EEG 10-10 cap layout.
- Checking whether electrode labels are in expected locations.
- Comparing two coordinate files for the same EEG cap.
- Inspecting how much the same electrode moved between two coordinate sets.
- Reviewing custom digitized electrode coordinates before further processing.
- Teaching or demonstrating approximate EEG electrode coverage.

---

## Important limitation

This is a visualization and quality-review tool.

It is not intended for:

- clinical diagnosis,
- source localization,
- subject-specific MRI coregistration,
- definitive cortical labeling,
- replacement of an anatomical atlas workflow.

The cortical-region descriptions are approximate and heuristic. For anatomical claims, use subject-specific MRI, digitized electrodes, proper coregistration, and an atlas-based labeling workflow.

---

## Repository files

Recommended repository layout:

```text
.
|-- eeg_1010_head_dash.py
|-- default_eeg_1010_reference.csv
|-- example_comparison_eeg_1010_mni_shifted.csv
|-- README.md
```

File descriptions:

| File | Description |
|---|---|
| `eeg_1010_head_dash.py` | app script. |
| `default_eeg_1010_reference.csv` | Default 10-10 reference electrode coordinate file. Place it in the same folder as the script. |
| `example_comparison_eeg_1010_mni_shifted.csv` | Example comparison cap with small coordinate shifts for testing two-cap comparison. |
| `README.md` | GitHub information page and user instructions. |

---

## Requirements

Use Python 3.10 or newer.

Python packages:

```text
dash
plotly
pandas
numpy
mne
```

---

## Installation

### 1. Download or clone the repository

Using Git:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Or manually download these files and place them in the same folder:

```text
eeg_1010_head_dash.py
default_eeg_1010_reference.csv
example_comparison_eeg_1010_mni_shifted.csv
```

---

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install --upgrade dash plotly pandas numpy mne
```

---

## How to run the app

### Option A: Run with the default reference cap

Make sure the script and default CSV are in the same folder:

```text
eeg_1010_head_dash.py
default_eeg_1010_reference.csv
```

Run:

```bash
python eeg_1010_head_dash.py
```

The terminal should print a local address similar to:

```text
Interactive 3D EEG 10-10 cap comparison viewer v2.0
Open http://127.0.0.1:8050
```

Open this address in your browser:

```text
http://127.0.0.1:8050
```

You should see the default 10-10 EEG cap on a 3D head surface.

---


### Option C: Run with your own reference CSV

```bash
python eeg_1010_head_dash.py --csv-reference my_reference_cap.csv --csv-units mm
```

Then open:

```text
http://127.0.0.1:8050
```

You can also use the legacy alias:

```bash
python eeg_1010_head_dash.py --csv my_reference_cap.csv --csv-units mm
```

---

### Option D: Run with both reference and comparison CSV files

```bash
python eeg_1010_head_dash.py --csv-reference default_eeg_1010_reference.csv --csv-compare example_comparison_eeg_1010_mni_shifted.csv --csv-units mm
```

Then open:

```text
http://127.0.0.1:8050
```

The plot will show two electrode sets:

- reference cap,
- comparison cap.

Electrodes with the same label are matched automatically. The app reports the distance between matched electrodes in millimeters.

---

### Option E: Upload CSV files inside the browser

1. Start the app:

   ```bash
   python eeg_1010_head_dash.py
   ```

2. Open:

   ```text
   http://127.0.0.1:8050
   ```

3. In the left control panel, use:

   - `Upload reference CSV` to replace the default reference cap.
   - `Upload comparison CSV` to add a second cap for comparison.

4. Select the coordinate units for each CSV:

   - `Auto`
   - `mm`
   - `m`

5. Review the updated 3D plot.

---

## CSV format

The CSV file must include electrode labels and 3D coordinates.

### Minimal format

```csv
label,x,y,z
F3,-45,51,63
F4,45,51,63
C3,-58,-15,76
C4,58,-15,76
Pz,0,-68,74
Oz,0,-96,31
```

### Alternative accepted format

```csv
electrode,mni_x,mni_y,mni_z
F3,-45,51,63
F4,45,51,63
C3,-58,-15,76
C4,58,-15,76
Pz,0,-68,74
Oz,0,-96,31
```

### Required columns

The required columns are one label column and three coordinate columns.

Accepted label column names include:

```text
label, electrode, channel, ch, name, electrode_label
```

Accepted coordinate column names include:

```text
x, y, z
mni_x, mni_y, mni_z
x_mm, y_mm, z_mm
ras_x, ras_y, ras_z
coord_x, coord_y, coord_z
```

### Optional columns

You can add optional anatomical information:

```text
region, hemisphere, confidence, region_note
```

Example:

```csv
label,x,y,z,region,hemisphere,confidence,region_note
F3,-45,51,63,"left dorsolateral prefrontal cortex",left,"user-defined","Provided by user CSV"
C3,-58,-15,76,"left primary motor and somatosensory cortex",left,"user-defined","Provided by user CSV"
Oz,0,-96,31,"midline occipital visual cortex",midline,"user-defined","Provided by user CSV"
```

When these optional columns are present, the app uses them in the electrode information panel.

---

## Coordinate convention

The app expects MNI/fsaverage RAS-style coordinates:

| Axis | Meaning |
|---|---|
| `x` | Right is positive, left is negative |
| `y` | Anterior/front is positive, posterior/back is negative |
| `z` | Superior/up is positive |

Coordinate units can be:

- millimeters, or
- meters.

The app internally works in millimeters. Use the unit controls if the displayed head size looks incorrect.

---

## Two-cap comparison

The comparison workflow is based on electrode labels.

For example, if both files contain `F3`, the app matches those two `F3` points and computes:

```text
dx = comparison_x - reference_x
dy = comparison_y - reference_y
dz = comparison_z - reference_z
distance = sqrt(dx^2 + dy^2 + dz^2)
```

The summary line above the plot reports:

- number of reference electrodes,
- number of comparison electrodes,
- number of matched labels,
- mean matched-electrode distance,
- maximum matched-electrode distance,
- label with the maximum distance.

The side information panel reports the same information for the clicked or hovered electrode.

Comparison distances are calculated from the original coordinates. The outward-projected display coordinates are used only for visualization and hit testing.

---

## Head model options

The app includes two head display modes.

### 1. Adaptive phantom shell

This is the default mode. The app builds a smooth head-like shell around the electrode coordinates. It is lightweight and does not require downloading additional template files.

Use this mode when you want a fast visualization and robust electrode selection.

### 2. Realistic public template head mesh

This mode uses the MNE fsaverage outer scalp/head surface when available. In the app, select:

```text
Realistic public template head mesh
```

Keep this option checked if you want the app to fetch the template when it is missing:

```text
Allow MNE to fetch fsaverage if missing
```

The first use may require internet access so MNE can download the fsaverage files. If the mesh is unavailable, the app falls back to a procedural head mesh and reports this in the plot title.

---

## Projection behavior

Electrode coordinates from MNI space or a CSV file can be slightly inside or outside the rendered scalp surface. For visualization, the app automatically projects electrode display positions to the outer head layer.

The app keeps two coordinate sets:

| Coordinate type | Purpose |
|---|---|
| Original `x, y, z` | Preserved input coordinate. Used for reporting and comparison distances. |
| Projected display coordinate | Used for plotting, hover, click selection, and visual placement on the outer scalp. |

This projection is intentional. The app is designed for visual review, so electrode visibility and selectability are prioritized.

---

## Interactive controls

The browser control panel includes:

### Data controls

- Upload reference CSV.
- Reload default reference CSV.
- Upload comparison CSV.
- Clear comparison cap.
- Select units separately for reference and comparison CSVs.

### View controls

- Default view.
- Left view.
- Right view.
- Top view.
- Front view.
- Back view.

### Head model controls

- Adaptive phantom shell.
- Realistic public template head mesh.
- Allow MNE to fetch fsaverage if missing.

### Display controls

- Show or hide scalp/head surface.
- Show or hide electrode labels.
- Show or hide matched-electrode displacement lines.
- Show or hide nose and ear landmarks.
- Show or hide head mesh edges.

### Size and opacity controls

- Electrode size.
- Label font size.
- Invisible click/hover hitbox size.
- Reference cap opacity.
- Comparison cap opacity.
- Head opacity.
- Outward projection offset.
- Head mesh detail.
- Head scale.

### Color controls

The app uses RGB slider bars instead of typed color codes. You can set:

- reference cap color,
- comparison cap color,
- label color,
- head color,
- background color.

Each color control has red, green, and blue sliders plus a live preview.

---

## How to interact with the plot

| Action | Control |
|---|---|
| Rotate | Left-click and drag |
| Pan | Right-click and drag |
| Zoom | Mouse wheel or trackpad scroll |
| Select electrode | Click an electrode marker or its larger invisible hitbox |
| Preview electrode info | Hover over an electrode |
| Change viewpoint | Use the view buttons |
| Compare two caps | Upload a comparison CSV with matching labels |

When an electrode is selected, the information panel shows:

- cap name,
- electrode label,
- original coordinate,
- projected display coordinate,
- whether the label is matched between caps,
- distance between reference and comparison caps,
- comparison-minus-reference delta,
- approximate hemisphere,
- approximate cortical region,
- source file.

---

## Command-line options

Run:

```bash
python eeg_1010_head_dash.py --help
```

Available options:

| Option | Description |
|---|---|
| `--csv` | Legacy alias for `--csv-reference`. |
| `--csv-reference` | Path to the reference electrode CSV. |
| `--csv-compare` | Path to the comparison electrode CSV. |
| `--csv-units` | Units for both CSV files unless overridden. Choices: `auto`, `mm`, `m`. Default: `auto`. |
| `--reference-units` | Units for the reference CSV only. Choices: `auto`, `mm`, `m`. |
| `--compare-units` | Units for the comparison CSV only. Choices: `auto`, `mm`, `m`. |
| `--host` | Dash host. Default: `127.0.0.1`. |
| `--port` | Dash port. Default: `8050`. |
| `--debug` | Run the Dash app in debug mode. |

Example using different units for the two CSV files:

```bash
python eeg_1010_head_dash.py --csv-reference reference_mm.csv --reference-units mm --csv-compare comparison_m.csv --compare-units m
```

Example using a different port:

```bash
python eeg_1010_head_dash.py --port 8060
```

Then open:

```text
http://127.0.0.1:8060
```

---

## Troubleshooting

### The browser does not open automatically

Open the local URL manually:

```text
http://127.0.0.1:8050
```

### Port 8050 is already in use

Run on another port:

```bash
python eeg_1010_head_dash.py --port 8060
```

Then open:

```text
http://127.0.0.1:8060
```

### The head or scalp surface is not visible

In the app:

1. Make sure `Show scalp/head surface` is checked.
2. Make sure `Show head mesh edges` is checked.
3. Increase `Head opacity`.
4. Try the `Adaptive phantom shell` mode.
5. Try the `Realistic public template head mesh` mode.
6. Keep `Allow MNE to fetch fsaverage if missing` checked if you want the public template mesh.

### The realistic head template does not load

The realistic template mode depends on MNE fsaverage files. The first use may require internet access. If the template cannot be fetched or found, the app uses a procedural fallback head mesh.

The fallback is still usable for visualization and comparison.

### Electrodes are hard to click

Try these controls:

- increase `Invisible click/hover hitbox extra size`,
- increase `Electrode size`,
- reduce `Head opacity`,
- keep `Show scalp/head surface` enabled but use lower opacity,
- use a lower `Head mesh detail` value.

The head surface is configured not to capture hover information, and electrode hitboxes are drawn to improve selection.

### The CSV does not load

Check that the file is a real comma-separated `.csv` file and has at least:

```csv
label,x,y,z
F3,-45,51,63
C3,-58,-15,76
Oz,0,-96,31
```

Also check that coordinate columns contain numeric values.

### The head looks much too large or too small

The unit setting may be wrong.

If your coordinates are in millimeters:

```bash
python eeg_1010_head_dash.py --csv-reference my_file.csv --csv-units mm
```

If your coordinates are in meters:

```bash
python eeg_1010_head_dash.py --csv-reference my_file.csv --csv-units m
```

You can also adjust `Head scale` inside the browser.

### The two caps do not compare

The app matches electrodes by label. Check that labels are spelled the same in both files.

For example, these match:

```text
F3  and  F3
Cz  and  Cz
Oz  and  Oz
```

These do not match:

```text
F3  and  F03
Cz  and  CZ_reference
Oz  and  O_z
```

---

## Suggested quick test

After installing the dependencies, run:

```bash
python eeg_1010_head_dash.py --csv-reference default_eeg_1010_reference.csv --csv-compare example_comparison_eeg_1010_mni_shifted.csv --csv-units mm
```

Open:

```text
http://127.0.0.1:8050
```

Then:

1. Rotate the head.
2. Click `F3`, `C3`, `Pz`, or `Oz`.
3. Change the reference and comparison colors using the RGB bars.
4. Increase or decrease cap opacity.
5. Switch between `Adaptive phantom shell` and `Realistic public template head mesh`.
6. Press `Left`, `Right`, and `Top` view buttons.
7. Inspect the distance between matched electrodes.

---

## Privacy note

The app runs locally on your machine through Dash. CSV files uploaded through the browser are sent to the local Dash server process running on your computer. The realistic head mesh option may request MNE fsaverage files if the template is missing and downloading is enabled.

---

## Acknowledgment

This tool was designed with assistance from ChatGPT. It was created mainly for quick visual review of EEG caps and for checking whether default or custom EEG electrode coordinates look reasonable in 3D.

---

## License

Add your preferred license before publishing the repository. Common options include:

- MIT
- BSD-3-Clause
- Apache-2.0
- GPL-3.0
