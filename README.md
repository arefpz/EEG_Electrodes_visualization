# Interactive 3D EEG 10-10 Head Phantom

A local, browser-based Python tool for visualizing EEG electrode positions on a rotatable 3D head-shaped phantom. The default layout uses a conventional 10-10 EEG montage in MNI/fsaverage-style coordinates, and users can also load their own electrode coordinates from a CSV file.

> Designed with ChatGPT as an AI-assisted visualization prototype.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Dash](https://img.shields.io/badge/App-Dash-lightgrey)
![Plotly](https://img.shields.io/badge/3D-Plotly-lightgrey)
![EEG](https://img.shields.io/badge/EEG-10--10%20Montage-green)

---

## Overview

This tool provides an interactive 3D viewer for EEG electrode locations. It is intended for EEG education, experiment planning, visualization, and communication of approximate scalp-to-cortex relationships.

The viewer displays:

- A 3D head-shaped phantom surface
- Default 10-10 EEG electrode locations
- Electrode labels directly on the electrodes
- Interactive rotation, zooming, and panning
- Clickable electrodes with region descriptions
- Optional custom electrode coordinates loaded from CSV

When an electrode is clicked, the app displays an approximate anatomical interpretation, such as whether the electrode is near dorsolateral prefrontal cortex, motor cortex, somatosensory cortex, parietal cortex, temporal cortex, or visual cortex.

---

## Main Features

### Interactive 3D head visualization

The head phantom can be rotated, zoomed, and inspected directly in the browser using Plotly 3D graphics.

### EEG 10-10 montage support

The default electrode set is based on a conventional 10-10 subset derived from MNE-Python's `standard_1005` montage. The coordinates are represented in MNI/fsaverage-style space and converted to millimeters for visualization.

### Electrode labels on the scalp

Each electrode is shown as a labeled 3D marker. Labels are visible on the electrodes, making the montage easy to inspect without hovering over every marker.

### Click-to-inspect brain-region text box

Clicking an electrode updates the information panel with:

- Electrode name
- MNI-style scalp coordinate
- Approximate hemisphere
- Approximate cortical region under the electrode
- Confidence and interpretation notes

### Custom CSV coordinate loading

Users can provide their own coordinate file either:

1. At launch using the command line
2. Inside the app using the CSV upload area

This makes the app useful for custom digitized electrode coordinates, modified caps, simulation layouts, or selected electrode subsets.

---

## Installation

Install the required Python packages:

```bash
pip install mne dash plotly pandas numpy
```

Recommended environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install mne dash plotly pandas numpy
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install mne dash plotly pandas numpy
```

---

## Quick Start

Run the app with the default 10-10 montage:

```bash
python eeg_1010_head_dash.py
```

Then open the local Dash app in your browser:

```text
http://127.0.0.1:8050
```

---

## Running with a Custom CSV File

Use your own electrode coordinate file:

```bash
python eeg_1010_head_dash.py --csv my_electrodes.csv --csv-units mm
```

If your coordinates are in meters:

```bash
python eeg_1010_head_dash.py --csv my_electrodes.csv --csv-units m
```

The app also supports automatic unit detection:

```bash
python eeg_1010_head_dash.py --csv my_electrodes.csv --csv-units auto
```

---

## CSV Format

The CSV file must include electrode labels and 3D coordinates.

### Accepted column format 1

```csv
label,x,y,z,region
F3,-45,51,63,"left dorsolateral prefrontal cortex"
C3,-58,-15,76,"left primary motor/somatosensory hand area"
Oz,0,-96,31,"midline occipital visual cortex"
```

### Accepted column format 2

```csv
electrode,mni_x,mni_y,mni_z,region
F4,45,51,63,"right dorsolateral prefrontal cortex"
C4,58,-15,76,"right primary motor/somatosensory hand area"
Pz,0,-68,74,"precuneus/posterior parietal midline"
```

### Required columns

Use either:

```text
label,x,y,z
```

or:

```text
electrode,mni_x,mni_y,mni_z
```

### Optional columns

```text
region
hemisphere
confidence
region_note
```

If a `region` column is included, the app will use that custom annotation when displaying electrode information.

---

## Coordinate Convention

The expected coordinate system is MNI/fsaverage-style RAS space:

| Axis | Positive direction |
|---|---|
| x | Right |
| y | Anterior |
| z | Superior |

Coordinates may be provided in either millimeters or meters. Use `--csv-units mm`, `--csv-units m`, or `--csv-units auto`.

---

## Command-Line Options

```bash
python eeg_1010_head_dash.py [options]
```

| Option | Description | Default |
|---|---|---|
| `--csv` | Optional electrode CSV file | None |
| `--csv-units` | Coordinate units: `auto`, `mm`, or `m` | `auto` |
| `--head-scale` | Scale factor for the phantom head surface | `1.08` |
| `--host` | Dash server host | `127.0.0.1` |
| `--port` | Dash server port | `8050` |
| `--debug` | Run Dash in debug mode | Off |

Example:

```bash
python eeg_1010_head_dash.py --csv subject01_electrodes.csv --csv-units mm --port 8060
```

---

## Example Repository Structure

```text
eeg-1010-head-viewer/
├── eeg_1010_head_dash.py
├── example_custom_eeg_coords.csv
├── README.md
└── requirements.txt
```

Suggested `requirements.txt`:

```text
mne
dash
plotly
pandas
numpy
```

---

## How the Region Labels Work

The region descriptions are approximate heuristic labels based on standard 10-10 electrode naming and general scalp-to-cortex relationships.

Examples:

| Electrode | Approximate region |
|---|---|
| F3 | Left dorsolateral prefrontal cortex |
| F4 | Right dorsolateral prefrontal cortex |
| C3 | Left primary motor/somatosensory hand area |
| C4 | Right primary motor/somatosensory hand area |
| Cz | Midline sensorimotor cortex |
| Pz | Precuneus/posterior parietal cortex |
| O1/O2/Oz | Occipital visual cortex |
| T7/T8 | Lateral temporal cortex |

These labels are not source localization results. They are intended for visualization and orientation only.

---

## Important Limitations

This app does not replace subject-specific anatomical modeling.

The displayed cortical-region labels are approximate because:

- Scalp electrode positions do not map one-to-one to cortical generators.
- Individual head anatomy varies.
- Cap placement varies across participants.
- MNI/fsaverage coordinates are template coordinates, not subject-specific coordinates.
- EEG signals are affected by volume conduction and cannot be anatomically assigned from electrode position alone.

For precise anatomical interpretation, use:

- Digitized electrode coordinates
- Subject-specific MRI
- MRI-to-head coregistration
- A cortical atlas
- Forward modeling and source localization

This tool should be treated as an educational, visualization, and planning aid, not as a diagnostic or clinical decision-making system.

---

## Use Cases

This viewer may be useful for:

- Teaching EEG 10-10 electrode placement
- Planning EEG or tES/tDCS experiments
- Communicating electrode locations in papers or presentations
- Checking approximate electrode coverage
- Visualizing custom electrode subsets
- Demonstrating approximate scalp-to-cortex relationships

---

## Designed with ChatGPT

This tool was designed and initially drafted with assistance from ChatGPT. The goal was to create a practical, shareable Python visualization tool for interactive EEG electrode inspection using a default 10-10 montage and optional user-supplied coordinate files.

Users should review and validate the code before using it in research, teaching, or clinical-adjacent workflows.

---

## Citation / Acknowledgment

When sharing or adapting this tool, please acknowledge:

```text
Interactive 3D EEG 10-10 Head Phantom, designed with assistance from ChatGPT.
```

You may also cite the software packages used by the app, including MNE-Python, Dash, Plotly, NumPy, and pandas, according to their respective citation guidelines.

---

## License

Choose a license before publishing the repository. A common option for open-source sharing is the MIT License.

Suggested placeholder:

```text
MIT License

Copyright (c) YEAR YOUR NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the software
without restriction, subject to the full MIT License terms.
```

---

## Disclaimer

This project is provided for research visualization, education, and exploratory use. It is not a medical device, not a clinical localization tool, and not a substitute for expert neuroanatomical analysis.
