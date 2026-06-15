#!/usr/bin/env python3
"""
Interactive 3D EEG 10-10 montage viewer on a head-shaped phantom.

Features
--------
- Default 10-10 electrode positions are loaded from MNE's standard_1005 montage,
  filtered to a conventional 10-10 subset, and converted from meters to MNI/fsaverage
  millimeters.
- User can provide a CSV file at launch or upload one in the web app.
- 3D head phantom can be rotated, panned, and zoomed in the browser.
- Electrode labels are displayed on the electrode markers by default.
- Clicking an electrode fills a textbox with approximate underlying cortical regions.

CSV format
----------
Required columns, case-insensitive:
    label,x,y,z
or:
    electrode,mni_x,mni_y,mni_z
Optional region column:
    region

Coordinates should be MNI/fsaverage RAS coordinates:
    x: right positive, y: anterior positive, z: superior positive.
Units may be millimeters or meters. Use --csv-units or the app radio buttons.

Install
-------
    pip install mne dash plotly pandas numpy

Run
---
    python eeg_1010_head_dash.py
    python eeg_1010_head_dash.py --csv my_electrodes.csv --csv-units mm
"""

from __future__ import annotations

import argparse
import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update


APP_TITLE = "Interactive 3D EEG 10-10 Head Phantom"

# Conventional 10-10 subset. MNE's standard_1005 contains these plus denser 10-05 sites.
TEN_TEN_LABELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]


@dataclass(frozen=True)
class RegionGuess:
    hemisphere: str
    region: str
    confidence: str
    note: str


def canonical_label(label: object) -> str:
    """Normalize common EEG labels while keeping readable 10-10 capitalization."""
    raw = str(label).strip()
    if not raw:
        return raw
    upper = raw.upper().replace(" ", "")
    # Keep standard lowercase p in Fp labels.
    if upper.startswith("FP"):
        upper = "Fp" + upper[2:]
    elif upper.startswith("AF"):
        upper = "AF" + upper[2:]
    elif upper.startswith("FT"):
        upper = "FT" + upper[2:]
    elif upper.startswith("FC"):
        upper = "FC" + upper[2:]
    elif upper.startswith("TP"):
        upper = "TP" + upper[2:]
    elif upper.startswith("CP"):
        upper = "CP" + upper[2:]
    elif upper.startswith("PO"):
        upper = "PO" + upper[2:]
    else:
        # One-letter rows such as F3, C4, Pz, O1, T8, Iz.
        upper = upper[0] + upper[1:].lower() if len(upper) > 1 else upper
        upper = upper.replace("Z", "z")
    upper = upper.replace("Z", "z")
    return upper


def label_prefix(label: str) -> str:
    match = re.match(r"([A-Za-z]+)", label)
    return match.group(1) if match else ""


def label_number(label: str) -> int | None:
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else None


def infer_hemisphere(label: str, x_mm: float | None = None) -> str:
    """Infer hemisphere from the label first, then from x coordinate if needed."""
    lab = canonical_label(label)
    if lab.endswith("z") or lab in {"Cz", "Fz", "Pz", "Oz", "Iz", "AFz", "FCz", "CPz", "POz"}:
        return "midline"
    num = label_number(lab)
    if num is not None:
        if num % 2 == 1:
            return "left hemisphere"
        return "right hemisphere"
    if x_mm is not None and np.isfinite(x_mm):
        if x_mm < -3:
            return "left hemisphere"
        if x_mm > 3:
            return "right hemisphere"
    return "uncertain hemisphere"


# Specific mappings for clinically common EEG sites. The fallback rules below handle the rest.
SPECIFIC_REGION_MAP = {
    "F3": "left dorsolateral prefrontal cortex, approximately middle frontal gyrus, BA 9/46",
    "F4": "right dorsolateral prefrontal cortex, approximately middle frontal gyrus, BA 9/46",
    "F5": "left lateral prefrontal cortex, often near inferior/middle frontal gyrus",
    "F6": "right lateral prefrontal cortex, often near inferior/middle frontal gyrus",
    "F7": "left inferior frontal cortex, ventrolateral prefrontal cortex, frontal operculum",
    "F8": "right inferior frontal cortex, ventrolateral prefrontal cortex, frontal operculum",
    "Fz": "medial frontal cortex, pre-SMA/SMA region",
    "FC3": "left premotor cortex and precentral gyrus, near hand motor network",
    "FC4": "right premotor cortex and precentral gyrus, near hand motor network",
    "FCz": "supplementary motor area and medial premotor cortex",
    "C3": "left primary motor and primary somatosensory cortex, hand area approximation",
    "C4": "right primary motor and primary somatosensory cortex, hand area approximation",
    "Cz": "midline sensorimotor cortex, leg/foot area and supplementary motor area approximation",
    "CP3": "left postcentral gyrus and superior parietal cortex, somatosensory association area",
    "CP4": "right postcentral gyrus and superior parietal cortex, somatosensory association area",
    "CPz": "midline parietal cortex, posterior cingulate/precuneus approximation",
    "P3": "left posterior parietal cortex, superior/inferior parietal lobule",
    "P4": "right posterior parietal cortex, superior/inferior parietal lobule",
    "Pz": "precuneus and posterior parietal midline cortex",
    "PO3": "left parieto-occipital cortex and dorsal visual association cortex",
    "PO4": "right parieto-occipital cortex and dorsal visual association cortex",
    "POz": "midline parieto-occipital cortex and visual association cortex",
    "O1": "left occipital visual cortex, primary/extrastriate visual cortex approximation",
    "O2": "right occipital visual cortex, primary/extrastriate visual cortex approximation",
    "Oz": "midline occipital cortex, primary visual cortex approximation",
    "T7": "left lateral temporal cortex, superior temporal gyrus/auditory-language network approximation",
    "T8": "right lateral temporal cortex, superior temporal gyrus/auditory network approximation",
    "TP7": "left posterior temporal cortex and temporoparietal junction approximation",
    "TP8": "right posterior temporal cortex and temporoparietal junction approximation",
    "P7": "left temporoparietal/occipitotemporal cortex approximation",
    "P8": "right temporoparietal/occipitotemporal cortex approximation",
    "Fp1": "left frontopolar prefrontal cortex, orbitofrontal/frontopolar approximation",
    "Fp2": "right frontopolar prefrontal cortex, orbitofrontal/frontopolar approximation",
    "Fpz": "midline frontopolar prefrontal cortex",
}


def infer_region(label: str, x_mm: float | None = None, y_mm: float | None = None, z_mm: float | None = None) -> RegionGuess:
    """Return an approximate cortical region for a 10-10 electrode label.

    This is a heuristic scalp-to-cortex label, not a source-localization result.
    For precise anatomical assignment, use the subject MRI, digitized electrodes,
    coregistration, and a cortical atlas.
    """
    lab = canonical_label(label)
    hemisphere = infer_hemisphere(lab, x_mm)

    if lab in SPECIFIC_REGION_MAP:
        return RegionGuess(
            hemisphere=hemisphere,
            region=SPECIFIC_REGION_MAP[lab],
            confidence="medium for standard 10-10 anatomy; low for individual anatomy",
            note="Specific 10-10 site rule used.",
        )

    prefix = label_prefix(lab)
    num = label_number(lab)
    lateral = num is not None and num >= 7
    very_lateral = num is not None and num >= 9

    if prefix == "Fp":
        region = "frontopolar and orbitofrontal prefrontal cortex"
    elif prefix == "AF":
        region = "anterior prefrontal cortex; lateral sites approach lateral prefrontal/frontal eye field territory"
    elif prefix == "F":
        if lab.endswith("z") or num in {1, 2}:
            region = "medial/superior frontal cortex, pre-SMA/SMA and superior frontal gyrus approximation"
        elif num in {3, 4}:
            region = "dorsolateral prefrontal cortex, middle frontal gyrus approximation"
        elif num in {5, 6, 7, 8}:
            region = "lateral/inferior frontal cortex, ventrolateral prefrontal cortex approximation"
        else:
            region = "very lateral frontal pole/anterior temporal transition area"
    elif prefix == "FC":
        if lab.endswith("z") or num in {1, 2}:
            region = "medial premotor cortex and supplementary motor area"
        elif num in {3, 4}:
            region = "dorsal premotor cortex and precentral gyrus"
        else:
            region = "lateral premotor cortex, frontal operculum, and inferior precentral region"
    elif prefix == "C":
        if lab.endswith("z") or num in {1, 2}:
            region = "central sensorimotor strip, medial hand/leg representation approximation"
        elif num in {3, 4}:
            region = "primary motor and primary somatosensory cortex, hand area approximation"
        else:
            region = "lateral sensorimotor cortex, inferior precentral/postcentral gyri"
    elif prefix == "CP":
        if lab.endswith("z") or num in {1, 2}:
            region = "postcentral and superior parietal cortex; midline sites approach precuneus"
        elif num in {3, 4, 5, 6}:
            region = "somatosensory association cortex and posterior parietal cortex"
        else:
            region = "inferior parietal and temporoparietal junction approximation"
    elif prefix == "P":
        if lab.endswith("z") or num in {1, 2}:
            region = "precuneus and posterior parietal cortex"
        elif num in {3, 4, 5, 6}:
            region = "superior/inferior parietal lobule, posterior parietal association cortex"
        elif lateral:
            region = "temporoparietal and occipitotemporal cortex"
        else:
            region = "parietal association cortex"
    elif prefix == "PO":
        region = "parieto-occipital cortex and extrastriate visual association cortex"
    elif prefix == "O":
        region = "occipital visual cortex, primary and extrastriate visual cortex approximation"
    elif prefix == "T":
        if very_lateral:
            region = "very lateral temporal scalp area near temporal muscle and lateral temporal cortex"
        else:
            region = "lateral temporal cortex, superior/middle temporal gyri"
    elif prefix == "FT":
        region = "inferior frontal, frontal opercular, and anterior temporal transition area"
    elif prefix == "TP":
        region = "posterior temporal cortex and temporoparietal junction"
    elif prefix == "I":
        region = "inion/posterior occipital midline; inferior occipital/cerebellar transition nearby"
    else:
        region = "unknown or nonstandard electrode label; provide a region column in the CSV for exact annotation"

    note = "Rule based on electrode row/prefix."
    if lateral:
        note += " Very lateral sites may be farther from cortex and more affected by skull/scalp geometry."

    return RegionGuess(
        hemisphere=hemisphere,
        region=f"{hemisphere}; {region}" if hemisphere != "midline" else f"midline; {region}",
        confidence="low-to-medium heuristic",
        note=note,
    )


def is_blank_value(value: object) -> bool:
    """Treat empty strings and pandas missing markers as blank."""
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def add_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure label, region, hemisphere, confidence, and note columns exist."""
    out = df.copy()
    out["label"] = out["label"].map(canonical_label)

    if "region" not in out.columns:
        out["region"] = ""
    if "hemisphere" not in out.columns:
        out["hemisphere"] = ""
    if "confidence" not in out.columns:
        out["confidence"] = ""
    if "region_note" not in out.columns:
        out["region_note"] = ""

    for idx, row in out.iterrows():
        guess = infer_region(row["label"], row.get("x"), row.get("y"), row.get("z"))
        if is_blank_value(row.get("region", "")):
            out.at[idx, "region"] = guess.region
        if is_blank_value(row.get("hemisphere", "")):
            out.at[idx, "hemisphere"] = guess.hemisphere
        if is_blank_value(row.get("confidence", "")):
            out.at[idx, "confidence"] = guess.confidence
        if is_blank_value(row.get("region_note", "")):
            out.at[idx, "region_note"] = guess.note
    return out


def load_default_mni_1010() -> pd.DataFrame:
    """Load default 10-10 positions from MNE standard_1005 and convert meters to millimeters."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError(
            "Default montage requires MNE. Install it with: pip install mne. "
            "Alternatively start with --csv your_electrodes.csv."
        ) from exc

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos_m = montage.get_positions()["ch_pos"]
    pos_lookup = {canonical_label(name): np.asarray(pos, dtype=float) for name, pos in ch_pos_m.items()}

    records = []
    missing = []
    for label in TEN_TEN_LABELS:
        key = canonical_label(label)
        if key in pos_lookup:
            xyz_mm = pos_lookup[key] * 1000.0
            records.append({"label": label, "x": xyz_mm[0], "y": xyz_mm[1], "z": xyz_mm[2]})
        else:
            missing.append(label)

    if not records:
        raise RuntimeError("No 10-10 labels were found in MNE's standard_1005 montage.")

    df = pd.DataFrame.from_records(records)
    df = add_region_columns(df)
    df.attrs["source"] = "MNE standard_1005 filtered to conventional 10-10 labels"
    if missing:
        df.attrs["missing"] = ", ".join(missing)
    return df


def _lower_col_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def _pick_column(colmap: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate.lower() in colmap:
            return colmap[candidate.lower()]
    return None


def normalize_coordinate_units(df: pd.DataFrame, units: str) -> pd.DataFrame:
    out = df.copy()
    coords = out[["x", "y", "z"]].astype(float)
    max_abs = float(np.nanmax(np.abs(coords.to_numpy())))

    if units == "auto":
        # MNI coordinates in meters are usually around 0.001 to 0.2.
        # MNI coordinates in millimeters are usually around 1 to 120.
        units = "m" if max_abs < 2.0 else "mm"

    if units == "m":
        out[["x", "y", "z"]] = coords * 1000.0
    elif units == "mm":
        out[["x", "y", "z"]] = coords
    else:
        raise ValueError("units must be one of: auto, mm, m")
    return out


def load_csv_dataframe(csv_text_or_path: str | Path, units: str = "auto", from_text: bool = False) -> pd.DataFrame:
    """Load user electrodes from CSV.

    Accepts label/name/electrode/channel columns and x/y/z or mni_x/mni_y/mni_z.
    """
    if from_text:
        source = io.StringIO(str(csv_text_or_path))
    else:
        source = Path(csv_text_or_path)

    raw = pd.read_csv(source)
    if raw.empty:
        raise ValueError("CSV is empty.")

    colmap = _lower_col_map(raw.columns)
    label_col = _pick_column(colmap, ["label", "electrode", "name", "ch_name", "channel", "site"])
    x_col = _pick_column(colmap, ["x", "mni_x", "x_mni", "ras_x"])
    y_col = _pick_column(colmap, ["y", "mni_y", "y_mni", "ras_y"])
    z_col = _pick_column(colmap, ["z", "mni_z", "z_mni", "ras_z"])

    if label_col is None or x_col is None or y_col is None or z_col is None:
        raise ValueError(
            "CSV must contain a label/electrode/name column plus x,y,z columns "
            "or mni_x,mni_y,mni_z columns."
        )

    region_col = _pick_column(colmap, ["region", "brain_region", "cortex", "cortical_region"])
    hemisphere_col = _pick_column(colmap, ["hemisphere", "hemi"])
    confidence_col = _pick_column(colmap, ["confidence", "certainty"])
    note_col = _pick_column(colmap, ["region_note", "note", "notes"])

    out = pd.DataFrame(
        {
            "label": raw[label_col].astype(str),
            "x": pd.to_numeric(raw[x_col], errors="coerce"),
            "y": pd.to_numeric(raw[y_col], errors="coerce"),
            "z": pd.to_numeric(raw[z_col], errors="coerce"),
        }
    )
    if region_col is not None:
        out["region"] = raw[region_col].astype(str).replace({"nan": ""})
    if hemisphere_col is not None:
        out["hemisphere"] = raw[hemisphere_col].astype(str).replace({"nan": ""})
    if confidence_col is not None:
        out["confidence"] = raw[confidence_col].astype(str).replace({"nan": ""})
    if note_col is not None:
        out["region_note"] = raw[note_col].astype(str).replace({"nan": ""})

    out = out.dropna(subset=["x", "y", "z"])
    if out.empty:
        raise ValueError("No valid numeric coordinates were found in the CSV.")

    out = normalize_coordinate_units(out, units=units)
    return add_region_columns(out)


def robust_center_and_radii(df: pd.DataFrame, scale: float = 1.08) -> tuple[np.ndarray, np.ndarray]:
    coords = df[["x", "y", "z"]].astype(float).to_numpy()
    center = np.nanmedian(coords, axis=0)
    # For MNI RAS, x is naturally centered around 0. Keep it centered if possible.
    if np.nanmin(coords[:, 0]) < 0 < np.nanmax(coords[:, 0]):
        center[0] = 0.0

    # Use robust percentiles so one custom point does not distort the phantom too much.
    lower = np.nanpercentile(coords, 3, axis=0)
    upper = np.nanpercentile(coords, 97, axis=0)
    radii = np.maximum((upper - lower) / 2.0, [55.0, 70.0, 65.0])
    radii *= scale

    # Put the vertical center slightly below the median electrode cloud to form a head-like scalp.
    zmin = float(np.nanmin(coords[:, 2]))
    zmax = float(np.nanmax(coords[:, 2]))
    center[2] = zmin + 0.45 * (zmax - zmin)
    radii[2] = max(radii[2], 0.62 * (zmax - zmin))

    return center.astype(float), radii.astype(float)


def make_head_surface(center: np.ndarray, radii: np.ndarray, n_theta: int = 60, n_phi: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Ellipsoid: +x right, +y anterior, +z superior.
    x = center[0] + radii[0] * np.sin(theta_grid) * np.cos(phi_grid)
    y = center[1] + radii[1] * np.sin(theta_grid) * np.sin(phi_grid)
    z = center[2] + radii[2] * np.cos(theta_grid)
    return x, y, z


def make_nose_mesh(center: np.ndarray, radii: np.ndarray) -> go.Mesh3d:
    n = 28
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    base_x = center[0] + 0.12 * radii[0] * np.cos(angles)
    base_y = np.full(n, center[1] + 0.96 * radii[1])
    base_z = center[2] - 0.12 * radii[2] + 0.16 * radii[2] * np.sin(angles)
    tip = np.array([[center[0], center[1] + 1.20 * radii[1], center[2] - 0.10 * radii[2]]])

    vertices = np.vstack([np.column_stack([base_x, base_y, base_z]), tip])
    tip_idx = n
    i, j, k = [], [], []
    for idx in range(n):
        i.append(idx)
        j.append((idx + 1) % n)
        k.append(tip_idx)

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i,
        j=j,
        k=k,
        name="nose/anterior",
        opacity=0.45,
        color="tan",
        hovertemplate="Nose / anterior direction<extra></extra>",
        showscale=False,
    )


def make_ear_surface(center: np.ndarray, radii: np.ndarray, sign: int) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 24)
    v = np.linspace(0.0, np.pi, 16)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = center[0] + sign * (1.02 * radii[0] + 0.06 * radii[0] * np.sin(vv) * np.cos(uu))
    y = center[1] - 0.02 * radii[1] + 0.10 * radii[1] * np.sin(vv) * np.sin(uu)
    z = center[2] - 0.08 * radii[2] + 0.18 * radii[2] * np.cos(vv)
    label = "right ear" if sign > 0 else "left ear"
    return go.Surface(
        x=x,
        y=y,
        z=z,
        name=label,
        opacity=0.30,
        colorscale=[[0, "tan"], [1, "tan"]],
        showscale=False,
        hovertemplate=f"{label}<extra></extra>",
    )


def figure_from_records(records: list[dict], show_labels: bool = True, head_scale: float = 1.08) -> go.Figure:
    df = add_region_columns(pd.DataFrame(records))
    center, radii = robust_center_and_radii(df, scale=head_scale)
    hx, hy, hz = make_head_surface(center, radii)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=hx,
            y=hy,
            z=hz,
            name="scalp phantom",
            opacity=0.28,
            colorscale=[[0, "rgb(230,190,160)"], [1, "rgb(230,190,160)"]],
            showscale=False,
            hovertemplate="Head-shaped scalp phantom<extra></extra>",
        )
    )
    fig.add_trace(make_nose_mesh(center, radii))
    fig.add_trace(make_ear_surface(center, radii, sign=-1))
    fig.add_trace(make_ear_surface(center, radii, sign=1))

    customdata = df[["label", "region", "hemisphere", "confidence", "region_note"]].astype(str).to_numpy()
    mode = "markers+text" if show_labels else "markers"
    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["z"],
            mode=mode,
            name="EEG electrodes",
            text=df["label"],
            textposition="top center",
            textfont={"size": 9, "color": "black"},
            customdata=customdata,
            marker={
                "size": 5,
                "color": "crimson",
                "line": {"color": "black", "width": 1},
            },
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Region: %{customdata[1]}<br>"
                "MNI scalp xyz: %{x:.1f}, %{y:.1f}, %{z:.1f} mm"
                "<extra></extra>"
            ),
        )
    )

    # Orientation labels. These are separate text points, not electrodes.
    orientation_x = [center[0], center[0], center[0] + 1.12 * radii[0], center[0] - 1.12 * radii[0]]
    orientation_y = [center[1] + 1.28 * radii[1], center[1] - 1.15 * radii[1], center[1], center[1]]
    orientation_z = [center[2], center[2], center[2], center[2]]
    orientation_text = ["Anterior +Y", "Posterior -Y", "Right +X", "Left -X"]
    fig.add_trace(
        go.Scatter3d(
            x=orientation_x,
            y=orientation_y,
            z=orientation_z,
            mode="text",
            text=orientation_text,
            textfont={"size": 11, "color": "gray"},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    lim = float(max(radii) * 1.55)
    fig.update_layout(
        title=APP_TITLE,
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        height=780,
        scene={
            "xaxis": {"title": "MNI x, mm (right +)", "range": [center[0] - lim, center[0] + lim]},
            "yaxis": {"title": "MNI y, mm (anterior +)", "range": [center[1] - lim, center[1] + lim]},
            "zaxis": {"title": "MNI z, mm (superior +)", "range": [center[2] - lim, center[2] + lim]},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": -1.9, "z": 1.2}},
        },
        legend={"x": 0.01, "y": 0.99},
        uirevision="keep-camera",
    )
    return fig


def records_from_df(df: pd.DataFrame) -> list[dict]:
    cols = ["label", "x", "y", "z", "region", "hemisphere", "confidence", "region_note"]
    clean = df.copy()
    for col in cols:
        if col not in clean.columns:
            clean[col] = ""
    return clean[cols].to_dict("records")


def parse_uploaded_csv(contents: str, units: str) -> pd.DataFrame:
    """Decode Dash upload contents and parse as CSV."""
    if not contents or "," not in contents:
        raise ValueError("No upload content found.")
    _, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded).decode("utf-8-sig")
    return load_csv_dataframe(decoded, units=units, from_text=True)


def electrode_info_text(label: str, records: list[dict]) -> str:
    df = add_region_columns(pd.DataFrame(records))
    matches = df[df["label"].astype(str) == str(label)]
    if matches.empty:
        return "Clicked item was not found in the electrode table. Try clicking directly on an electrode marker."

    row = matches.iloc[0]
    return (
        f"Electrode: {row['label']}\n"
        f"MNI scalp coordinate, mm: x={row['x']:.2f}, y={row['y']:.2f}, z={row['z']:.2f}\n"
        f"Hemisphere: {row.get('hemisphere', '')}\n\n"
        f"Likely underlying cortical region(s):\n{row.get('region', '')}\n\n"
        f"Confidence: {row.get('confidence', '')}\n"
        f"Note: {row.get('region_note', '')}\n\n"
        "Important: this is a 10-10 scalp-to-cortex heuristic on a phantom head, not a subject-specific "
        "anatomical or source-localization result. For precise cortical labeling, use digitized electrodes, "
        "the subject MRI, coregistration, and an atlas."
    )


def build_app(initial_df: pd.DataFrame, head_scale: float = 1.08) -> Dash:
    initial_records = records_from_df(initial_df)
    app = Dash(__name__)
    app.title = APP_TITLE

    app.layout = html.Div(
        [
            html.H2(APP_TITLE, style={"marginBottom": "6px"}),
            html.Div(
                "Rotate with left mouse drag, pan with right mouse drag, zoom with scroll. "
                "Click an electrode marker or label to show the approximate cortical region.",
                style={"marginBottom": "12px", "color": "#333"},
            ),
            dcc.Store(id="electrode-data-store", data=initial_records),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id="head-graph",
                                figure=figure_from_records(initial_records, show_labels=True, head_scale=head_scale),
                                config={"displaylogo": False, "scrollZoom": True},
                                style={"height": "800px"},
                            ),
                        ],
                        style={"flex": "3", "minWidth": "720px"},
                    ),
                    html.Div(
                        [
                            html.H3("Electrode information"),
                            dcc.Textarea(
                                id="electrode-info-text",
                                value="Click an electrode to display its MNI coordinate and approximate underlying cortical region.",
                                readOnly=True,
                                style={
                                    "width": "100%",
                                    "height": "260px",
                                    "fontFamily": "monospace",
                                    "fontSize": "13px",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                            html.H3("Custom CSV"),
                            html.Div(
                                "CSV columns: label,x,y,z. Optional: region, hemisphere, confidence, region_note.",
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                            dcc.RadioItems(
                                id="csv-units",
                                options=[
                                    {"label": "auto units", "value": "auto"},
                                    {"label": "mm", "value": "mm"},
                                    {"label": "m", "value": "m"},
                                ],
                                value="auto",
                                inline=True,
                                style={"marginBottom": "8px"},
                            ),
                            dcc.Upload(
                                id="upload-coords",
                                children=html.Div(["Drag and drop or click to upload a CSV"]),
                                style={
                                    "width": "100%",
                                    "height": "68px",
                                    "lineHeight": "68px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "marginBottom": "8px",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload-status", style={"fontSize": "13px", "color": "#333"}),
                            html.H3("Display"),
                            dcc.Checklist(
                                id="label-toggle",
                                options=[{"label": "Show electrode labels", "value": "labels"}],
                                value=["labels"],
                            ),
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "340px",
                            "padding": "8px 16px",
                            "borderLeft": "1px solid #ddd",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "8px", "alignItems": "stretch", "flexWrap": "wrap"},
            ),
        ],
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
    )

    @app.callback(
        Output("electrode-data-store", "data"),
        Output("upload-status", "children"),
        Input("upload-coords", "contents"),
        Input("csv-units", "value"),
        State("upload-coords", "filename"),
        prevent_initial_call=True,
    )
    def update_uploaded_data(contents: str | None, units: str, filename: str | None):
        if contents is None:
            return no_update, no_update
        try:
            uploaded = parse_uploaded_csv(contents, units=units)
            status = f"Loaded {len(uploaded)} electrodes from {filename or 'uploaded CSV'} using units={units}."
            return records_from_df(uploaded), status
        except Exception as exc:
            return no_update, f"CSV load failed: {exc}"

    @app.callback(
        Output("head-graph", "figure"),
        Input("electrode-data-store", "data"),
        Input("label-toggle", "value"),
    )
    def update_figure(records: list[dict], label_toggle: list[str]):
        show_labels = "labels" in (label_toggle or [])
        return figure_from_records(records, show_labels=show_labels, head_scale=head_scale)

    @app.callback(
        Output("electrode-info-text", "value"),
        Input("head-graph", "clickData"),
        State("electrode-data-store", "data"),
    )
    def update_textbox(click_data: dict | None, records: list[dict]):
        if not click_data or not click_data.get("points"):
            return "Click an electrode to display its MNI coordinate and approximate underlying cortical region."
        point = click_data["points"][0]
        custom = point.get("customdata")
        if not custom:
            return "Click directly on an electrode marker or electrode label."
        label = custom[0] if isinstance(custom, (list, tuple)) else str(custom)
        return electrode_info_text(str(label), records)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--csv", type=str, default=None, help="Optional electrode CSV with label,x,y,z columns.")
    parser.add_argument("--csv-units", choices=["auto", "mm", "m"], default="auto", help="Coordinate units for --csv.")
    parser.add_argument("--head-scale", type=float, default=1.08, help="Scale factor for the scalp phantom around electrodes.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash host.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port.")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.csv:
        initial_df = load_csv_dataframe(args.csv, units=args.csv_units, from_text=False)
    else:
        initial_df = load_default_mni_1010()

    print(f"Loaded {len(initial_df)} electrodes.")
    print(f"Open http://{args.host}:{args.port} in your browser.")
    app = build_app(initial_df, head_scale=args.head_scale)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
