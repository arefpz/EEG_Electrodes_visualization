#!/usr/bin/env python3
"""
Interactive 3D EEG 10-10 cap comparison viewer.

Version: 2.0

Main features
-------------
- Reference EEG cap plus optional comparison EEG cap in the same 3D scene.
- Default reference cap can be loaded from default_eeg_1010_reference.csv placed
  next to this script. If that file is missing, the app falls back to MNE's
  standard_1005 montage, then to a lightweight approximate fallback.
- Upload a second CSV to compare electrode locations against the reference cap.
- Interactively adjust electrode size, label size, cap colors, cap opacity,
  head opacity, mesh detail, projection offset, and hitbox size.
- Color controls use RGB slider bars rather than typed hex strings.
- View buttons for default, left, right, top, front, and back camera views.
- Two head display modes:
    1. Adaptive phantom shell, similar to the earlier version.
    2. Realistic public template head mesh using MNE fsaverage outer_skin.surf
       with a visible rendered scalp layer and stronger default opacity
       when available or downloadable. If unavailable, a procedural fallback
       mesh is used and the figure title reports the fallback.
- Electrodes are projected to the fitted outer head surface for visualization
  and hit testing. Original coordinates are preserved in hover/click text.

Install
-------
    pip install dash plotly pandas numpy mne

Run
---
    python eeg_1010_head_dash_interactive_v3.py

Run with CSV files
------------------
    python eeg_1010_head_dash_interactive_v3.py --csv-reference reference.csv --csv-compare comparison.csv

Accepted CSV columns
--------------------
Required, case-insensitive:
    label,x,y,z
or:
    electrode,mni_x,mni_y,mni_z

Optional:
    region, hemisphere, confidence, region_note

Coordinate convention
---------------------
MNI/fsaverage RAS style coordinates:
    x: right positive
    y: anterior positive
    z: superior positive

Units may be millimeters or meters. Use --csv-units auto, mm, or m.

Important
---------
This app is for visualization and quick EEG cap review. It is not an anatomical
source-localization or subject-specific coregistration tool.
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import io
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


APP_TITLE = "Interactive 3D EEG 10-10 Cap Comparison Viewer"
APP_VERSION = "3.2"
DEFAULT_REFERENCE_CSV_NAME = "default_eeg_1010_reference.csv"

DEFAULT_REFERENCE_COLOR = "#d62728"
DEFAULT_COMPARISON_COLOR = "#1f77b4"
DEFAULT_LABEL_COLOR = "#111111"
DEFAULT_HEAD_COLOR = "#e8c1a4"
DEFAULT_BACKGROUND_COLOR = "#ffffff"
DEFAULT_DIFFERENCE_LINE_COLOR = "#555555"

DEFAULT_ELECTRODE_SIZE = 8
DEFAULT_LABEL_SIZE = 10
DEFAULT_HITBOX_EXTRA_SIZE = 22
DEFAULT_REFERENCE_OPACITY = 1.0
DEFAULT_COMPARISON_OPACITY = 0.78
DEFAULT_HEAD_OPACITY = 0.45
DEFAULT_MESH_DETAIL = 18
DEFAULT_PROJECTION_OFFSET_MM = 8.0
DEFAULT_HEAD_SCALE = 1.0

EMPTY_RECORDS: list[dict[str, Any]] = []

# Conventional 10-10 subset from MNE standard_1005.
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


@dataclass(frozen=True)
class HeadFit:
    center: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    source: str
    is_public_template: bool


@dataclass(frozen=True)
class HeadGeometry:
    kind: str
    fit: HeadFit
    status: str
    surface_x: np.ndarray | None = None
    surface_y: np.ndarray | None = None
    surface_z: np.ndarray | None = None
    vertices: np.ndarray | None = None
    faces: np.ndarray | None = None
    # Dense vertices used only for projecting electrodes to the outer skin.
    # The rendered mesh can be decimated separately for browser speed.
    projection_vertices: np.ndarray | None = None


SPECIFIC_REGION_MAP: dict[str, str] = {
    "Fp1": "left frontopolar prefrontal cortex; orbitofrontal/frontopolar approximation",
    "Fp2": "right frontopolar prefrontal cortex; orbitofrontal/frontopolar approximation",
    "Fpz": "midline frontopolar prefrontal cortex",
    "AF3": "left anterior prefrontal cortex; superior/middle frontal approximation",
    "AF4": "right anterior prefrontal cortex; superior/middle frontal approximation",
    "AFz": "midline anterior prefrontal cortex",
    "F3": "left dorsolateral prefrontal cortex, approximately middle frontal gyrus, BA 9/46",
    "F4": "right dorsolateral prefrontal cortex, approximately middle frontal gyrus, BA 9/46",
    "F5": "left lateral prefrontal cortex, often near inferior/middle frontal gyrus",
    "F6": "right lateral prefrontal cortex, often near inferior/middle frontal gyrus",
    "F7": "left inferior frontal cortex; ventrolateral prefrontal cortex/frontal operculum approximation",
    "F8": "right inferior frontal cortex; ventrolateral prefrontal cortex/frontal operculum approximation",
    "Fz": "medial frontal cortex; pre-SMA/SMA approximation",
    "FC3": "left premotor cortex and precentral gyrus, near hand motor network",
    "FC4": "right premotor cortex and precentral gyrus, near hand motor network",
    "FCz": "supplementary motor area and medial premotor cortex",
    "C3": "left primary motor and primary somatosensory cortex; hand area approximation",
    "C4": "right primary motor and primary somatosensory cortex; hand area approximation",
    "Cz": "midline sensorimotor cortex; leg/foot area and supplementary motor area approximation",
    "CP3": "left postcentral gyrus and superior parietal cortex; somatosensory association area",
    "CP4": "right postcentral gyrus and superior parietal cortex; somatosensory association area",
    "CPz": "midline parietal cortex; posterior cingulate/precuneus approximation",
    "P3": "left posterior parietal cortex; superior/inferior parietal lobule",
    "P4": "right posterior parietal cortex; superior/inferior parietal lobule",
    "Pz": "precuneus and posterior parietal midline cortex",
    "PO3": "left parieto-occipital cortex and dorsal visual association cortex",
    "PO4": "right parieto-occipital cortex and dorsal visual association cortex",
    "POz": "midline parieto-occipital cortex and visual association cortex",
    "O1": "left occipital visual cortex; primary/extrastriate visual cortex approximation",
    "O2": "right occipital visual cortex; primary/extrastriate visual cortex approximation",
    "Oz": "midline occipital cortex; primary visual cortex approximation",
    "T7": "left lateral temporal cortex; superior temporal gyrus/auditory-language network approximation",
    "T8": "right lateral temporal cortex; superior temporal gyrus/auditory network approximation",
    "TP7": "left posterior temporal cortex and temporoparietal junction approximation",
    "TP8": "right posterior temporal cortex and temporoparietal junction approximation",
    "P7": "left temporoparietal/occipitotemporal cortex approximation",
    "P8": "right temporoparietal/occipitotemporal cortex approximation",
}


# -----------------------------
# Label and region helpers
# -----------------------------


def canonical_label(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        return raw

    compact = raw.replace(" ", "").replace("_", "")
    upper = compact.upper()

    for prefix in ("FP", "AF", "FT", "FC", "TP", "CP", "PO"):
        if upper.startswith(prefix):
            rest = upper[len(prefix):]
            if prefix == "FP":
                return "Fp" + rest.replace("Z", "z")
            return prefix + rest.replace("Z", "z")

    first = upper[0]
    rest = upper[1:].replace("Z", "z")
    return first + rest


def label_prefix(label: str) -> str:
    match = re.match(r"([A-Za-z]+)", label)
    return match.group(1) if match else ""


def label_number(label: str) -> int | None:
    match = re.search(r"(\d+)", label)
    if not match:
        return None
    return int(match.group(1))


def infer_hemisphere(label: str, x_mm: float | None = None) -> str:
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


def infer_region(label: str, x_mm: float | None = None, y_mm: float | None = None, z_mm: float | None = None) -> RegionGuess:
    lab = canonical_label(label)
    hemi = infer_hemisphere(lab, x_mm)

    if lab in SPECIFIC_REGION_MAP:
        return RegionGuess(
            hemisphere=hemi,
            region=SPECIFIC_REGION_MAP[lab],
            confidence="medium for standard 10-10 anatomy; low for individual anatomy",
            note="Specific 10-10 electrode rule used.",
        )

    prefix = label_prefix(lab)
    num = label_number(lab)

    if prefix == "Fp":
        region = "frontopolar and orbitofrontal prefrontal cortex"
    elif prefix == "AF":
        region = "anterior prefrontal cortex; lateral sites approach lateral prefrontal/frontal eye field territory"
    elif prefix == "F":
        if lab.endswith("z") or num in {1, 2}:
            region = "medial/superior frontal cortex; pre-SMA/SMA and superior frontal gyrus approximation"
        elif num in {3, 4}:
            region = "dorsolateral prefrontal cortex; middle frontal gyrus approximation"
        elif num in {5, 6, 7, 8}:
            region = "lateral/inferior frontal cortex; ventrolateral prefrontal approximation"
        else:
            region = "very lateral frontal/anterior temporal transition area"
    elif prefix == "FC":
        if lab.endswith("z") or num in {1, 2}:
            region = "medial premotor cortex and supplementary motor area"
        elif num in {3, 4}:
            region = "dorsal premotor cortex and precentral gyrus"
        else:
            region = "lateral premotor cortex, frontal operculum, and inferior precentral region"
    elif prefix == "C":
        if lab.endswith("z") or num in {1, 2}:
            region = "central sensorimotor strip; medial hand/leg representation approximation"
        elif num in {3, 4}:
            region = "primary motor and primary somatosensory cortex; hand area approximation"
        else:
            region = "lateral sensorimotor cortex; inferior precentral/postcentral gyri"
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
            region = "superior/inferior parietal lobule; posterior parietal association cortex"
        else:
            region = "temporoparietal and occipitotemporal cortex"
    elif prefix == "PO":
        region = "parieto-occipital cortex and extrastriate visual association cortex"
    elif prefix == "O":
        region = "occipital visual cortex; primary/extrastriate visual cortex approximation"
    elif prefix in {"T", "TP", "FT"}:
        if prefix == "FT":
            region = "inferior frontal, insular-opercular, and anterior temporal transition region"
        elif prefix == "TP":
            region = "posterior temporal cortex and temporoparietal junction"
        else:
            region = "lateral temporal cortex; superior/middle temporal gyri approximation"
    elif prefix == "Iz":
        region = "inion/posterior occipital midline area"
    else:
        if y_mm is not None and z_mm is not None:
            if y_mm > 45:
                region = "anterior prefrontal/frontal cortex approximation"
            elif -20 <= y_mm <= 20 and z_mm > 45:
                region = "central sensorimotor cortex approximation"
            elif y_mm < -55:
                region = "occipital/parieto-occipital cortex approximation"
            else:
                region = "association cortex approximation based on scalp coordinate"
        else:
            region = "unknown; no standard 10-10 mapping rule matched"

    return RegionGuess(
        hemisphere=hemi,
        region=region,
        confidence="low to medium; heuristic scalp-to-cortex approximation",
        note="General 10-10 row/column rule used.",
    )


# -----------------------------
# Color helpers for slider bars
# -----------------------------


def clean_hex_color(value: Any, fallback: str = "#000000") -> str:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
            return text.lower()
    if isinstance(fallback, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", fallback.strip()):
        return fallback.strip().lower()
    return "#000000"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    text = clean_hex_color(hex_color)
    return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)


def rgb_values_to_hex(r: Any, g: Any, b: Any) -> str:
    def channel(v: Any) -> int:
        try:
            return int(np.clip(round(float(v)), 0, 255))
        except Exception:
            return 0
    rr, gg, bb = channel(r), channel(g), channel(b)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    a = min(max(float(alpha), 0.0), 1.0)
    return f"rgba({r},{g},{b},{a:.4f})"


def blend_hex_with(hex_color: str, other_hex: str = "#000000", amount: float = 0.35) -> str:
    """Blend two hex colors. amount=0 returns hex_color; amount=1 returns other_hex."""
    r1, g1, b1 = hex_to_rgb(hex_color)
    r2, g2, b2 = hex_to_rgb(other_hex)
    a = min(max(float(amount), 0.0), 1.0)
    r = int(round((1 - a) * r1 + a * r2))
    g = int(round((1 - a) * g1 + a * g2))
    b = int(round((1 - a) * b1 + a * b2))
    return f"#{r:02x}{g:02x}{b:02x}"


def contrast_text_color(hex_color: str) -> str:
    r, g, b = hex_to_rgb(hex_color)
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#111111" if lum > 0.58 else "#ffffff"


def color_preview_style(hex_color: str) -> dict[str, Any]:
    return {
        "height": "22px",
        "width": "100%",
        "border": "1px solid #777",
        "borderRadius": "4px",
        "backgroundColor": clean_hex_color(hex_color),
        "color": contrast_text_color(hex_color),
        "fontSize": "11px",
        "textAlign": "center",
        "lineHeight": "22px",
        "marginTop": "4px",
        "boxSizing": "border-box",
    }


def color_slider_marks(channel: str) -> dict[int, dict[str, Any]]:
    color = {"r": "#cc0000", "g": "#008000", "b": "#0033cc"}.get(channel, "#444")
    return {
        0: {"label": "0", "style": {"color": "#555", "fontSize": "10px"}},
        128: {"label": "128", "style": {"color": color, "fontSize": "10px"}},
        255: {"label": "255", "style": {"color": color, "fontSize": "10px"}},
    }


def make_rgb_picker(prefix: str, label: str, default_hex: str) -> html.Div:
    r, g, b = hex_to_rgb(default_hex)
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "700", "fontSize": "13px", "marginBottom": "4px"}),
            html.Div(id=f"{prefix}-preview", children=default_hex, style=color_preview_style(default_hex)),
            html.Div("Red", style={"fontSize": "11px", "marginTop": "5px"}),
            dcc.Slider(id=f"{prefix}-r", min=0, max=255, step=1, value=r, marks=color_slider_marks("r"), tooltip={"placement": "bottom"}),
            html.Div("Green", style={"fontSize": "11px", "marginTop": "2px"}),
            dcc.Slider(id=f"{prefix}-g", min=0, max=255, step=1, value=g, marks=color_slider_marks("g"), tooltip={"placement": "bottom"}),
            html.Div("Blue", style={"fontSize": "11px", "marginTop": "2px"}),
            dcc.Slider(id=f"{prefix}-b", min=0, max=255, step=1, value=b, marks=color_slider_marks("b"), tooltip={"placement": "bottom"}),
        ],
        style={"border": "1px solid #e2e2e2", "borderRadius": "6px", "padding": "8px", "marginBottom": "10px"},
    )


# -----------------------------
# Data loading
# -----------------------------


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {c.lower().strip().replace(" ", "_"): c for c in columns}
    for cand in candidates:
        key = cand.lower().strip().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def detect_units_from_values(coords: np.ndarray) -> str:
    finite = coords[np.isfinite(coords)]
    if finite.size == 0:
        return "mm"
    max_abs = float(np.nanmax(np.abs(finite)))
    return "m" if max_abs < 2.5 else "mm"


def normalize_units(coords: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    units = units.lower().strip()
    if units not in {"auto", "mm", "m", "meter", "meters"}:
        raise ValueError("Units must be one of: auto, mm, m")

    detected = detect_units_from_values(coords) if units == "auto" else units
    if detected in {"m", "meter", "meters"}:
        return coords * 1000.0, "m converted to mm"
    return coords, "mm"


def standard_empty_df(source: str = "empty") -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "label", "x", "y", "z", "region", "hemisphere", "confidence", "region_note", "source", "cap",
    ]).assign(source=source)


def finalize_electrode_df(df: pd.DataFrame, source: str, cap_name: str = "reference") -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].map(canonical_label)
    out = out.dropna(subset=["label", "x", "y", "z"])
    out = out[out["label"].astype(str).str.len() > 0]
    out = out.reset_index(drop=True)

    for col in ["x", "y", "z"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["x", "y", "z"]).reset_index(drop=True)

    if "region" not in out.columns:
        out["region"] = ""
    if "hemisphere" not in out.columns:
        out["hemisphere"] = ""
    if "confidence" not in out.columns:
        out["confidence"] = ""
    if "region_note" not in out.columns:
        out["region_note"] = ""

    inferred_rows: list[dict[str, str]] = []
    for _, row in out.iterrows():
        guess = infer_region(row["label"], row["x"], row["y"], row["z"])
        region = str(row.get("region", "")).strip() or guess.region
        hemisphere = str(row.get("hemisphere", "")).strip() or guess.hemisphere
        confidence = str(row.get("confidence", "")).strip() or guess.confidence
        note = str(row.get("region_note", "")).strip() or guess.note
        inferred_rows.append({
            "region": region,
            "hemisphere": hemisphere,
            "confidence": confidence,
            "region_note": note,
        })

    inferred = pd.DataFrame(inferred_rows)
    for col in inferred.columns:
        out[col] = inferred[col]

    out["source"] = source
    out["cap"] = cap_name
    out = out[["label", "x", "y", "z", "region", "hemisphere", "confidence", "region_note", "source", "cap"]]
    return out


def parse_csv_dataframe(raw_df: pd.DataFrame, units: str = "auto", source: str = "custom CSV", cap_name: str = "reference") -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The CSV is empty.")

    label_col = _find_column(raw_df.columns, ["label", "electrode", "channel", "ch", "name", "electrode_label"])
    x_col = _find_column(raw_df.columns, ["x", "mni_x", "x_mm", "ras_x", "coord_x"])
    y_col = _find_column(raw_df.columns, ["y", "mni_y", "y_mm", "ras_y", "coord_y"])
    z_col = _find_column(raw_df.columns, ["z", "mni_z", "z_mm", "ras_z", "coord_z"])

    missing = [name for name, col in [("label", label_col), ("x", x_col), ("y", y_col), ("z", z_col)] if col is None]
    if missing:
        raise ValueError(
            "Missing required CSV column(s): " + ", ".join(missing) +
            ". Use label,x,y,z or electrode,mni_x,mni_y,mni_z."
        )

    parsed = pd.DataFrame({
        "label": raw_df[label_col],
        "x": pd.to_numeric(raw_df[x_col], errors="coerce"),
        "y": pd.to_numeric(raw_df[y_col], errors="coerce"),
        "z": pd.to_numeric(raw_df[z_col], errors="coerce"),
    })

    optional_map = {
        "region": ["region", "brain_region", "area", "cortex", "description"],
        "hemisphere": ["hemisphere", "hemi"],
        "confidence": ["confidence", "mapping_confidence"],
        "region_note": ["region_note", "note", "notes", "comment"],
    }
    for out_col, candidates in optional_map.items():
        in_col = _find_column(raw_df.columns, candidates)
        if in_col is not None:
            parsed[out_col] = raw_df[in_col].fillna("").astype(str)

    coords = parsed[["x", "y", "z"]].to_numpy(dtype=float)
    coords_mm, units_note = normalize_units(coords, units)
    parsed[["x", "y", "z"]] = coords_mm
    parsed.attrs["units_note"] = units_note

    return finalize_electrode_df(parsed, source=source, cap_name=cap_name)


def load_custom_csv(path: str | Path, units: str = "auto", cap_name: str = "reference", source_prefix: str = "custom CSV") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    raw = pd.read_csv(path)
    return parse_csv_dataframe(raw, units=units, source=f"{source_prefix}: {path.name}", cap_name=cap_name)


def load_default_montage_from_mne(cap_name: str = "reference") -> pd.DataFrame:
    try:
        import mne  # type: ignore
    except Exception:
        return load_fallback_montage(cap_name=cap_name)

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]

    rows = []
    for label in TEN_TEN_LABELS:
        if label not in ch_pos:
            continue
        xyz_mm = np.asarray(ch_pos[label], dtype=float) * 1000.0
        rows.append({"label": label, "x": xyz_mm[0], "y": xyz_mm[1], "z": xyz_mm[2]})

    if not rows:
        return load_fallback_montage(cap_name=cap_name)
    return finalize_electrode_df(pd.DataFrame(rows), source="MNE standard_1005 10-10 subset", cap_name=cap_name)


def load_default_reference(cap_name: str = "reference") -> pd.DataFrame:
    """Use bundled default CSV when present; otherwise use MNE standard_1005."""
    local_csv = Path(__file__).resolve().with_name(DEFAULT_REFERENCE_CSV_NAME)
    if local_csv.exists():
        try:
            return load_custom_csv(local_csv, units="mm", cap_name=cap_name, source_prefix="default reference CSV")
        except Exception:
            pass
    return load_default_montage_from_mne(cap_name=cap_name)


def load_fallback_montage(cap_name: str = "reference") -> pd.DataFrame:
    rows = []
    row_definitions = {
        "Fp": (70, ["Fp1", "Fpz", "Fp2"]),
        "AF": (52, ["AF7", "AF3", "AFz", "AF4", "AF8"]),
        "F": (34, ["F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10"]),
        "FC": (16, ["FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10"]),
        "C": (0, ["T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10"]),
        "CP": (-18, ["TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10"]),
        "P": (-38, ["P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10"]),
        "PO": (-60, ["PO7", "PO3", "POz", "PO4", "PO8"]),
        "O": (-78, ["O1", "Oz", "O2"]),
        "Iz": (-92, ["Iz"]),
    }

    center = np.array([0.0, -5.0, 35.0])
    radii = np.array([85.0, 105.0, 95.0])
    z_by_row = {
        "Fp": 15, "AF": 38, "F": 60, "FC": 75, "C": 82,
        "CP": 76, "P": 62, "PO": 42, "O": 20, "Iz": 0,
    }

    for row_name, (y_deg, labels) in row_definitions.items():
        if len(labels) == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-0.88, 0.88, len(labels)) * radii[0]
        y = center[1] + math.sin(math.radians(y_deg)) * radii[1]
        z = z_by_row[row_name]
        for label, x in zip(labels, xs):
            rows.append({"label": label, "x": float(x), "y": float(y), "z": float(z)})

    return finalize_electrode_df(
        pd.DataFrame(rows),
        source="fallback approximate 10-10 montage; install MNE for standard_1005",
        cap_name=cap_name,
    )


def parse_uploaded_csv(contents: str, filename: str | None, units: str, cap_name: str) -> pd.DataFrame:
    if not contents or "," not in contents:
        raise ValueError("Upload contents were not recognized.")
    _header, encoded = contents.split(",", 1)
    decoded = base64.b64decode(encoded)
    text = decoded.decode("utf-8-sig")
    raw = pd.read_csv(io.StringIO(text))
    source = f"uploaded CSV: {filename or 'unnamed'}"
    return parse_csv_dataframe(raw, units=units, source=source, cap_name=cap_name)


# -----------------------------
# Data serialization and comparison
# -----------------------------


def records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return EMPTY_RECORDS.copy()
    clean = df.copy()
    for col in ["x", "y", "z"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace({np.nan: ""})
    return clean.to_dict(orient="records")


def df_from_records(records: list[dict[str, Any]] | None, fallback_cap: str) -> pd.DataFrame:
    if not records:
        return standard_empty_df(source="empty")
    df = pd.DataFrame(records)
    for col in ["label", "x", "y", "z", "region", "hemisphere", "confidence", "region_note", "source", "cap"]:
        if col not in df.columns:
            df[col] = "" if col not in ["x", "y", "z"] else np.nan
    df["cap"] = df["cap"].replace("", fallback_cap)
    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)
    return df


def add_comparison_metrics(reference_df: pd.DataFrame, comparison_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ref = reference_df.copy()
    cmp = comparison_df.copy()

    for df in [ref, cmp]:
        df["matched_label"] = False
        df["delta_x_mm"] = np.nan
        df["delta_y_mm"] = np.nan
        df["delta_z_mm"] = np.nan
        df["distance_to_reference_mm"] = np.nan

    if ref.empty or cmp.empty:
        return ref, cmp, {
            "n_reference": int(len(ref)),
            "n_comparison": int(len(cmp)),
            "n_matched": 0,
            "mean_distance": np.nan,
            "median_distance": np.nan,
            "max_distance": np.nan,
            "max_label": "",
        }

    ref_first = ref.drop_duplicates(subset=["label"]).set_index("label")
    cmp_first = cmp.drop_duplicates(subset=["label"]).set_index("label")
    matched_labels = sorted(set(ref_first.index).intersection(set(cmp_first.index)))

    distances: list[float] = []
    max_label = ""
    max_distance = -np.inf

    for label in matched_labels:
        ref_xyz = ref_first.loc[label, ["x", "y", "z"]].to_numpy(dtype=float)
        cmp_xyz = cmp_first.loc[label, ["x", "y", "z"]].to_numpy(dtype=float)
        delta = cmp_xyz - ref_xyz
        distance = float(np.linalg.norm(delta))
        distances.append(distance)
        if distance > max_distance:
            max_distance = distance
            max_label = label

        ref_mask = ref["label"] == label
        cmp_mask = cmp["label"] == label
        for target, mask in [(ref, ref_mask), (cmp, cmp_mask)]:
            target.loc[mask, "matched_label"] = True
            target.loc[mask, "delta_x_mm"] = delta[0]
            target.loc[mask, "delta_y_mm"] = delta[1]
            target.loc[mask, "delta_z_mm"] = delta[2]
            target.loc[mask, "distance_to_reference_mm"] = distance

    distances_arr = np.asarray(distances, dtype=float)
    summary = {
        "n_reference": int(len(ref)),
        "n_comparison": int(len(cmp)),
        "n_matched": int(len(matched_labels)),
        "mean_distance": float(np.nanmean(distances_arr)) if distances_arr.size else np.nan,
        "median_distance": float(np.nanmedian(distances_arr)) if distances_arr.size else np.nan,
        "max_distance": float(np.nanmax(distances_arr)) if distances_arr.size else np.nan,
        "max_label": max_label if distances_arr.size else "",
    }
    return ref, cmp, summary


def format_distance(value: Any) -> str:
    try:
        f = float(value)
        if np.isfinite(f):
            return f"{f:.2f} mm"
    except Exception:
        pass
    return "not matched"


# -----------------------------
# Head model and projection
# -----------------------------


def combined_coordinate_df(reference_df: pd.DataFrame, comparison_df: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [reference_df, comparison_df] if df is not None and not df.empty]
    if not frames:
        return load_default_reference(cap_name="reference")
    return pd.concat(frames, ignore_index=True)


def fit_head_to_coordinates(df: pd.DataFrame, head_scale: float = 1.0) -> HeadFit:
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.size == 0:
        return HeadFit(center=np.array([0.0, 0.0, 35.0]), radii=np.array([85.0, 105.0, 95.0]))

    if len(coords) >= 8:
        low = np.nanpercentile(coords, 2, axis=0)
        high = np.nanpercentile(coords, 98, axis=0)
    else:
        low = np.nanmin(coords, axis=0)
        high = np.nanmax(coords, axis=0)

    center = (low + high) / 2.0
    half_span = np.maximum((high - low) / 2.0, np.array([20.0, 20.0, 20.0]))

    rx = max(float(half_span[0]) * 1.18, 65.0)
    ry = max(float(half_span[1]) * 1.18, 75.0)
    rz = max(float(half_span[2]) * 1.35, 75.0)

    # EEG electrode clouds cover mostly the upper head. Shift the fitted head
    # inferiorly, and scale the head mesh itself to put the cap on the outer shell.
    center[2] = center[2] - 0.10 * rz
    radii = np.array([rx, ry, rz], dtype=float) * float(head_scale)
    return HeadFit(center=center.astype(float), radii=radii)


def make_phantom_surface(fit: HeadFit, detail: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Keep the lower bound small by default; dense surfaces make 3D point
    # picking less reliable in browsers.
    res = int(np.clip(detail, 6, 80))
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2 * np.pi, res)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = fit.center[0] + fit.radii[0] * np.sin(theta_grid) * np.cos(phi_grid)
    y = fit.center[1] + fit.radii[1] * np.sin(theta_grid) * np.sin(phi_grid)
    z = fit.center[2] + fit.radii[2] * np.cos(theta_grid)
    return x, y, z


def structured_grid_to_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n0, n1 = x.shape
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = []
    for i in range(n0 - 1):
        for j in range(n1 - 1):
            a = i * n1 + j
            b = a + 1
            c = (i + 1) * n1 + j
            d = c + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices, np.asarray(faces, dtype=int)


def make_procedural_realistic_mesh(fit: HeadFit, detail: int) -> MeshData:
    """Fallback anatomical-looking mesh if fsaverage is not available."""
    res = int(np.clip(detail, 10, 70))
    theta = np.linspace(0.03, np.pi - 0.03, res)
    phi = np.linspace(0, 2 * np.pi, res)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    sx = np.sin(theta_grid) * np.cos(phi_grid)
    sy = np.sin(theta_grid) * np.sin(phi_grid)
    sz = np.cos(theta_grid)

    # Mild anatomical deformations: face/nose region, flattened lower posterior,
    # and a narrower lower head. It is only a fallback visualization mesh.
    anterior = np.clip(sy, 0, 1)
    superior = np.clip(sz, 0, 1)
    inferior = np.clip(-sz, 0, 1)
    nose_bump = np.exp(-((sx / 0.22) ** 2 + ((sy - 0.95) / 0.18) ** 2 + ((sz - 0.10) / 0.35) ** 2))
    lower_taper = 1.0 - 0.16 * inferior
    face_forward = 1.0 + 0.12 * anterior * (1.0 - 0.5 * superior)

    x = fit.center[0] + fit.radii[0] * sx * lower_taper
    y = fit.center[1] + fit.radii[1] * sy * face_forward + 0.20 * fit.radii[1] * nose_bump
    z = fit.center[2] + fit.radii[2] * sz * (1.0 - 0.07 * inferior)

    vertices, faces = structured_grid_to_mesh(x, y, z)
    return MeshData(vertices=vertices, faces=faces, source="procedural fallback mesh; fsaverage unavailable", is_public_template=False)


def surface_file_candidates(fs_dir: Path) -> list[Path]:
    return [
        fs_dir / "bem" / "outer_skin.surf",
        fs_dir / "bem" / "outer_skin.surf.fif",
        fs_dir / "fsaverage" / "bem" / "outer_skin.surf",
        fs_dir / "subjects" / "fsaverage" / "bem" / "outer_skin.surf",
    ]


@lru_cache(maxsize=2)
def load_fsaverage_outer_skin_mesh(allow_download: bool = True) -> MeshData:
    """Load MNE fsaverage outer_skin.surf.

    MNE fetch_fsaverage provides a public template subject with head surfaces.
    This function returns a public template mesh when available. If it fails,
    it raises a RuntimeError and the caller can use a fallback mesh.
    """
    try:
        import mne  # type: ignore
    except Exception as exc:
        raise RuntimeError("MNE is not installed; cannot load fsaverage outer_skin.surf") from exc

    candidates: list[Path] = []
    env_subjects = os.environ.get("SUBJECTS_DIR")
    if env_subjects:
        candidates.append(Path(env_subjects).expanduser() / "fsaverage" / "bem" / "outer_skin.surf")

    fs_dir: Path | None = None
    fetch_error: Exception | None = None
    if allow_download:
        try:
            fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
        except Exception as exc:  # network, permission, or MNE-data errors
            fetch_error = exc
    else:
        try:
            subjects_dir = mne.get_config("SUBJECTS_DIR")
            if subjects_dir:
                fs_dir = Path(subjects_dir).expanduser() / "fsaverage"
        except Exception:
            fs_dir = None

    if fs_dir is not None:
        candidates.extend(surface_file_candidates(fs_dir))

    # Also try common MNE data path used by fetch_fsaverage.
    try:
        mne_data = Path(mne.get_config("MNE_DATA") or "").expanduser()
        if str(mne_data):
            candidates.append(mne_data / "MNE-fsaverage-data" / "fsaverage" / "bem" / "outer_skin.surf")
    except Exception:
        pass

    seen: set[Path] = set()
    unique_candidates = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            unique_candidates.append(p)

    read_errors: list[str] = []
    for path in unique_candidates:
        if not path.exists() or path.suffix == ".fif":
            continue
        try:
            rr, tris = mne.read_surface(str(path))
            vertices = np.asarray(rr, dtype=float)
            faces = np.asarray(tris, dtype=int)
            # FreeSurfer/MNE surface files are normally in mm. Convert only if
            # the values look like meters.
            if np.nanmax(np.abs(vertices)) < 2.5:
                vertices = vertices * 1000.0
            return MeshData(vertices=vertices, faces=faces, source=str(path), is_public_template=True)
        except Exception as exc:
            read_errors.append(f"{path}: {exc}")

    msg = "Could not find/read fsaverage outer_skin.surf."
    if fetch_error is not None:
        msg += f" Fetch error: {fetch_error}"
    if read_errors:
        msg += " Read errors: " + " | ".join(read_errors[:2])
    raise RuntimeError(msg)


def robust_center_and_radii(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.nanpercentile(vertices, 2, axis=0)
    high = np.nanpercentile(vertices, 98, axis=0)
    center = (low + high) / 2.0
    radii = np.maximum((high - low) / 2.0, np.array([1.0, 1.0, 1.0]))
    return center, radii


def fit_template_mesh_to_cap(mesh: MeshData, fit: HeadFit) -> np.ndarray:
    """Scale and center the head mesh to the electrode cloud.

    This changes the visual head shape/size to fit the cap rather than changing
    original electrode coordinates. Display coordinates are still projected for
    selection, but original MNI/CSV values remain unchanged.
    """
    native_center, native_radii = robust_center_and_radii(mesh.vertices)
    normalized = (mesh.vertices - native_center.reshape(1, 3)) / native_radii.reshape(1, 3)
    fitted = fit.center.reshape(1, 3) + normalized * fit.radii.reshape(1, 3)
    return fitted.astype(float)


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) <= target_faces:
        return vertices, faces
    target = max(int(target_faces), 100)
    idx = np.linspace(0, len(faces) - 1, target, dtype=int)
    sub_faces = faces[idx]
    unique_vertices, inverse = np.unique(sub_faces.ravel(), return_inverse=True)
    remapped_faces = inverse.reshape(-1, 3)
    return vertices[unique_vertices], remapped_faces.astype(int)


def real_mesh_target_faces(detail: int) -> int:
    detail = int(np.clip(detail, 6, 80))
    # Very sparse triangular meshes can appear invisible or fragmented in Plotly,
    # especially at low opacity. Keep the minimum dense enough to show a head.
    return int(np.interp(detail, [6, 80], [4500, 36000]))


def build_head_geometry(head_model: str, fit: HeadFit, detail: int, allow_download: bool) -> HeadGeometry:
    if head_model == "realistic":
        try:
            native = load_fsaverage_outer_skin_mesh(bool(allow_download))
            fitted_vertices = fit_template_mesh_to_cap(native, fit)
            v, f = decimate_mesh(fitted_vertices, native.faces, real_mesh_target_faces(detail))
            status = "realistic head: MNE fsaverage outer_skin.surf"
            return HeadGeometry(
                kind="realistic",
                fit=fit,
                vertices=v,
                faces=f,
                projection_vertices=fitted_vertices,
                status=status,
            )
        except Exception as exc:
            fallback = make_procedural_realistic_mesh(fit, detail=max(detail, 18))
            status = "realistic fallback: fsaverage unavailable; " + str(exc)[:160]
            return HeadGeometry(
                kind="procedural-realistic",
                fit=fit,
                vertices=fallback.vertices,
                faces=fallback.faces,
                projection_vertices=fallback.vertices,
                status=status,
            )

    sx, sy, sz = make_phantom_surface(fit, detail)
    return HeadGeometry(kind="phantom", fit=fit, surface_x=sx, surface_y=sy, surface_z=sz, status="adaptive phantom shell")


def project_points_to_ellipsoid(df: pd.DataFrame, fit: HeadFit, offset_mm: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in ["display_x", "display_y", "display_z"]:
            out[col] = []
        return out

    coords = out[["x", "y", "z"]].to_numpy(dtype=float)
    center = fit.center.reshape(1, 3)
    radii = fit.radii.reshape(1, 3)
    vectors = coords - center
    denom = np.sqrt(np.sum((vectors / radii) ** 2, axis=1))
    denom = np.where(np.isfinite(denom) & (denom > 1e-9), denom, 1.0)
    mean_radius = float(np.mean(fit.radii))
    shell_factor = 1.0 + max(float(offset_mm), 0.0) / max(mean_radius, 1.0)
    display = center + vectors * (shell_factor / denom).reshape(-1, 1)

    out["display_x"] = display[:, 0]
    out["display_y"] = display[:, 1]
    out["display_z"] = display[:, 2]
    out["projection_offset_mm"] = float(offset_mm)
    return out


def project_points_to_mesh(df: pd.DataFrame, vertices: np.ndarray, center: np.ndarray, offset_mm: float) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in ["display_x", "display_y", "display_z"]:
            out[col] = []
        return out

    coords = out[["x", "y", "z"]].to_numpy(dtype=float)
    rel_vertices = vertices - center.reshape(1, 3)
    vertex_radius = np.linalg.norm(rel_vertices, axis=1)
    valid = np.isfinite(vertex_radius) & (vertex_radius > 1e-6)
    rel_vertices = rel_vertices[valid]
    vertex_radius = vertex_radius[valid]

    if len(vertex_radius) == 0:
        return project_points_to_ellipsoid(out, HeadFit(center=center, radii=np.array([80.0, 95.0, 85.0])), offset_mm)

    vertex_dirs = rel_vertices / vertex_radius.reshape(-1, 1)
    point_vecs = coords - center.reshape(1, 3)
    point_norm = np.linalg.norm(point_vecs, axis=1)
    point_norm = np.where(np.isfinite(point_norm) & (point_norm > 1e-6), point_norm, 1.0)
    point_dirs = point_vecs / point_norm.reshape(-1, 1)

    display = np.zeros_like(coords, dtype=float)
    for i, direction in enumerate(point_dirs):
        if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
        # Approximate radial surface radius using the closest vertex direction.
        cosines = vertex_dirs @ direction
        idx = int(np.nanargmax(cosines))
        radius = float(vertex_radius[idx])
        display[i, :] = center + direction * (radius + max(float(offset_mm), 0.0))

    out["display_x"] = display[:, 0]
    out["display_y"] = display[:, 1]
    out["display_z"] = display[:, 2]
    out["projection_offset_mm"] = float(offset_mm)
    return out


def project_points_to_head(df: pd.DataFrame, geometry: HeadGeometry, offset_mm: float) -> pd.DataFrame:
    projection_vertices = geometry.projection_vertices if geometry.projection_vertices is not None else geometry.vertices
    if projection_vertices is not None:
        return project_points_to_mesh(df, projection_vertices, geometry.fit.center, offset_mm)
    return project_points_to_ellipsoid(df, geometry.fit, offset_mm)


def make_nose_and_ears(fit: HeadFit) -> list[go.Scatter3d]:
    cx, cy, cz = fit.center
    rx, ry, rz = fit.radii

    nose_x = [cx - 0.09 * rx, cx, cx + 0.09 * rx, cx]
    nose_y = [cy + 0.91 * ry, cy + 1.18 * ry, cy + 0.91 * ry, cy + 0.91 * ry]
    nose_z = [cz + 0.18 * rz, cz + 0.24 * rz, cz + 0.18 * rz, cz + 0.18 * rz]

    ear_t = np.linspace(0, 2 * np.pi, 40)
    left_ear_x = cx - 1.03 * rx + 0.06 * rx * np.cos(ear_t)
    left_ear_y = cy + 0.00 * ry + 0.02 * ry * np.sin(ear_t)
    left_ear_z = cz + 0.03 * rz + 0.18 * rz * np.sin(ear_t)

    right_ear_x = cx + 1.03 * rx + 0.06 * rx * np.cos(ear_t)
    right_ear_y = cy + 0.00 * ry + 0.02 * ry * np.sin(ear_t)
    right_ear_z = cz + 0.03 * rz + 0.18 * rz * np.sin(ear_t)

    common_line = dict(color="#333333", width=4)
    return [
        go.Scatter3d(x=nose_x, y=nose_y, z=nose_z, mode="lines", line=common_line, hoverinfo="skip", showlegend=False, name="nose"),
        go.Scatter3d(x=left_ear_x, y=left_ear_y, z=left_ear_z, mode="lines", line=common_line, hoverinfo="skip", showlegend=False, name="left ear"),
        go.Scatter3d(x=right_ear_x, y=right_ear_y, z=right_ear_z, mode="lines", line=common_line, hoverinfo="skip", showlegend=False, name="right ear"),
    ]


# -----------------------------
# Plotting
# -----------------------------


def make_surface_wireframe_trace(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    line_color: str,
    opacity: float = 0.46,
    max_lines: int = 14,
) -> go.Scatter3d:
    """Create sparse latitude/longitude lines so transparent scalp surfaces remain visible."""
    n0, n1 = x.shape
    row_stride = max(1, int(math.ceil(n0 / max_lines)))
    col_stride = max(1, int(math.ceil(n1 / max_lines)))

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for i in range(0, n0, row_stride):
        xs.extend([float(v) for v in x[i, :]] + [None])
        ys.extend([float(v) for v in y[i, :]] + [None])
        zs.extend([float(v) for v in z[i, :]] + [None])
    for j in range(0, n1, col_stride):
        xs.extend([float(v) for v in x[:, j]] + [None])
        ys.extend([float(v) for v in y[:, j]] + [None])
        zs.extend([float(v) for v in z[:, j]] + [None])

    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=hex_to_rgba(line_color, opacity), width=1.5),
        hoverinfo="skip",
        showlegend=False,
        name="scalp outline",
    )


def add_visible_head_traces(
    fig: go.Figure,
    geometry: HeadGeometry,
    head_color: str,
    head_opacity: float,
    mesh_detail: int,
    show_mesh_edges: bool = False,
) -> None:
    """Add a visibly rendered scalp/head layer.

    Earlier versions used a very low default opacity and a sparse triangular
    mesh. In some browsers that made the scalp look absent. This function uses
    stronger lighting, enough triangles for the realistic mesh, and a surface
    trace for the phantom shell. Hover is skipped so electrode selection remains
    attached to electrode traces.
    """
    opacity = float(np.clip(head_opacity, 0.02, 1.0))
    color = clean_hex_color(head_color)

    if geometry.surface_x is not None and geometry.surface_y is not None and geometry.surface_z is not None:
        fig.add_trace(
            go.Surface(
                x=geometry.surface_x,
                y=geometry.surface_y,
                z=geometry.surface_z,
                surfacecolor=np.ones_like(geometry.surface_x, dtype=float),
                colorscale=[[0, color], [1, color]],
                cmin=0,
                cmax=1,
                showscale=False,
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False,
                name="visible scalp shell",
                lighting=dict(ambient=0.58, diffuse=0.72, roughness=0.55, specular=0.12),
                lightposition=dict(x=0, y=120, z=220),
            )
        )
        if show_mesh_edges:
            fig.add_trace(make_surface_wireframe_trace(
                geometry.surface_x, geometry.surface_y, geometry.surface_z,
                blend_hex_with(color, "#000000", 0.52),
                opacity=0.46,
            ))

    if geometry.vertices is not None and geometry.faces is not None and len(geometry.faces) > 0:
        fig.add_trace(
            go.Mesh3d(
                x=geometry.vertices[:, 0],
                y=geometry.vertices[:, 1],
                z=geometry.vertices[:, 2],
                i=geometry.faces[:, 0],
                j=geometry.faces[:, 1],
                k=geometry.faces[:, 2],
                color=color,
                opacity=opacity,
                hoverinfo="skip",
                flatshading=False,
                lighting=dict(ambient=0.60, diffuse=0.75, roughness=0.50, specular=0.16, fresnel=0.05),
                lightposition=dict(x=0, y=140, z=260),
                name="visible head mesh",
                showlegend=False,
            )
        )
        if show_mesh_edges:
            # Optional diagnostic wireframe. Off by default because it adds traces.
            edge_x: list[float | None] = []
            edge_y: list[float | None] = []
            edge_z: list[float | None] = []
            face_sample = geometry.faces[:: max(1, len(geometry.faces) // 1500)]
            for tri in face_sample:
                pts = geometry.vertices[tri]
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    edge_x.extend([float(pts[a, 0]), float(pts[b, 0]), None])
                    edge_y.extend([float(pts[a, 1]), float(pts[b, 1]), None])
                    edge_z.extend([float(pts[a, 2]), float(pts[b, 2]), None])
            fig.add_trace(
                go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode="lines",
                    line=dict(color="rgba(60,60,60,0.22)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                    name="head mesh edges",
                )
            )


def make_customdata(df: pd.DataFrame, cap_display_name: str) -> np.ndarray:
    cols = [
        "label", "x", "y", "z", "display_x", "display_y", "display_z",
        "region", "hemisphere", "confidence", "region_note", "source", "cap",
        "matched_label", "delta_x_mm", "delta_y_mm", "delta_z_mm", "distance_to_reference_mm",
    ]
    data = []
    for _, row in df.iterrows():
        row_vals = []
        for col in cols:
            row_vals.append(row.get(col, ""))
        row_vals.append(cap_display_name)
        data.append(row_vals)
    return np.asarray(data, dtype=object)


def cap_hovertemplate() -> str:
    return (
        "<b>%{customdata[18]}: %{customdata[0]}</b><br>"
        "Region: %{customdata[7]}<br>"
        "Hemisphere: %{customdata[8]}<br>"
        "Original MNI/mm: "
        "x=%{customdata[1]:.1f}, y=%{customdata[2]:.1f}, z=%{customdata[3]:.1f}<br>"
        "Projected display: "
        "x=%{customdata[4]:.1f}, y=%{customdata[5]:.1f}, z=%{customdata[6]:.1f}<br>"
        "Distance between caps: %{customdata[17]:.2f} mm<br>"
        "<extra></extra>"
    )


def add_cap_traces(
    fig: go.Figure,
    plot_df: pd.DataFrame,
    cap_name: str,
    cap_display_name: str,
    electrode_color: str,
    line_color: str,
    opacity: float,
    electrode_size: int,
    label_size: int,
    label_color: str,
    hitbox_extra_size: int,
    labels_visible: bool,
) -> None:
    if plot_df.empty:
        return

    customdata = make_customdata(plot_df, cap_display_name)
    text_values = plot_df["label"] if labels_visible else [""] * len(plot_df)
    hovertemplate = cap_hovertemplate()

    # Larger translucent hitbox. This is intentionally drawn after the head mesh.
    fig.add_trace(
        go.Scatter3d(
            x=plot_df["display_x"],
            y=plot_df["display_y"],
            z=plot_df["display_z"],
            mode="markers",
            marker=dict(
                size=max(int(electrode_size) + int(hitbox_extra_size), 4),
                color=hex_to_rgba(electrode_color, max(0.035, min(float(opacity), 1.0) * 0.08)),
                line=dict(width=0),
                opacity=max(0.035, min(float(opacity), 1.0) * 0.08),
            ),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=False,
            name=f"{cap_display_name} hitbox",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=plot_df["display_x"],
            y=plot_df["display_y"],
            z=plot_df["display_z"],
            mode="markers+text" if labels_visible else "markers",
            marker=dict(
                size=int(electrode_size),
                color=hex_to_rgba(electrode_color, float(opacity)),
                line=dict(width=1.5, color=line_color),
                opacity=float(opacity),
            ),
            text=text_values,
            textposition="top center",
            textfont=dict(size=int(label_size), color=label_color),
            customdata=customdata,
            hovertemplate=hovertemplate,
            showlegend=True,
            name=cap_display_name,
        )
    )


def add_difference_lines(fig: go.Figure, ref_df: pd.DataFrame, cmp_df: pd.DataFrame, line_color: str) -> None:
    if ref_df.empty or cmp_df.empty:
        return
    ref_first = ref_df.drop_duplicates(subset=["label"]).set_index("label")
    cmp_first = cmp_df.drop_duplicates(subset=["label"]).set_index("label")
    matched = sorted(set(ref_first.index).intersection(set(cmp_first.index)))
    if not matched:
        return

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for lab in matched:
        r = ref_first.loc[lab]
        c = cmp_first.loc[lab]
        xs.extend([float(r["display_x"]), float(c["display_x"]), None])
        ys.extend([float(r["display_y"]), float(c["display_y"]), None])
        zs.extend([float(r["display_z"]), float(c["display_z"]), None])

    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=line_color, width=3),
            hoverinfo="skip",
            showlegend=True,
            name="matched-electrode displacement",
        )
    )


def camera_for_view(view_name: str | None) -> dict[str, Any]:
    view = (view_name or "default").lower()
    if view == "left":
        return dict(eye=dict(x=-2.4, y=0.0, z=0.15), up=dict(x=0, y=0, z=1))
    if view == "right":
        return dict(eye=dict(x=2.4, y=0.0, z=0.15), up=dict(x=0, y=0, z=1))
    if view == "top":
        return dict(eye=dict(x=0.0, y=0.0, z=2.6), up=dict(x=0, y=1, z=0))
    if view == "front":
        return dict(eye=dict(x=0.0, y=2.5, z=0.15), up=dict(x=0, y=0, z=1))
    if view == "back":
        return dict(eye=dict(x=0.0, y=-2.5, z=0.15), up=dict(x=0, y=0, z=1))
    return dict(eye=dict(x=1.55, y=-1.85, z=1.25), up=dict(x=0, y=0, z=1))


def plotly_view_buttons() -> list[dict[str, Any]]:
    def button(label: str, view: str) -> dict[str, Any]:
        return dict(
            label=label,
            method="relayout",
            args=[{"scene.camera": camera_for_view(view)}],
        )

    return [
        button("Default", "default"),
        button("Left", "left"),
        button("Right", "right"),
        button("Top", "top"),
        button("Front", "front"),
        button("Back", "back"),
    ]


def all_plot_coordinates(geometry: HeadGeometry, ref_df: pd.DataFrame, cmp_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    if geometry.surface_x is not None:
        xs.append(geometry.surface_x.ravel())
        ys.append(geometry.surface_y.ravel())
        zs.append(geometry.surface_z.ravel())
    if geometry.vertices is not None:
        xs.append(geometry.vertices[:, 0])
        ys.append(geometry.vertices[:, 1])
        zs.append(geometry.vertices[:, 2])
    for df in [ref_df, cmp_df]:
        if df is not None and not df.empty and "display_x" in df.columns:
            xs.append(df["display_x"].to_numpy(dtype=float))
            ys.append(df["display_y"].to_numpy(dtype=float))
            zs.append(df["display_z"].to_numpy(dtype=float))
    if not xs:
        return np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def make_figure(
    reference_records: list[dict[str, Any]],
    comparison_records: list[dict[str, Any]],
    electrode_size: int,
    label_size: int,
    reference_color: str,
    comparison_color: str,
    label_color: str,
    head_color: str,
    background_color: str,
    reference_opacity: float,
    comparison_opacity: float,
    head_opacity: float,
    mesh_detail: int,
    projection_offset_mm: float,
    head_scale: float,
    hitbox_extra_size: int,
    show_options: list[str] | None,
    head_model: str,
    allow_fsaverage_download: bool,
    view_name: str,
) -> go.Figure:
    ref_df = df_from_records(reference_records, fallback_cap="reference")
    cmp_df = df_from_records(comparison_records, fallback_cap="comparison")

    if ref_df.empty:
        ref_df = load_default_reference(cap_name="reference")

    ref_df, cmp_df, summary = add_comparison_metrics(ref_df, cmp_df)
    fit = fit_head_to_coordinates(combined_coordinate_df(ref_df, cmp_df), head_scale=head_scale)
    geometry = build_head_geometry(head_model=head_model, fit=fit, detail=mesh_detail, allow_download=allow_fsaverage_download)

    ref_plot = project_points_to_head(ref_df, geometry=geometry, offset_mm=projection_offset_mm)
    cmp_plot = project_points_to_head(cmp_df, geometry=geometry, offset_mm=projection_offset_mm)

    visible_options = show_options or []
    labels_visible = "labels" in visible_options
    show_lines = "difference-lines" in visible_options
    show_landmarks = "landmarks" in visible_options
    show_head = "head" in visible_options or not visible_options
    show_mesh_edges = "mesh-edges" in visible_options

    fig = go.Figure()

    if show_head:
        add_visible_head_traces(
            fig=fig,
            geometry=geometry,
            head_color=head_color,
            head_opacity=head_opacity,
            mesh_detail=mesh_detail,
            show_mesh_edges=show_mesh_edges,
        )

    if show_landmarks or geometry.kind == "phantom":
        for trace in make_nose_and_ears(fit):
            fig.add_trace(trace)

    if show_lines:
        add_difference_lines(fig, ref_plot, cmp_plot, DEFAULT_DIFFERENCE_LINE_COLOR)

    add_cap_traces(
        fig=fig,
        plot_df=ref_plot,
        cap_name="reference",
        cap_display_name="Reference cap",
        electrode_color=reference_color,
        line_color="#111111",
        opacity=reference_opacity,
        electrode_size=electrode_size,
        label_size=label_size,
        label_color=label_color,
        hitbox_extra_size=hitbox_extra_size,
        labels_visible=labels_visible,
    )
    add_cap_traces(
        fig=fig,
        plot_df=cmp_plot,
        cap_name="comparison",
        cap_display_name="Comparison cap",
        electrode_color=comparison_color,
        line_color="#111111",
        opacity=comparison_opacity,
        electrode_size=electrode_size,
        label_size=label_size,
        label_color=label_color,
        hitbox_extra_size=hitbox_extra_size,
        labels_visible=labels_visible,
    )

    all_x, all_y, all_z = all_plot_coordinates(geometry, ref_plot, cmp_plot)
    pad = max(20.0, float(np.mean(fit.radii)) * 0.12)

    if summary["n_matched"]:
        distance_text = (
            f"matched={summary['n_matched']} | "
            f"mean={summary['mean_distance']:.2f} mm | "
            f"max={summary['max_distance']:.2f} mm at {summary['max_label']}"
        )
    else:
        distance_text = "no matched labels loaded for cap comparison"

    title = (
        f"{APP_TITLE} v{APP_VERSION} | "
        f"reference={summary['n_reference']} | comparison={summary['n_comparison']} | "
        f"{distance_text}<br><sup>Head model: {geometry.status}. Scalp/head surface is rendered as a visible layer; electrodes are projected outward for visualization only.</sup>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        margin=dict(l=0, r=0, t=68, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.55)"),
        scene=dict(
            xaxis=dict(title="MNI x / right-positive (mm)", range=[float(np.nanmin(all_x) - pad), float(np.nanmax(all_x) + pad)]),
            yaxis=dict(title="MNI y / anterior-positive (mm)", range=[float(np.nanmin(all_y) - pad), float(np.nanmax(all_y) + pad)]),
            zaxis=dict(title="MNI z / superior-positive (mm)", range=[float(np.nanmin(all_z) - pad), float(np.nanmax(all_z) + pad)]),
            aspectmode="data",
            camera=camera_for_view(view_name),
            dragmode="orbit",
        ),
        hovermode="closest",
        uirevision=f"{view_name}-{head_model}",
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=False,
                x=0.5,
                xanchor="center",
                y=1.03,
                yanchor="top",
                buttons=plotly_view_buttons(),
                pad={"r": 4, "t": 2},
            )
        ],
    )

    return fig


# -----------------------------
# Dash UI helpers
# -----------------------------


def format_status(df: pd.DataFrame, label: str, status_prefix: str = "Loaded") -> html.Div:
    if df is None or df.empty:
        return html.Div(
            [html.Strong(f"{label}: "), html.Span("not loaded.")],
            style={"fontSize": "13px", "color": "#555"},
        )
    source = str(df["source"].iloc[0]) if "source" in df.columns and len(df) else "unknown"
    return html.Div(
        [
            html.Strong(f"{label}: "),
            html.Span(f"{status_prefix}; {len(df)} electrodes from {source}."),
        ],
        style={"fontSize": "13px", "color": "#333"},
    )


def make_slider(id_: str, label: str, min_: float, max_: float, step: float, value: float, marks: dict | None = None) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=id_, style={"fontWeight": "600", "fontSize": "13px"}),
            dcc.Slider(
                id=id_,
                min=min_,
                max=max_,
                step=step,
                value=value,
                marks=marks or None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        style={"marginBottom": "14px"},
    )


def get_triggered_id() -> str | None:
    try:
        triggered = dash.ctx.triggered_id
        return triggered
    except Exception:
        try:
            ctx = dash.callback_context
            if not ctx.triggered:
                return None
            prop_id = ctx.triggered[0].get("prop_id", "")
            return prop_id.split(".")[0] if prop_id else None
        except Exception:
            return None


def button_style(width: str = "31%") -> dict[str, Any]:
    return {
        "width": width,
        "margin": "2px",
        "padding": "6px 4px",
        "fontSize": "12px",
        "border": "1px solid #aaa",
        "borderRadius": "4px",
        "backgroundColor": "#f7f7f7",
        "cursor": "pointer",
    }


def build_app(initial_reference_df: pd.DataFrame, initial_comparison_df: pd.DataFrame, reference_status: str, comparison_status: str) -> Dash:
    app = Dash(__name__)
    app.title = APP_TITLE

    app.layout = html.Div(
        [
            dcc.Store(id="reference-store", data=records_from_df(initial_reference_df)),
            dcc.Store(id="comparison-store", data=records_from_df(initial_comparison_df)),
            dcc.Store(id="view-store", data="default"),
            html.Div(
                [
                    html.H2(APP_TITLE, style={"margin": "0 0 4px 0"}),
                    html.Div(
                        f"Version {APP_VERSION}. Quick review of reference and comparison EEG cap locations.",
                        style={"fontSize": "13px", "color": "#555", "marginBottom": "8px"},
                    ),
                    html.Div(id="reference-status", children=format_status(initial_reference_df, "Reference", reference_status)),
                    html.Div(id="comparison-status", children=format_status(initial_comparison_df, "Comparison", comparison_status)),
                ],
                style={"padding": "12px 16px", "borderBottom": "1px solid #ddd"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Data", style={"marginTop": 0}),
                            html.Div("Reference cap", style={"fontWeight": "700", "fontSize": "13px", "marginTop": "4px"}),
                            dcc.Upload(
                                id="upload-reference-csv",
                                children=html.Div(["Upload reference CSV"]),
                                multiple=False,
                                style={
                                    "border": "1px dashed #888", "borderRadius": "6px", "padding": "10px",
                                    "textAlign": "center", "fontSize": "13px", "marginBottom": "8px",
                                },
                            ),
                            html.Button("Reload default reference", id="reload-reference-default", n_clicks=0, style={"width": "100%", "marginBottom": "8px"}),
                            html.Label("Reference CSV units", style={"fontWeight": "600", "fontSize": "12px"}),
                            dcc.RadioItems(
                                id="reference-csv-units",
                                options=[{"label": "Auto", "value": "auto"}, {"label": "mm", "value": "mm"}, {"label": "m", "value": "m"}],
                                value="auto", inline=True, style={"fontSize": "12px", "marginBottom": "10px"},
                            ),
                            html.Div("Comparison cap", style={"fontWeight": "700", "fontSize": "13px", "marginTop": "8px"}),
                            dcc.Upload(
                                id="upload-comparison-csv",
                                children=html.Div(["Upload comparison CSV"]),
                                multiple=False,
                                style={
                                    "border": "1px dashed #888", "borderRadius": "6px", "padding": "10px",
                                    "textAlign": "center", "fontSize": "13px", "marginBottom": "8px",
                                },
                            ),
                            html.Button("Clear comparison cap", id="clear-comparison", n_clicks=0, style={"width": "100%", "marginBottom": "8px"}),
                            html.Label("Comparison CSV units", style={"fontWeight": "600", "fontSize": "12px"}),
                            dcc.RadioItems(
                                id="comparison-csv-units",
                                options=[{"label": "Auto", "value": "auto"}, {"label": "mm", "value": "mm"}, {"label": "m", "value": "m"}],
                                value="auto", inline=True, style={"fontSize": "12px", "marginBottom": "12px"},
                            ),
                            html.Hr(),
                            html.H3("View"),
                            html.Div(
                                [
                                    html.Button("Default", id="view-default", n_clicks=0, style=button_style()),
                                    html.Button("Left", id="view-left", n_clicks=0, style=button_style()),
                                    html.Button("Right", id="view-right", n_clicks=0, style=button_style()),
                                    html.Button("Top", id="view-top", n_clicks=0, style=button_style()),
                                    html.Button("Front", id="view-front", n_clicks=0, style=button_style()),
                                    html.Button("Back", id="view-back", n_clicks=0, style=button_style()),
                                ],
                                style={"marginBottom": "10px"},
                            ),
                            html.Hr(),
                            html.H3("Head model"),
                            dcc.RadioItems(
                                id="head-model",
                                options=[
                                    {"label": "Adaptive phantom shell", "value": "phantom"},
                                    {"label": "Realistic public template head mesh", "value": "realistic"},
                                ],
                                value="phantom",
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                            dcc.Checklist(
                                id="allow-fsaverage-download",
                                options=[{"label": "Allow MNE to fetch fsaverage if missing", "value": "fetch"}],
                                value=["fetch"],
                                style={"fontSize": "12px", "marginBottom": "8px"},
                            ),
                            html.Hr(),
                            html.H3("Display controls"),
                            dcc.Checklist(
                                id="show-options",
                                options=[
                                    {"label": "Show scalp/head surface", "value": "head"},
                                    {"label": "Show electrode labels", "value": "labels"},
                                    {"label": "Show matched-electrode displacement lines", "value": "difference-lines"},
                                    {"label": "Show nose/ear landmarks", "value": "landmarks"},
                                    {"label": "Show head mesh edges", "value": "mesh-edges"},
                                ],
                                value=["head", "mesh-edges", "labels", "difference-lines", "landmarks"],
                                style={"fontSize": "13px", "marginBottom": "8px"},
                            ),
                            make_slider("electrode-size", "Electrode size", 3, 28, 1, DEFAULT_ELECTRODE_SIZE, {3: "3", 14: "14", 28: "28"}),
                            make_slider("label-size", "Label font size", 6, 30, 1, DEFAULT_LABEL_SIZE, {6: "6", 18: "18", 30: "30"}),
                            make_slider("hitbox-extra-size", "Invisible click/hover hitbox extra size", 0, 42, 1, DEFAULT_HITBOX_EXTRA_SIZE, {0: "0", 22: "22", 42: "42"}),
                            make_slider("reference-opacity", "Reference cap opacity", 0.1, 1.0, 0.05, DEFAULT_REFERENCE_OPACITY, {0.1: "0.1", 0.5: "0.5", 1.0: "1"}),
                            make_slider("comparison-opacity", "Comparison cap opacity", 0.1, 1.0, 0.05, DEFAULT_COMPARISON_OPACITY, {0.1: "0.1", 0.5: "0.5", 1.0: "1"}),
                            make_slider("head-opacity", "Head opacity", 0.05, 1.0, 0.01, DEFAULT_HEAD_OPACITY, {0.05: "0.05", 0.45: "0.45", 1.0: "1"}),
                            make_slider("projection-offset", "Outward projection offset, mm", 0, 28, 1, DEFAULT_PROJECTION_OFFSET_MM, {0: "0", 8: "8", 28: "28"}),
                            make_slider("mesh-detail", "Head mesh detail", 6, 80, 1, DEFAULT_MESH_DETAIL, {6: "low", 30: "med", 80: "high"}),
                            make_slider("head-scale", "Head scale", 0.75, 1.35, 0.01, DEFAULT_HEAD_SCALE, {0.75: "0.75", 1.0: "1", 1.35: "1.35"}),
                            html.Hr(),
                            html.H3("Color bars"),
                            make_rgb_picker("reference-color", "Reference cap color", DEFAULT_REFERENCE_COLOR),
                            make_rgb_picker("comparison-color", "Comparison cap color", DEFAULT_COMPARISON_COLOR),
                            make_rgb_picker("label-color", "Label color", DEFAULT_LABEL_COLOR),
                            make_rgb_picker("head-color", "Head color", DEFAULT_HEAD_COLOR),
                            make_rgb_picker("background-color", "Background color", DEFAULT_BACKGROUND_COLOR),
                            html.Hr(),
                            html.H3("Selected electrode"),
                            html.Pre(
                                id="electrode-info",
                                children="Click or hover an electrode.",
                                style={
                                    "whiteSpace": "pre-wrap", "fontSize": "12px", "border": "1px solid #ddd",
                                    "backgroundColor": "#f7f7f7", "padding": "10px", "minHeight": "210px",
                                },
                            ),
                        ],
                        style={
                            "width": "360px", "minWidth": "360px", "height": "calc(100vh - 75px)",
                            "overflowY": "auto", "padding": "14px", "borderRight": "1px solid #ddd",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="head-graph",
                                    style={"height": "calc(100vh - 78px)", "width": "100%"},
                                    config={"scrollZoom": True, "displaylogo": False},
                                    clear_on_unhover=False,
                                ),
                                type="default",
                            ),
                            html.Div(
                                "Selection tip: use low mesh detail, increase projection offset, or increase hitbox size if points are difficult to select.",
                                style={"fontSize": "12px", "color": "#555", "padding": "2px 10px"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": 0},
                    ),
                ],
                style={"display": "flex"},
            ),
        ],
        style={"fontFamily": "Arial, Helvetica, sans-serif"},
    )

    @app.callback(
        Output("reference-store", "data"),
        Output("reference-status", "children"),
        Input("upload-reference-csv", "contents"),
        Input("reload-reference-default", "n_clicks"),
        State("upload-reference-csv", "filename"),
        State("reference-csv-units", "value"),
        prevent_initial_call=True,
    )
    def update_reference_data(upload_contents: str | None, reload_clicks: int, upload_filename: str | None, csv_units: str):
        trigger = get_triggered_id()
        try:
            if trigger == "reload-reference-default":
                df = load_default_reference(cap_name="reference")
                return records_from_df(df), format_status(df, "Reference", "Loaded default")
            if upload_contents:
                df = parse_uploaded_csv(upload_contents, upload_filename, csv_units or "auto", cap_name="reference")
                return records_from_df(df), format_status(df, "Reference", "Loaded upload")
            return records_from_df(initial_reference_df), format_status(initial_reference_df, "Reference", "Preserved")
        except Exception as exc:
            err = html.Div([html.Strong("Reference CSV load error: "), html.Span(str(exc))], style={"fontSize": "13px", "color": "#a00000"})
            return records_from_df(initial_reference_df), err

    @app.callback(
        Output("comparison-store", "data"),
        Output("comparison-status", "children"),
        Input("upload-comparison-csv", "contents"),
        Input("clear-comparison", "n_clicks"),
        State("upload-comparison-csv", "filename"),
        State("comparison-csv-units", "value"),
        prevent_initial_call=True,
    )
    def update_comparison_data(upload_contents: str | None, clear_clicks: int, upload_filename: str | None, csv_units: str):
        trigger = get_triggered_id()
        try:
            if trigger == "clear-comparison":
                empty = standard_empty_df(source="empty")
                return EMPTY_RECORDS.copy(), format_status(empty, "Comparison", "Cleared")
            if upload_contents:
                df = parse_uploaded_csv(upload_contents, upload_filename, csv_units or "auto", cap_name="comparison")
                return records_from_df(df), format_status(df, "Comparison", "Loaded upload")
            return records_from_df(initial_comparison_df), format_status(initial_comparison_df, "Comparison", "Preserved")
        except Exception as exc:
            err = html.Div([html.Strong("Comparison CSV load error: "), html.Span(str(exc))], style={"fontSize": "13px", "color": "#a00000"})
            return records_from_df(initial_comparison_df), err

    @app.callback(
        Output("view-store", "data"),
        Input("view-default", "n_clicks"),
        Input("view-left", "n_clicks"),
        Input("view-right", "n_clicks"),
        Input("view-top", "n_clicks"),
        Input("view-front", "n_clicks"),
        Input("view-back", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_view(*_clicks):
        trigger = get_triggered_id() or "view-default"
        return trigger.replace("view-", "")

    @app.callback(
        Output("reference-color-preview", "style"), Output("reference-color-preview", "children"),
        Output("comparison-color-preview", "style"), Output("comparison-color-preview", "children"),
        Output("label-color-preview", "style"), Output("label-color-preview", "children"),
        Output("head-color-preview", "style"), Output("head-color-preview", "children"),
        Output("background-color-preview", "style"), Output("background-color-preview", "children"),
        Input("reference-color-r", "value"), Input("reference-color-g", "value"), Input("reference-color-b", "value"),
        Input("comparison-color-r", "value"), Input("comparison-color-g", "value"), Input("comparison-color-b", "value"),
        Input("label-color-r", "value"), Input("label-color-g", "value"), Input("label-color-b", "value"),
        Input("head-color-r", "value"), Input("head-color-g", "value"), Input("head-color-b", "value"),
        Input("background-color-r", "value"), Input("background-color-g", "value"), Input("background-color-b", "value"),
    )
    def update_color_previews(*values):
        triples = [values[i:i + 3] for i in range(0, len(values), 3)]
        outputs = []
        for triple in triples:
            hex_color = rgb_values_to_hex(*triple)
            outputs.extend([color_preview_style(hex_color), hex_color])
        return outputs

    @app.callback(
        Output("head-graph", "figure"),
        Input("reference-store", "data"),
        Input("comparison-store", "data"),
        Input("electrode-size", "value"),
        Input("label-size", "value"),
        Input("reference-color-r", "value"), Input("reference-color-g", "value"), Input("reference-color-b", "value"),
        Input("comparison-color-r", "value"), Input("comparison-color-g", "value"), Input("comparison-color-b", "value"),
        Input("label-color-r", "value"), Input("label-color-g", "value"), Input("label-color-b", "value"),
        Input("head-color-r", "value"), Input("head-color-g", "value"), Input("head-color-b", "value"),
        Input("background-color-r", "value"), Input("background-color-g", "value"), Input("background-color-b", "value"),
        Input("reference-opacity", "value"),
        Input("comparison-opacity", "value"),
        Input("head-opacity", "value"),
        Input("mesh-detail", "value"),
        Input("projection-offset", "value"),
        Input("head-scale", "value"),
        Input("hitbox-extra-size", "value"),
        Input("show-options", "value"),
        Input("head-model", "value"),
        Input("allow-fsaverage-download", "value"),
        Input("view-store", "data"),
    )
    def update_figure(
        reference_records,
        comparison_records,
        electrode_size,
        label_size,
        ref_r, ref_g, ref_b,
        cmp_r, cmp_g, cmp_b,
        label_r, label_g, label_b,
        head_r, head_g, head_b,
        bg_r, bg_g, bg_b,
        reference_opacity,
        comparison_opacity,
        head_opacity,
        mesh_detail,
        projection_offset,
        head_scale,
        hitbox_extra_size,
        show_options,
        head_model,
        allow_fsaverage_download,
        view_name,
    ):
        reference_color = rgb_values_to_hex(ref_r, ref_g, ref_b)
        comparison_color = rgb_values_to_hex(cmp_r, cmp_g, cmp_b)
        label_color = rgb_values_to_hex(label_r, label_g, label_b)
        head_color = rgb_values_to_hex(head_r, head_g, head_b)
        background_color = rgb_values_to_hex(bg_r, bg_g, bg_b)
        allow_download = "fetch" in (allow_fsaverage_download or [])
        return make_figure(
            reference_records=reference_records or [],
            comparison_records=comparison_records or [],
            electrode_size=int(electrode_size),
            label_size=int(label_size),
            reference_color=reference_color,
            comparison_color=comparison_color,
            label_color=label_color,
            head_color=head_color,
            background_color=background_color,
            reference_opacity=float(reference_opacity),
            comparison_opacity=float(comparison_opacity),
            head_opacity=float(head_opacity),
            mesh_detail=int(mesh_detail),
            projection_offset_mm=float(projection_offset),
            head_scale=float(head_scale),
            hitbox_extra_size=int(hitbox_extra_size),
            show_options=show_options or [],
            head_model=head_model or "phantom",
            allow_fsaverage_download=allow_download,
            view_name=view_name or "default",
        )

    @app.callback(
        Output("electrode-info", "children"),
        Input("head-graph", "clickData"),
        Input("head-graph", "hoverData"),
    )
    def update_selected_electrode(click_data, hover_data):
        data = click_data or hover_data
        if not data or "points" not in data or not data["points"]:
            return "Click or hover an electrode."

        point = data["points"][0]
        custom = point.get("customdata")
        if custom is None or len(custom) < 19:
            return "Click or hover an electrode."

        (
            label, x, y, z, dx, dy, dz, region, hemi, conf, note, source, cap,
            matched, delta_x, delta_y, delta_z, distance, cap_display,
        ) = custom[:19]

        try:
            coord_text = f"x={float(x):.2f}, y={float(y):.2f}, z={float(z):.2f} mm"
            display_text = f"x={float(dx):.2f}, y={float(dy):.2f}, z={float(dz):.2f} mm"
        except Exception:
            coord_text = f"x={x}, y={y}, z={z}"
            display_text = f"x={dx}, y={dy}, z={dz}"

        try:
            delta_text = f"dx={float(delta_x):.2f}, dy={float(delta_y):.2f}, dz={float(delta_z):.2f} mm"
        except Exception:
            delta_text = "not matched"

        return (
            f"Cap: {cap_display}\n"
            f"Electrode: {label}\n"
            f"\nOriginal coordinate, preserved:\n  {coord_text}\n"
            f"\nProjected display coordinate:\n  {display_text}\n"
            f"\nMatched between reference and comparison:\n  {bool(matched)}\n"
            f"\nDistance between caps for this label:\n  {format_distance(distance)}\n"
            f"\nComparison minus reference delta:\n  {delta_text}\n"
            f"\nHemisphere:\n  {hemi}\n"
            f"\nApproximate cortical region:\n  {region}\n"
            f"\nConfidence:\n  {conf}\n"
            f"\nNote:\n  {note}\n"
            f"\nData source:\n  {source}\n"
            f"\nReminder: this is a visualization heuristic, not source localization."
        )

    return app


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--csv", type=str, default=None, help="Legacy alias for --csv-reference.")
    parser.add_argument("--csv-reference", type=str, default=None, help="Optional reference electrode CSV path.")
    parser.add_argument("--csv-compare", type=str, default=None, help="Optional comparison electrode CSV path.")
    parser.add_argument(
        "--csv-units",
        type=str,
        default="auto",
        choices=["auto", "mm", "m"],
        help="Coordinate units for CSV files when a specific reference/compare unit is not set. Default: auto.",
    )
    parser.add_argument("--reference-units", type=str, default=None, choices=["auto", "mm", "m"], help="Reference CSV units.")
    parser.add_argument("--compare-units", type=str, default=None, choices=["auto", "mm", "m"], help="Comparison CSV units.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8050, help="Dash port. Default: 8050")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_path = args.csv_reference or args.csv
    ref_units = args.reference_units or args.csv_units
    cmp_units = args.compare_units or args.csv_units

    if ref_path:
        try:
            initial_reference_df = load_custom_csv(ref_path, units=ref_units, cap_name="reference", source_prefix="reference CSV")
            reference_status = "Loaded reference CSV"
        except Exception as exc:
            print(f"Could not load reference CSV: {exc}")
            print("Falling back to default reference cap.")
            initial_reference_df = load_default_reference(cap_name="reference")
            reference_status = "Loaded default"
    else:
        initial_reference_df = load_default_reference(cap_name="reference")
        reference_status = "Loaded default"

    if args.csv_compare:
        try:
            initial_comparison_df = load_custom_csv(args.csv_compare, units=cmp_units, cap_name="comparison", source_prefix="comparison CSV")
            comparison_status = "Loaded comparison CSV"
        except Exception as exc:
            print(f"Could not load comparison CSV: {exc}")
            initial_comparison_df = standard_empty_df(source="empty")
            comparison_status = "Not loaded"
    else:
        initial_comparison_df = standard_empty_df(source="empty")
        comparison_status = "Not loaded"

    app = build_app(initial_reference_df, initial_comparison_df, reference_status, comparison_status)
    print(f"{APP_TITLE} v{APP_VERSION}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
