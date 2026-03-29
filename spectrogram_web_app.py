# =========================
# IMPORTS
# =========================

import csv
import io
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 


# =========================
# CONSTANTS
# =========================

THIRD_OCTAVE_BANDS = np.array([
    1.0,1.25,1.6,2.0,2.5,3.15,4.0,5.0,6.3,8.0,
    10.0,12.5,16.0,20.0,25.0,31.5,40.0,50.0,63.0,
    80.0,100.0,125.0,160.0,200.0,250.0,315.0
], dtype=float)

VC_CURVES_MM_S = {
    "Op Rooms": 0.10000,
    "VC-A": 0.05000,
    "VC-B": 0.02500,
    "VC-C": 0.01250,
    "VC-D": 0.00625,
    "VC-E": 0.00312,
    "VC-F": 0.00156,
    "VC-G": 0.00078,
    "VC-H": 0.00039,
    "VC-I": 0.000195,
    "VC-J": 0.0000975,
    "VC-K": 0.0000488,
}

VC_LABELS = list(VC_CURVES_MM_S.keys())
NON_FREQ_COLS = {"location","equipment","date","time","axis","datetime"}

# =========================
# HELPERS
# =========================

def vc_display_label(label):
    return f"{label} ({VC_CURVES_MM_S[label]} mm/s)"

def bounding_vc_curves(value):
    ordered = sorted(VC_CURVES_MM_S.items(), key=lambda x: x[1], reverse=True)

    if value > ordered[0][1]:
        return f"Above {ordered[0][0]}"
    if value < ordered[-1][1]:
        return f"Below {ordered[-1][0]}"

    for i in range(len(ordered)-1):
        u_label, u_val = ordered[i]
        l_label, l_val = ordered[i+1]
        if u_val >= value >= l_val:
            return f"Between {u_label} and {l_label}"

# =========================
# LOAD CSV
# =========================

def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["datetime"])

    freq_cols = [c for c in df.columns if c not in NON_FREQ_COLS]
    freq_map = {c: float(c) for c in freq_cols}

    return df, freq_cols, freq_map

# =========================
# VC SCALE
# =========================

def build_vc_boundaries(low, high):
    vals = [(k,v) for k,v in VC_CURVES_MM_S.items()
            if VC_CURVES_MM_S[low] <= v <= VC_CURVES_MM_S[high]
            or VC_CURVES_MM_S[high] <= v <= VC_CURVES_MM_S[low]]

    vals = sorted(vals, key=lambda x: x[1])
    labels = [v[0] for v in vals]
    boundaries = np.array([v[1] for v in vals])
    return labels, boundaries

# =========================
# MATRIX
# =========================

def prepare_matrix(df, freq_cols, freq_map, location, axis, start, end, master_times):

    df = df[(df["location"] == location) &
            (df["datetime"] >= start) &
            (df["datetime"] <= end)]

    if axis not in ["x","y","z"]:
        df = df.groupby("datetime")[freq_cols].mean().reset_index()

    else:
        df = df[df["axis"] == axis]

    if df.empty:
        return None, None, None

    pivot = df.pivot_table(index="datetime", values=freq_cols)
    pivot = pivot.reindex(master_times)

    matrix = pivot.values.T
    matrix[matrix <= 0] = np.nan

    return pivot.index, THIRD_OCTAVE_BANDS, matrix

# =========================
# STREAMLIT APP
# =========================

st.title("Spectrogram Viewer (Web)")

uploaded = st.file_uploader("Upload CSV")

if uploaded:

    df, freq_cols, freq_map = load_csv(uploaded)

    locations = df["location"].unique()

    start = st.date_input("Start date")
    end = st.date_input("End date")

    plot_count = st.slider("Number of plots", 1, 6, 3)

    fig = make_subplots(rows=plot_count, cols=1, shared_xaxes=True)

    for i in range(plot_count):

        loc = st.selectbox(f"Location {i+1}", locations, key=f"loc{i}")
        axis = st.selectbox(f"Axis {i+1}", ["x","y","z"], key=f"axis{i}")

        low = st.selectbox(f"Low VC {i+1}", VC_LABELS, index=VC_LABELS.index("VC-K"), key=f"low{i}")
        high = st.selectbox(f"High VC {i+1}", VC_LABELS, index=VC_LABELS.index("Op Rooms"), key=f"high{i}")

        labels, bounds = build_vc_boundaries(low, high)

        master_times = pd.date_range(start=start, end=end, freq="1s")

        times, freqs, matrix = prepare_matrix(
            df, freq_cols, freq_map,
            loc, axis,
            start, end,
            master_times
        )

        if matrix is None:
            continue

        fig.add_trace(
            go.Heatmap(
                x=times,
                y=freqs,
                z=matrix,
                colorscale="Viridis",
                zmin=min(bounds),
                zmax=max(bounds),
                colorbar=dict(
                    tickmode="array",
                    tickvals=bounds,
                    ticktext=[vc_display_label(l) for l in labels]
                )
            ),
            row=i+1, col=1
        )

    st.plotly_chart(fig, use_container_width=True)
