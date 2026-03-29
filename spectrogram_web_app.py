import csv
import io
import math
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from plotly.subplots import make_subplots


THIRD_OCTAVE_BANDS = np.array([
    1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
    10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0,
    80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0
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
NON_FREQ_COLS = {"location", "equipment", "date", "time", "axis", "datetime"}
MAX_PLOTS = 6


# ----------------------------
# Utility helpers
# ----------------------------

def format_decimal(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    if value >= 1:
        s = f"{value:.3f}"
    elif value >= 0.1:
        s = f"{value:.4f}"
    elif value >= 0.01:
        s = f"{value:.5f}"
    else:
        s = f"{value:.6f}"
    return s.rstrip("0").rstrip(".") or "0"


def vc_display_label(label: str) -> str:
    return f"{label} ({format_decimal(VC_CURVES_MM_S[label])} mm/s)"


def bounding_vc_curves(value_mm_s: float) -> str:
    ordered = sorted(VC_CURVES_MM_S.items(), key=lambda x: x[1], reverse=True)
    if value_mm_s > ordered[0][1]:
        return f"Above {ordered[0][0]}"
    if value_mm_s < ordered[-1][1]:
        return f"Below {ordered[-1][0]}"
    for i in range(len(ordered) - 1):
        upper_label, upper_val = ordered[i]
        lower_label, lower_val = ordered[i + 1]
        if upper_val >= value_mm_s >= lower_val:
            if np.isclose(value_mm_s, upper_val):
                return upper_label
            if np.isclose(value_mm_s, lower_val):
                return lower_label
            return f"Between {upper_label} and {lower_label}"
    return "Outside VC range"


def normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def standardise_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    canonical = {
        "location": "location",
        "equipment": "equipment",
        "date": "date",
        "time": "time",
        "axis": "axis",
    }
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("\ufeff", "")
        if key in canonical:
            rename_map[col] = canonical[key]
    df = df.rename(columns=rename_map)
    required = {"location", "equipment", "date", "time", "axis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def combine_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_required_columns(normalise_headers(df))
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )
    df = df.dropna(subset=["datetime"]).copy()
    df["axis"] = df["axis"].astype(str).str.strip().str.lower()
    df["location"] = df["location"].astype(str).str.strip()
    df["equipment"] = df["equipment"].astype(str).str.strip()
    return df


def sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        if ";" in sample:
            return ";"
        return ","


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    sep = sniff_delimiter(text[:4096])
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    return combine_datetime(df)


def find_frequency_columns_from_columns(columns) -> dict[str, float]:
    freq_map = {}
    for col in columns:
        if col in NON_FREQ_COLS:
            continue
        try:
            freq_map[col] = float(str(col).strip())
        except ValueError:
            pass
    return freq_map


def energetic_sum(df: pd.DataFrame, freq_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(["datetime", "location", "equipment"], dropna=False)
    rows = []
    for (dt, location, equipment), group in grouped:
        vals = group[freq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        summed = np.sqrt(np.nansum(np.square(vals), axis=0))
        row = {
            "datetime": dt,
            "location": location,
            "equipment": equipment,
            "axis": "vsum(x,y,z)",
        }
        for col, val in zip(freq_cols, summed):
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)


def axis_maximum(df: pd.DataFrame, freq_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(["datetime", "location", "equipment"], dropna=False)
    rows = []
    for (dt, location, equipment), group in grouped:
        vals = group[freq_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        maximum = np.nanmax(vals, axis=0)
        row = {
            "datetime": dt,
            "location": location,
            "equipment": equipment,
            "axis": "max(x,y,z)",
        }
        for col, val in zip(freq_cols, maximum):
            row[col] = val
        rows.append(row)
    return pd.DataFrame(rows)


def build_vc_boundaries(low_label: str, high_label: str):
    low_val = VC_CURVES_MM_S[low_label]
    high_val = VC_CURVES_MM_S[high_label]
    lo = min(low_val, high_val)
    hi = max(low_val, high_val)
    selected = [(label, val) for label, val in VC_CURVES_MM_S.items() if lo <= val <= hi]
    selected_sorted = sorted(selected, key=lambda x: x[1])
    labels = [label for label, _ in selected_sorted]
    values = [val for _, val in selected_sorted]
    if len(values) < 2:
        raise ValueError("Selected VC range must include at least two VC levels.")
    return labels, np.array(values, dtype=float)


def infer_time_step(times: pd.Series | pd.DatetimeIndex):
    times = pd.DatetimeIndex(times).sort_values().unique()
    if len(times) < 2:
        return pd.Timedelta(seconds=1)
    diffs = np.diff(times.view("i8"))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return pd.Timedelta(seconds=1)
    step_ns = int(np.median(diffs))
    return pd.to_timedelta(max(step_ns, 1_000_000_000), unit="ns")


def make_master_time_index(df_window: pd.DataFrame, start, end):
    if df_window is None or df_window.empty:
        return pd.DatetimeIndex([start, end]).drop_duplicates().sort_values()
    step = infer_time_step(df_window["datetime"])
    return pd.date_range(start=start, end=end, freq=step)


def with_nan_as_white(scale):
    if hasattr(scale, "copy"):
        scale = scale.copy()
    scale.set_bad(color="white")
    return scale


# ----------------------------
# Cached data loading
# ----------------------------

@st.cache_data(show_spinner=False)
def load_uploaded_to_parquet_bytes(file_name: str, file_bytes: bytes):
    text = file_bytes.decode("utf-8-sig", errors="replace")
    sep = sniff_delimiter(text[:4096])
    raw_df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df = combine_datetime(raw_df)
    freq_map = find_frequency_columns_from_columns(df.columns)
    freq_cols = list(freq_map.keys())

    tmpdir = Path(tempfile.gettempdir()) / "spectrogram_web_cache"
    tmpdir.mkdir(parents=True, exist_ok=True)
    parquet_path = tmpdir / f"{Path(file_name).name}.parquet"
    df.to_parquet(parquet_path, index=False)

    meta = {
        "parquet_path": str(parquet_path),
        "rows": len(df),
        "freq_map": freq_map,
        "freq_cols": freq_cols,
        "locations": sorted(df["location"].dropna().astype(str).unique().tolist()),
        "dt_min": df["datetime"].min(),
        "dt_max": df["datetime"].max(),
    }
    return meta


@st.cache_data(show_spinner=False)
def load_dataframe_from_parquet(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


@st.cache_data(show_spinner=False)
def fast_filter_parquet(parquet_path: str, location: str, start, end) -> pd.DataFrame:
    try:
        query = """
            SELECT *
            FROM read_parquet(?)
            WHERE location = ?
              AND datetime >= ?
              AND datetime <= ?
        """
        return duckdb.execute(query, [parquet_path, location, pd.Timestamp(start), pd.Timestamp(end)]).df()
    except Exception:
        df = pd.read_parquet(parquet_path)
        pl_df = pl.from_pandas(df)
        return pl_df.filter(
            (pl.col("location") == location) &
            (pl.col("datetime") >= pl.lit(start)) &
            (pl.col("datetime") <= pl.lit(end))
        ).to_pandas()


def prepare_matrix_fast(parquet_path: str, freq_cols: list[str], freq_map: dict[str, float],
                        location: str, axis: str, start_time, end_time, master_times: pd.DatetimeIndex):
    df = fast_filter_parquet(parquet_path, location, start_time, end_time)
    if df.empty:
        return None, None, None

    if axis == "vsum(x,y,z)":
        df = energetic_sum(df, freq_cols)
    elif axis == "max(x,y,z)":
        df = axis_maximum(df, freq_cols)
    else:
        df = df[df["axis"] == axis]

    if df.empty:
        return None, None, None

    df = df.copy()
    for col in freq_cols:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    pivot = df.pivot_table(index="datetime", values=freq_cols, aggfunc="mean").sort_index()
    if pivot.empty:
        return None, None, None

    pivot = pivot.reindex(master_times)

    matrix = np.full((len(THIRD_OCTAVE_BANDS), len(pivot.index)), np.nan, dtype=float)
    col_freqs = np.array([freq_map[c] for c in pivot.columns], dtype=float)

    for j, f_target in enumerate(THIRD_OCTAVE_BANDS):
        matches = np.where(np.isclose(col_freqs, f_target, rtol=0, atol=1e-9))[0]
        if len(matches) == 1:
            src_col = pivot.columns[matches[0]]
            matrix[j, :] = pivot[src_col].to_numpy(dtype=float)

    matrix[matrix <= 0] = np.nan
    return pivot.index, THIRD_OCTAVE_BANDS.copy(), matrix


def build_selected_window_df(parquet_path: str, start, end):
    df = load_dataframe_from_parquet(parquet_path)
    return df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()


# ----------------------------
# Plotting
# ----------------------------

def add_plot_trace(fig, row, times, freqs, matrix_mm_s, low_label, high_label, banded, location, axis_label):
    vc_labels, vc_boundaries = build_vc_boundaries(low_label, high_label)
    vmin = float(np.min(vc_boundaries))
    vmax = float(np.max(vc_boundaries))

    if banded:
        n_intervals = len(vc_boundaries) - 1
        base = plt.cm.viridis(np.linspace(0.12, 0.95, n_intervals))
        colorscale = []
        for i in range(n_intervals):
            p0 = i / n_intervals
            p1 = (i + 1) / n_intervals
            color = tuple((base[i, :3] * 255).astype(int))
            rgb = f"rgb({color[0]},{color[1]},{color[2]})"
            colorscale.append([p0, rgb])
            colorscale.append([p1, rgb])
    else:
        colorscale = "Viridis"

    z = np.array(matrix_mm_s, dtype=float)

    fig.add_trace(
        go.Heatmap(
            x=times,
            y=freqs,
            z=z,
            zmin=vmin,
            zmax=vmax,
            colorscale=colorscale,
            hovertemplate=(
                "%{x|%d/%m/%Y %H:%M:%S}<br>"
                "%{y:.3g} Hz<br>"
                "%{z:.6g} mm/s<extra></extra>"
            ),
            colorbar=dict(
                tickmode="array",
                tickvals=vc_boundaries.tolist(),
                ticktext=[vc_display_label(label) for label in vc_labels],
                ticks="outside",
                ticklen=6,
                x=1.02 + (row - 1) * 0.02,
            ),
            showscale=True,
        ),
        row=row,
        col=1,
    )

    if banded and len(vc_boundaries) > 2:
        x_num = mdates.date2num(pd.DatetimeIndex(times).to_pydatetime())
        X, Y = np.meshgrid(x_num, freqs)
        try:
            cs = plt.contour(X, Y, z, levels=vc_boundaries[1:-1])
            for collection in cs.collections:
                for path in collection.get_paths():
                    v = path.vertices
                    fig.add_trace(
                        go.Scatter(
                            x=mdates.num2date(v[:, 0]),
                            y=v[:, 1],
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0.35)", width=1),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
            plt.close()
        except Exception:
            pass

    fig.update_yaxes(type="log", row=row, col=1, title_text="Frequency (Hz, 1/3 octave)")
    fig.update_xaxes(row=row, col=1)
    fig.layout.annotations[row - 1].text = f"{location} - {axis_label}"


def build_hover_summary(value_mm_s: float) -> str:
    return bounding_vc_curves(value_mm_s)


# ----------------------------
# Streamlit app
# ----------------------------

st.set_page_config(page_title="Vibration Spectrogram Viewer", layout="wide")
st.title("Vibration Spectrogram Viewer")

uploaded = st.file_uploader("Load CSV", type=["csv", "txt"])

if "plot_count" not in st.session_state:
    st.session_state.plot_count = 3
if "plot_cfg" not in st.session_state:
    st.session_state.plot_cfg = []

if uploaded is not None:
    with st.spinner("Loading and caching file..."):
        meta = load_uploaded_to_parquet_bytes(uploaded.name, uploaded.getvalue())

    parquet_path = meta["parquet_path"]
    locations = meta["locations"]
    freq_cols = meta["freq_cols"]
    freq_map = meta["freq_map"]
    dt_min = pd.Timestamp(meta["dt_min"])
    dt_max = pd.Timestamp(meta["dt_max"])

    st.success(f"Loaded {meta['rows']} rows. Range: {dt_min.strftime('%d/%m/%Y %H:%M:%S')} to {dt_max.strftime('%d/%m/%Y %H:%M:%S')}")

    c1, c2, c3, c4, c5, c6 = st.columns([1.4, 1.2, 1.4, 1.2, 1.0, 1.0])
    with c1:
        start_date = st.date_input("Start date", value=dt_min.date(), format="DD/MM/YYYY")
    with c2:
        start_time = st.time_input("Start time", value=dt_min.to_pydatetime().time().replace(second=0, microsecond=0), step=60)
    with c3:
        end_date = st.date_input("End date", value=dt_max.date(), format="DD/MM/YYYY")
    with c4:
        end_time = st.time_input("End time", value=dt_max.to_pydatetime().time().replace(second=0, microsecond=0), step=60)
    with c5:
        fixed_scale = st.checkbox("Fix colour scale", value=False)
    with c6:
        banded_vc = st.checkbox("Banded VC classes", value=True)

    gc1, gc2, gc3, gc4 = st.columns([1, 1, 1, 1])
    with gc1:
        if st.button("Add plot"):
            st.session_state.plot_count = min(MAX_PLOTS, st.session_state.plot_count + 1)
    with gc2:
        if st.button("Remove plot"):
            st.session_state.plot_count = max(1, st.session_state.plot_count - 1)
    with gc3:
        global_low = st.selectbox("Global Low VC", VC_LABELS, index=VC_LABELS.index("VC-K"), disabled=not fixed_scale)
    with gc4:
        global_high = st.selectbox("Global High VC", VC_LABELS, index=VC_LABELS.index("Op Rooms"), disabled=not fixed_scale)

    while len(st.session_state.plot_cfg) < st.session_state.plot_count:
        defaults = ["x", "y", "z", "Vsum(x,y,z)", "max(x,y,z)", "x"]
        i = len(st.session_state.plot_cfg)
        st.session_state.plot_cfg.append({
            "location": locations[0] if locations else "",
            "axis": defaults[min(i, len(defaults)-1)],
            "low_vc": "VC-K",
            "high_vc": "Op Rooms",
        })
    st.session_state.plot_cfg = st.session_state.plot_cfg[:st.session_state.plot_count]

    axis_options = ["x", "y", "z", "Vsum(x,y,z)", "max(x,y,z)"]

    for i in range(st.session_state.plot_count):
        st.markdown(f"**Plot {i+1}**")
        pc1, pc2, pc3, pc4 = st.columns([2.5, 1.2, 1.1, 1.1])
        with pc1:
            loc = st.selectbox(
                f"Location {i+1}", locations,
                index=locations.index(st.session_state.plot_cfg[i]["location"]) if st.session_state.plot_cfg[i]["location"] in locations else 0,
                key=f"loc_{i}"
            )
        with pc2:
            axis = st.selectbox(
                f"Axis {i+1}", axis_options,
                index=axis_options.index(st.session_state.plot_cfg[i]["axis"]) if st.session_state.plot_cfg[i]["axis"] in axis_options else 0,
                key=f"axis_{i}"
            )
        with pc3:
            low_vc = st.selectbox(
                f"Low VC {i+1}", VC_LABELS,
                index=VC_LABELS.index(st.session_state.plot_cfg[i]["low_vc"]) if st.session_state.plot_cfg[i]["low_vc"] in VC_LABELS else VC_LABELS.index("VC-K"),
                disabled=fixed_scale,
                key=f"low_{i}"
            )
        with pc4:
            high_vc = st.selectbox(
                f"High VC {i+1}", VC_LABELS,
                index=VC_LABELS.index(st.session_state.plot_cfg[i]["high_vc"]) if st.session_state.plot_cfg[i]["high_vc"] in VC_LABELS else VC_LABELS.index("Op Rooms"),
                disabled=fixed_scale,
                key=f"high_{i}"
            )

        st.session_state.plot_cfg[i] = {
            "location": loc,
            "axis": axis,
            "low_vc": low_vc,
            "high_vc": high_vc,
        }

    start = pd.Timestamp.combine(pd.Timestamp(start_date).date(), start_time)
    end = pd.Timestamp.combine(pd.Timestamp(end_date).date(), end_time)

    if end < start:
        st.error("End time must be after start time.")
    else:
        selected_window_df = build_selected_window_df(parquet_path, start, end)
        master_times = make_master_time_index(selected_window_df, start, end)

        fig = make_subplots(
            rows=st.session_state.plot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[""] * st.session_state.plot_count,
        )

        hover_table_rows = []

        for i in range(st.session_state.plot_count):
            cfg = st.session_state.plot_cfg[i]
            low_label = global_low if fixed_scale else cfg["low_vc"]
            high_label = global_high if fixed_scale else cfg["high_vc"]

            try:
                _ = build_vc_boundaries(low_label, high_label)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            times, freqs, matrix = prepare_matrix_fast(
                parquet_path,
                freq_cols,
                freq_map,
                cfg["location"],
                cfg["axis"].lower(),
                start,
                end,
                master_times,
            )

            if matrix is None:
                fig.add_annotation(text="No data", x=0.5, y=0.5, xref=f"x{i+1} domain", yref=f"y{i+1} domain", showarrow=False)
                continue

            add_plot_trace(fig, i + 1, times, freqs, matrix, low_label, high_label, banded_vc, cfg["location"], cfg["axis"])

            finite = np.argwhere(np.isfinite(matrix))
            if finite.size > 0:
                r, c = finite[0]
                hover_table_rows.append({
                    "Plot": i + 1,
                    "Location": cfg["location"],
                    "Axis": cfg["axis"],
                    "Time": pd.Timestamp(times[c]).strftime("%d/%m/%Y %H:%M:%S"),
                    "Frequency (Hz)": format_decimal(freqs[r]),
                    "Velocity (mm/s)": format_decimal(matrix[r, c]),
                    "VC Band": build_hover_summary(matrix[r, c]),
                })

        fig.update_layout(height=max(350, 280 * st.session_state.plot_count), margin=dict(l=60, r=240, t=40, b=40))
        fig.update_xaxes(range=[master_times[0], master_times[-1]])
        st.plotly_chart(fig, use_container_width=True)

        if hover_table_rows:
            st.caption("Example hover summary format")
            st.dataframe(pd.DataFrame(hover_table_rows), use_container_width=True, hide_index=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            png_bytes = fig.to_image(format="png")
            st.download_button("Export plots PNG", png_bytes, file_name="spectrogram_plots.png", mime="image/png")
        with ec2:
            pdf_bytes = fig.to_image(format="pdf")
            st.download_button("Export plots PDF", pdf_bytes, file_name="spectrogram_plots.pdf", mime="application/pdf")

        export_parts = []
        for i in range(st.session_state.plot_count):
            cfg = st.session_state.plot_cfg[i]
            df_sel = fast_filter_parquet(parquet_path, cfg["location"], start, end)
            if df_sel.empty:
                continue
            if cfg["axis"].lower() == "vsum(x,y,z)":
                df_sel = energetic_sum(df_sel, freq_cols)
            elif cfg["axis"].lower() == "max(x,y,z)":
                df_sel = axis_maximum(df_sel, freq_cols)
            else:
                df_sel = df_sel[df_sel["axis"] == cfg["axis"].lower()].copy()
            if df_sel.empty:
                continue
            df_sel.insert(0, "plot", f"Plot {i+1}")
            df_sel.insert(1, "selected_axis", cfg["axis"])
            export_parts.append(df_sel)

        if export_parts:
            export_df = pd.concat(export_parts, ignore_index=True)
            if "datetime" in export_df.columns:
                export_df = export_df.sort_values(["plot", "location", "datetime"]).drop(columns=["datetime"])
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button("Export window CSV", csv_bytes, file_name="selected_window.csv", mime="text/csv")
else:
    st.info("Upload a CSV or TXT export to begin.")
