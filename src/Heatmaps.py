# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: XY Heatmaps (Before | After | Removed)
# Filters: Species (multi), Management (multi)
# Z-value: chosen column; if non-numeric => 1 per point (counts)
# Smoothing: Gaussian blur in meters (σx, σy), auto-off for single point
# Colors: Light→Dark forest palette, horizontal colorbar under plots
# Overlay: optional Target tree points colored by managementStatusColorHex/managementColorHex
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.io_utils as iou
import re

# ---------- PAGE TITLE ----------
st.markdown("### Heatmaps for selected Species & Management")

# ---------- DATA ----------
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ---------- SETTINGS ----------
CHART_HEIGHT = 420
HEATMAP_NBINS = 50

# Masks for panels
keep_status = {"Target tree", "Untouched"}
mask_after   = df.get("management_status", pd.Series(False, index=df.index)).isin(keep_status)
mask_removed = ~mask_after if "management_status" in df.columns else pd.Series(False, index=df.index)
mask_before  = pd.Series(True, index=df.index)

# ---------- HELPERS ----------
def _is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False

def _unique_sorted(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())

def _safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

# Gaussian smoothing
def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0 or not np.isfinite(sigma_bins):
        return np.array([1.0], dtype=float)
    r = int(max(1, round(3.0 * sigma_bins)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k

def _conv1d(A: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    Ap = np.pad(A, pad_width=pad, mode="reflect")
    if axis == 1:
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=1, arr=Ap)
    else:
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), axis=0, arr=Ap)

def _blur2d(A: np.ndarray, sx_bins: float, sy_bins: float) -> np.ndarray:
    out = A
    if sx_bins > 0:
        out = _conv1d(out, _gaussian_kernel_1d(sx_bins), axis=1)
    if sy_bins > 0:
        out = _conv1d(out, _gaussian_kernel_1d(sy_bins), axis=0)
    return out

# ---------- UI CONTROLS ----------
sp_all = _unique_sorted(df.get("species", pd.Series(dtype=object)))
mg_all = _unique_sorted(df.get("management_status", pd.Series(dtype=object)))

mg_lower_map = {orig: str(orig).strip().lower() for orig in mg_all}
lower_to_orig = {}
for orig, low in mg_lower_map.items():
    lower_to_orig.setdefault(low, orig)

retain_targets = {"untouched", "target tree"}
retain_opts = [lower_to_orig[l] for l in mg_lower_map.values() if l in retain_targets]
remove_opts = [orig for orig, low in mg_lower_map.items() if low not in retain_targets]
retain_opts = retain_opts or ["(none)"]
remove_opts = remove_opts or ["(none)"]

exclude = {"x","y","species","speciesColorHex","management_status","managementColorHex","managementStatusColorHex"}
z_candidates = [c for c in df.columns if c not in exclude]

c1, c2, c3,c4,c5 = st.columns([4,0.5,2,1.5,0.5])
with c1:
    species_sel = st.segmented_control(
        "**Species**",
        options=sp_all if sp_all else ["(none)"],
        default=sp_all if sp_all else ["(none)"],
        selection_mode="multi",
        help="Pick one or more species to include. If none selected, all species are used."
    )
with c3:
    mgmt_keep = st.segmented_control(
        "**Management – Retain in Stand**",
        options=retain_opts,
        default=retain_opts,
        selection_mode="multi",
        help="Retained trees (e.g., Untouched, Target Tree). Empty selection means 'all retained'."
    )
with c4:
    show_targets = st.toggle(
    "Show position of **Target Trees** in map",
    value=False,
    help="Add positions of all 'Target trees' as points over each heatmap. "
    )

c11, c12, c13 = st.columns([3,1.5,4])

with c11:
    z_col = st.selectbox(
        "**Weight value**",
        options=z_candidates if z_candidates else ["(no usable columns)"],
        index=0 if z_candidates else 0,
        help="This column provides the heatmap value. If it is non-numeric, each point contributes 1 (counts)."
    )

with c13:
    mgmt_rm = st.segmented_control(
        "**Management – Remove from Stand**",
        options=remove_opts,
        default=remove_opts,
        selection_mode="multi",
        help="Categories marked for removal. Empty selection means 'all removable'."
    )

# Toggle: overlay Target trees (colored by managementStatusColorHex / managementColorHex)


# Sloučení výběrů managementu
def _normalize_list(lst):
    return [] if (not lst or "(none)" in lst) else lst

mgmt_keep_sel = _normalize_list(mgmt_keep)
mgmt_rm_sel   = _normalize_list(mgmt_rm)

if not mgmt_keep_sel and not mgmt_rm_sel:
    mgmt_sel = mg_all[:]
else:
    mgmt_sel = mgmt_keep_sel + mgmt_rm_sel


# Smoothing sigmas in meters

sigma_x_m = sigma_y_m = 3

# ---------- FILTERED SUBSET ----------
def _apply_filters(dfin: pd.DataFrame, species_sel: list[str], mgmt_sel: list[str]) -> pd.DataFrame:
    d = dfin.copy()
    if "species" in d.columns:
        d["species"] = d["species"].astype(str)
    if "management_status" in d.columns:
        d["management_status"] = d["management_status"].astype(str)
    if species_sel and "(none)" not in species_sel:
        d = d[d["species"].isin(species_sel)]
    if mgmt_sel and "(none)" not in mgmt_sel and "management_status" in d.columns:
        d = d[d["management_status"].isin(mgmt_sel)]
    return d

df_f = _apply_filters(df, species_sel, mgmt_sel)

# ---------- REQUIREMENTS ----------
need = {"x", "y"}
miss = need - set(df_f.columns)
if miss:
    st.warning("Missing columns for heatmaps: " + ", ".join(sorted(miss)))
    st.stop()

x_all = _safe_num(df_f["x"])
y_all = _safe_num(df_f["y"])
valid_xy = x_all.notna() & y_all.notna()
if not valid_xy.any():
    st.info("No valid (x, y) coordinates after filtering.")
    st.stop()

# weights for Z
if z_col in df_f.columns and _is_numeric_like(df_f[z_col]):
    w_all = _safe_num(df_f[z_col]).fillna(0.0)
    colorbar_title = f"{z_col}"
else:
    w_all = pd.Series(1.0, index=df_f.index, dtype=float)
    colorbar_title = "Count"

# ---------- GRID ----------
xmin, xmax = float(x_all[valid_xy].min()), float(x_all[valid_xy].max())
ymin, ymax = float(y_all[valid_xy].min()), float(y_all[valid_xy].max())
if xmin == xmax: xmin, xmax = xmin - 0.5, xmax + 0.5
if ymin == ymax: ymin, ymax = ymin - 0.5, ymax + 0.5

x_edges = np.linspace(xmin, xmax, HEATMAP_NBINS + 1)
y_edges = np.linspace(ymin, ymax, HEATMAP_NBINS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
bin_wx = (xmax - xmin) / HEATMAP_NBINS if HEATMAP_NBINS > 0 else 1.0
bin_wy = (ymax - ymin) / HEATMAP_NBINS if HEATMAP_NBINS > 0 else 1.0

# ---------- PANEL BUILDER ----------
def _panel_hist(mask_panel: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    m = (mask_panel.reindex(df_f.index).fillna(False)) & valid_xy
    if not m.any():
        Z = np.zeros((HEATMAP_NBINS, HEATMAP_NBINS), dtype=float)
        H = np.full_like(Z, "No data", dtype=object)
        return Z, H

    n_points = m.sum()
    Z, _, _ = np.histogram2d(
        x_all[m].to_numpy(float),
        y_all[m].to_numpy(float),
        bins=[x_edges, y_edges],
        weights=w_all[m].to_numpy(float)
    )
    Z = Z.T

    # if only 1 point, skip smoothing
    if n_points > 1 and (sigma_x_m > 0 or sigma_y_m > 0):
        sx_bins = sigma_x_m / bin_wx if bin_wx > 0 else 0.0
        sy_bins = sigma_y_m / bin_wy if bin_wy > 0 else 0.0
        Z = _blur2d(Z, sx_bins, sy_bins)

    Htxt = np.empty_like(Z, dtype=object)
    for iy in range(Z.shape[0]):
        for ix in range(Z.shape[1]):
            val = Z[iy, ix]
            if colorbar_title == "Count":
                Htxt[iy, ix] = f"Count: {val:.0f}"
            else:
                Htxt[iy, ix] = f"{colorbar_title}: {val:.3f}"
    return Z, Htxt

# ---------- PANELS ----------
m_before  = pd.Series(True, index=df_f.index)
if "management_status" in df_f.columns:
    m_after   = df_f["management_status"].isin(keep_status)
    m_removed = ~m_after
else:
    m_after   = pd.Series(False, index=df_f.index)
    m_removed = pd.Series(False, index=df_f.index)

Z_before,  H_before  = _panel_hist(m_before)
Z_after,   H_after   = _panel_hist(m_after)
Z_removed, H_removed = _panel_hist(m_removed)

z_max = float(np.nanmax([Z_before.max(initial=0), Z_after.max(initial=0), Z_removed.max(initial=0)]) or 1.0)

# ---------- COLORS ----------
forest_colorscale = [
    (0.00, "#ffffff"),
    (0.01, "#e8f5e9"),
    (0.50, "#66bb6a"),
    (1.00, "#2e7d32"),
]

# ---------- PLOT ----------
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=(st.session_state.Before, st.session_state.After,st.session_state.Removed),
    specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
    horizontal_spacing=0.02
)

fig.update_layout(
    annotations=[
        dict(
            text=ann.text, 
            x=ann.x, 
            y=ann.y, 
            xref=ann.xref, 
            yref=ann.yref,
            showarrow=False,
            font=st.session_state.plot_title_font
            )
        for ann in fig.layout.annotations
    ]
)

def _add(Z, H, col):
    fig.add_trace(
        go.Heatmap(
            x=x_centers, y=y_centers, z=Z,
            text=H, hoverinfo="text",
            coloraxis="coloraxis"
        ),
        row=1, col=col
    )

_add(Z_before,  H_before,  col=1)
_add(Z_after,   H_after,   col=2)
_add(Z_removed, H_removed, col=3)

# --- Optional overlay: ALL Target trees (from original df), colored by managementStatusColorHex/managementColorHex ---
if show_targets:
    # Find target trees in the original full df (not filtered), so it's "all Target trees"
    m_target = df.get("management_status", pd.Series("", index=df.index)).astype(str).str.strip().str.lower() == "target tree"
    x_t = pd.to_numeric(df.loc[m_target, "x"], errors="coerce")
    y_t = pd.to_numeric(df.loc[m_target, "y"], errors="coerce")
    ok = x_t.notna() & y_t.notna()

    # choose color column: managementStatusColorHex -> managementColorHex -> black
    color_colname = "managementStatusColorHex" if "managementStatusColorHex" in df.columns else ("managementColorHex" if "managementColorHex" in df.columns else None)
    if color_colname:
        col_series = df.loc[m_target, color_colname].astype(str)
        # ensure valid hex, else fallback to black
        valid_hex = re.compile(r"^#([0-9A-Fa-f]{6})$")
        col_series = col_series.where(col_series.str.match(valid_hex), "#000000")
        col_vals = col_series[ok]
    else:
        col_vals = pd.Series("#000000", index=x_t.index)[ok]

    for c in (1, 2, 3):
        fig.add_trace(
            go.Scatter(
                x=x_t[ok], y=y_t[ok],
                mode="markers",
                marker=dict(
                    size=6,
                    color=col_vals.tolist(),  # per-point colors
                    line=dict(color="white", width=0.5)
                ),
                showlegend=False
            ),
            row=1, col=c
        )

# Layout + shared colorbar (horizontal under plots)
fig.update_layout(
    height=CHART_HEIGHT,
    margin=dict(l=10, r=10, t=60, b=100),
    coloraxis=dict(
        colorscale=forest_colorscale,
        cmin=0.0,
        cmax=z_max,
        colorbar=dict(
            title=colorbar_title,
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.25, yanchor="top",
            thickness=12,
            len=0.8
        )
    )
)
fig.update_layout(coloraxis=dict(
    colorbar=dict(len=0.75, thickness=10, y=-0.22)  # kratší a tenčí
))
for c in (1, 2, 3):
    fig.update_xaxes(title_text="x [m]", row=1, col=c, constrain="domain")
    fig.update_yaxes(
        title_text="y [m]" if c == 1 else None,
        row=1, col=c,
        scaleanchor=f"x{c}", scaleratio=1.0
    )

st.plotly_chart(fig, use_container_width=True)
