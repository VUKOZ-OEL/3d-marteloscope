# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# XY Heatmaps (Before | After | Removed)
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.io_utils as iou
import re


# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------
st.markdown("### Heatmaps for selected Attribute")


# ------------------------------------------------------------
# LOAD + NORMALIZE DATA
# ------------------------------------------------------------
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/"
        "SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
    )
    st.session_state.trees = iou.load_project_json(file_path)

df0 = st.session_state.trees.copy()

# set min=0 for XY
df0["x"] = df0["x"] - df0["x"].min()
df0["y"] = df0["y"] - df0["y"].min()


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
CHART_HEIGHT = 600
HEATMAP_NBINS = 50
keep_status = {"Target tree", "Untouched"}

COLOR_SPP = st.session_state.Species
COLOR_MGMT = st.session_state.Management


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def _is_numeric_like(s):
    try:
        pd.to_numeric(s)
        return True
    except:
        return False


def _normalize_list(lst):
    return [] if (not lst or "(none)" in lst) else lst


# smoothing helpers
def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0 or not np.isfinite(sigma_bins):
        return np.array([1.0])
    r = int(max(1, round(3 * sigma_bins)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= k.sum()
    return k


def _conv1d(A, k, axis):
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    Ap = np.pad(A, pad, mode="reflect")
    if axis == 1:
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, Ap)
    else:
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 0, Ap)


def _blur2d(A, sx_bins, sy_bins):
    out = A
    if sx_bins > 0:
        out = _conv1d(out, _gaussian_kernel_1d(sx_bins), axis=1)
    if sy_bins > 0:
        out = _conv1d(out, _gaussian_kernel_1d(sy_bins), axis=0)
    return out


# ------------------------------------------------------------
# FILTER SOURCES
# ------------------------------------------------------------
sp_all = sorted(df0["species"].astype(str).unique())
mg_all = sorted(df0["management_status"].astype(str).unique())

exclude = {
    "x",
    "y",
    "species",
    "speciesColorHex",
    "management_status",
    "managementColorHex",
    "managementStatusColorHex",
}

z_candidates = [c for c in df0.columns if c not in exclude]


# ------------------------------------------------------------
# UI — TOP
# ------------------------------------------------------------
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 4, 1, 4, 1, 4, 1, 4])

with c2:
    z_col = st.selectbox("**Weight column**", options=z_candidates)

with c4:
    dbh_vals = _safe_num(df0["dbh"]).dropna()
    dbh_range = st.slider(
        "**DBH filter [cm]**",
        int(dbh_vals.min()),
        int(dbh_vals.max()),
        (int(dbh_vals.min()), int(dbh_vals.max())),
    )

with c6:
    hvals = _safe_num(df0["height"]).dropna()
    height_range = st.slider(
        "**Height filter [m]**",
        int(hvals.min()),
        int(hvals.max()),
        (int(hvals.min()), int(hvals.max())),
    )

with c8:
    color_mode = st.segmented_control(
        "**Overlay color by**", options=[COLOR_SPP, COLOR_MGMT], default=COLOR_SPP
    )

show_overlay = st.checkbox("**Show tree positions (small dots)**", value=True)


# ------------------------------------------------------------
# UI — BOTTOM
# ------------------------------------------------------------
cA, cB = st.columns([2, 15])
with cA:
    species_sel = st.pills(
        "**Filter Species:**", sp_all, default=sp_all, selection_mode="multi"
    )
with cB:
    plot_container = st.container()

_, _, cC = st.columns([1, 1, 15])
with cC:
    mgmt_sel = st.pills(
        "**Filter Management:**", mg_all, default=mg_all, selection_mode="multi"
    )

species_sel = _normalize_list(species_sel)
mgmt_sel = _normalize_list(mgmt_sel)


# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------
def _apply_filters(df):
    d = df.copy()
    if species_sel:
        d = d[d["species"].astype(str).isin(species_sel)]
    if mgmt_sel:
        d = d[d["management_status"].astype(str).isin(mgmt_sel)]
    d = d[(d["dbh"] >= dbh_range[0]) & (d["dbh"] <= dbh_range[1])]
    d = d[(d["height"] >= height_range[0]) & (d["height"] <= height_range[1])]
    return d


df_f = _apply_filters(df0)


# ------------------------------------------------------------
# VALID XY
# ------------------------------------------------------------
x_f = _safe_num(df_f["x"])
y_f = _safe_num(df_f["y"])
valid_xy = x_f.notna() & y_f.notna()

if not valid_xy.any():
    st.info("No valid (x,y) after filtering.")
    st.stop()


# ------------------------------------------------------------
# XY RANGE FROM HEATMAP GRID (±1 m)
# ------------------------------------------------------------
buffer = 1.0

x_edges = np.linspace(
    float(df0["x"].min()),
    float(df0["x"].max()),
    HEATMAP_NBINS + 1,
)

y_edges = np.linspace(
    float(df0["y"].min()),
    float(df0["y"].max()),
    HEATMAP_NBINS + 1,
)

xmin = x_edges[0] - buffer
xmax = x_edges[-1] + buffer
ymin = y_edges[0] - buffer
ymax = y_edges[-1] + buffer


# ------------------------------------------------------------
# WEIGHTS
# ------------------------------------------------------------
if _is_numeric_like(df_f[z_col]):
    w = _safe_num(df_f[z_col]).fillna(0)
    colorbar_title = z_col
else:
    w = pd.Series(1, index=df_f.index, dtype=float)
    colorbar_title = "Count"


# ------------------------------------------------------------
# FINAL GRID FOR HISTOGRAM
# ------------------------------------------------------------
x_edges = np.linspace(xmin, xmax, HEATMAP_NBINS + 1)
y_edges = np.linspace(ymin, ymax, HEATMAP_NBINS + 1)

x_cent = (x_edges[:-1] + x_edges[1:]) / 2
y_cent = (y_edges[:-1] + y_edges[1:]) / 2

bin_wx = (xmax - xmin) / HEATMAP_NBINS
bin_wy = (ymax - ymin) / HEATMAP_NBINS


# ------------------------------------------------------------
# PANEL BUILDER
# ------------------------------------------------------------
def _panel(mask):
    m = (mask.reindex(df_f.index, fill_value=False)) & valid_xy

    if not m.any():
        Z = np.zeros((HEATMAP_NBINS, HEATMAP_NBINS))
        H = np.full_like(Z, "No data", dtype=object)
        return Z, H

    Z, _, _ = np.histogram2d(
        x_f[m],
        y_f[m],
        bins=[x_edges, y_edges],
        weights=w[m],
    )
    Z = Z.T

    # smoothing
    if m.sum() > 1:
        sigma = 3
        sx = sigma / bin_wx if bin_wx > 0 else 0
        sy = sigma / bin_wy if bin_wy > 0 else 0
        Z = _blur2d(Z, sx, sy)

    # hover labels
    H = np.empty_like(Z, dtype=object)
    for iy in range(Z.shape[0]):
        for ix in range(Z.shape[1]):
            val = Z[iy, ix]
            H[iy, ix] = (
                f"{colorbar_title}: {val:.2f}"
                if colorbar_title != "Count"
                else f"Count: {val:.0f}"
            )

    return Z, H


# ------------------------------------------------------------
# COMPUTE PANELS
# ------------------------------------------------------------
m_before = pd.Series(True, index=df_f.index)
m_after = df_f["management_status"].isin(keep_status)
m_removed = ~m_after

Zb, Hb = _panel(m_before)
Za, Ha = _panel(m_after)
Zr, Hr = _panel(m_removed)

zmax = max(Zb.max(), Za.max(), Zr.max())


# ------------------------------------------------------------
# FIGURE INIT
# ------------------------------------------------------------
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        st.session_state.Before,
        st.session_state.After,
        st.session_state.Removed,
    ),
    specs=[[{"type": "heatmap"}] * 3],
    horizontal_spacing=0.003,
)


# add panels
def _add(Z, H, col):
    fig.add_trace(
        go.Heatmap(
            x=x_cent,
            y=y_cent,
            z=Z,
            text=H,
            hoverinfo="text",
            coloraxis="coloraxis",
        ),
        row=1,
        col=col,
    )


_add(Zb, Hb, 1)
_add(Za, Ha, 2)
_add(Zr, Hr, 3)


# ------------------------------------------------------------
# OVERLAY POINTS
# ------------------------------------------------------------
if show_overlay:
    df_o = df_f.copy()
    ok = df_o["x"].notna() & df_o["y"].notna()

    if color_mode == COLOR_SPP:
        group_col = "species"
        color_col = "speciesColorHex"
    else:
        group_col = "management_status"
        color_col = "managementColorHex"

    valid_hex = re.compile(r"^#([0-9A-Fa-f]{6})$")
    df_o[color_col] = df_o[color_col].where(
        df_o[color_col].astype(str).str.match(valid_hex),
        "#000000",
    )

    for category, sub in df_o[ok].groupby(group_col):
        colvals = sub[color_col].tolist()
        for c in (1, 2, 3):
            fig.add_trace(
                go.Scatter(
                    x=sub["x"],
                    y=sub["y"],
                    mode="markers",
                    name=str(category),
                    hoverinfo="skip",
                    showlegend=(c == 1),
                    marker=dict(
                        size=4,
                        color=colvals,
                        opacity=0.9,
                        line=dict(width=0),
                    ),
                ),
                row=1,
                col=c,
            )


# ------------------------------------------------------------
# AXES — SQUARE PANELS
# ------------------------------------------------------------
for c in (1, 2, 3):
    fig.update_xaxes(
        range=[xmin, xmax],
        title="x [m]",
        row=1,
        col=c,
        ticks="outside",
        ticklen=4,
        showgrid=False,
    )

    if c == 1:
        fig.update_yaxes(
            range=[ymin, ymax],
            title="y [m]",
            showgrid=False,
            scaleanchor="x1",
            scaleratio=1,
            ticks="outside",
            ticklen=4,
            row=1,
            col=c,
        )
    else:
        fig.update_yaxes(
            range=[ymin, ymax],
            showticklabels=False,
            ticks="",
            title=None,
            showgrid=False,
            scaleanchor=f"x{c}",
            scaleratio=1,
            row=1,
            col=c,
        )


# ------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------
fig.update_layout(
    height=CHART_HEIGHT,
    margin=dict(l=4, r=4, t=50, b=150),
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.15,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.6)",
    ),
    coloraxis=dict(
        colorscale=[
            (0.00, "#ffffff"),
            (0.02, "#e8f5e9"),
            (0.55, "#66bb6a"),
            (1.00, "#2e7d32"),
        ],
        cmin=0,
        cmax=zmax,
        colorbar=dict(
            title=None,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.23,
            yanchor="top",
            thickness=12,
            len=0.70,
        ),
    ),
)


# ------------------------------------------------------------
# RENDER
# ------------------------------------------------------------
with plot_container:
    st.plotly_chart(fig, use_container_width=True)
