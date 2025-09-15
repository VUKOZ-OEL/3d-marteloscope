# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: Crown volume profiles by height (existing)
# + Added: 3-panel XY heatmaps (Before | After | Removed) for Crown volume
# The heatmaps aggregate crown volume [m³] into a 2D grid over x–y coordinates.
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.io_utils as iou

st.markdown("### Explore canopy statistics:")

# --- Data ---
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ========== SETTINGS ==========
CHART_HEIGHT = 420
HEATMAP_NBINS = 50  # number of grid cells along X and Y for heatmaps
Before = "Original Stand"
After = "Managed Stand"
Removed = "Removed from Stand"

# --- masks (Before/After/Removed) ---
keep_status = {"Target tree", "Untouched"}
mask_after   = df.get("management_status", pd.Series(False, index=df.index)).isin(keep_status)
mask_removed = ~mask_after if "management_status" in df.columns else pd.Series(False, index=df.index)

# --- species colors (kept from your original code, used by profiles) ---
def _species_colors(df_all: pd.DataFrame) -> dict:
    if "species" not in df_all.columns or "speciesColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(species=lambda d: d["species"].astype(str),
                         speciesColorHex=lambda d: d["speciesColorHex"])
                 .groupby("species")["speciesColorHex"].first())
    return tmp.to_dict()

# --- nice upper bound for axes ---
def _nice_upper(value: float, steps=(0.5, 1, 2, 5, 10)) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exp = 10 ** math.floor(math.log10(value))
    for s in steps:
        if s * exp >= value:
            return s * exp
    return 10 * exp

# --- parse crownVoxelCountPerMeters into list[int] ---
def _as_counts(seq) -> list[int]:
    if isinstance(seq, (list, tuple, np.ndarray, pd.Series)):
        return [int(x) if pd.notna(x) else 0 for x in seq]
    if isinstance(seq, str):
        try:
            return _as_counts(ast.literal_eval(seq))
        except Exception:
            return []
    return []

# --- compute per-tree crown volume [m³] (robust & vectorized) ---
def crown_volume_per_tree(df_sub: pd.DataFrame) -> pd.Series:
    """
    Compute crown volume per tree by summing voxel counts * voxel_size^3.
    Expects columns: 'crownVoxelCountPerMeters' (list-like/str), 'crownVoxelSize'.
    Returns a float Series aligned to df_sub.index.
    """
    if "crownVoxelSize" not in df_sub.columns or "crownVoxelCountPerMeters" not in df_sub.columns:
        # Missing columns → return zeros
        return pd.Series(0.0, index=df_sub.index, dtype=float)

    # voxel size → m³
    voxel_size_m = pd.to_numeric(df_sub["crownVoxelSize"], errors="coerce")
    voxel_vol = voxel_size_m.pow(3)

    # counts per tree (potentially stored as string-repr of list)
    counts_series = df_sub["crownVoxelCountPerMeters"].apply(_as_counts)

    # sum of counts per tree
    counts_sum = counts_series.apply(lambda lst: float(np.nansum(lst)) if len(lst) else 0.0)

    # final crown volume
    vol = counts_sum * voxel_vol.fillna(0.0)
    vol = vol.fillna(0.0).astype(float)
    return vol

# --- expand subset to long table (height_bin, species, volume) ---
def expand_crown_volume(df_sub: pd.DataFrame) -> pd.DataFrame:
    required = {"species", "crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize"}
    missing = required - set(df_sub.columns)
    if missing:
        st.warning("Missing columns for crown volume profile: " + ", ".join(sorted(missing)))
        return pd.DataFrame(columns=["height_bin", "species", "volume"])

    out_h, out_sp, out_v = [], [], []

    for _, row in df_sub.iterrows():
        sp = str(row.get("species"))
        h0 = pd.to_numeric(row.get("crownStartHeight"), errors="coerce")
        vs = pd.to_numeric(row.get("crownVoxelSize"), errors="coerce")
        seq = row.get("crownVoxelCountPerMeters")

        if not (np.isfinite(h0) and np.isfinite(vs) and vs > 0):
            continue
        counts = _as_counts(seq)
        if not counts:
            continue

        base = int(math.floor(float(h0)))   # first bin = base–base+1 m
        voxel_vol = float(vs) ** 3          # m³

        for i, c in enumerate(counts):
            try:
                c_int = int(c)
            except Exception:
                continue
            if c_int <= 0:
                continue
            hb = base + i                   # integer height bin (m above ground)
            out_h.append(hb)
            out_sp.append(sp)
            out_v.append(c_int * voxel_vol)

    if not out_h:
        return pd.DataFrame(columns=["height_bin", "species", "volume"])

    long = pd.DataFrame({"height_bin": out_h, "species": out_sp, "volume": out_v})
    return long.groupby(["height_bin", "species"], as_index=False)["volume"].sum()

# --- build per-subset profile with zero-filled bins for all heights & species ---
def profiles_by(df_sub: pd.DataFrame, H: int, species_colors: dict) -> pd.DataFrame:
    long = expand_crown_volume(df_sub)
    sp_list = sorted(df_sub["species"].astype(str).unique().tolist()) if "species" in df_sub.columns else []
    if not sp_list:
        full_idx = pd.MultiIndex.from_product([range(0, H), []], names=["height_bin", "species"])
        return pd.DataFrame(index=full_idx).reset_index().assign(volume=0.0, color="#AAAAAA")

    full_idx = pd.MultiIndex.from_product([range(0, H), sp_list], names=["height_bin", "species"])
    prof = (long.set_index(["height_bin", "species"])["volume"]
                .reindex(full_idx, fill_value=0.0)
                .reset_index())
    prof["color"] = prof["species"].map(species_colors).fillna("#AAAAAA")
    return prof

# --- existing main render: profiles by height ---
def render_crown_volume_profiles(df_all: pd.DataFrame):
    # required columns
    need = {"species", "height", "crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize"}
    miss = need - set(df_all.columns)
    if miss:
        st.warning("Missing columns: " + ", ".join(sorted(miss)))
        return

    # Y range 0 .. ceil(max(height))
    hmax = pd.to_numeric(df_all["height"], errors="coerce").max()
    if not np.isfinite(hmax):
        st.warning("Invalid values in 'height'.")
        return
    H = max(1, int(math.ceil(float(hmax))))
    y_centers = np.arange(0, H, 1) + 0.5

    # colors from full dataset (consistent across panels)
    color_map = _species_colors(df_all)

    # subsets
    prof_before  = profiles_by(df_all,            H, color_map)
    prof_after   = profiles_by(df_all[mask_after],   H, color_map)
    prof_removed = profiles_by(df_all[mask_removed], H, color_map)

    # species order by total volume in "Before"
    species_order = (prof_before.groupby("species")["volume"]
                               .sum()
                               .sort_values(ascending=False)
                               .index.tolist())

    # shared X upper (volume)
    x_max = max(prof_before["volume"].max(), prof_after["volume"].max(), prof_removed["volume"].max())
    x_upper = int(math.ceil(x_max / 100.0) * 100) if np.isfinite(x_max) else 1.0

    # subplots
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=(Before, After, Removed),
        horizontal_spacing=0.06
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
                font=st.session_state.plot_title_font,
            )
            for ann in fig.layout.annotations
        ]
    )

    def add_panel(prof_df: pd.DataFrame, col: int, show_legend: bool):
        for sp in species_order:
            d = (prof_df[prof_df["species"] == sp]
                 .set_index("height_bin")
                 .reindex(range(0, H), fill_value=0.0))
            x_series = d["volume"].to_numpy(dtype=float)
            if np.allclose(x_series, 0.0):
                continue
            col_hex = (d["color"].iloc[0] if "color" in d.columns else "#AAAAAA")
            fig.add_trace(
                go.Scatter(
                    x=x_series,
                    y=y_centers,
                    mode="lines",
                    name=sp,
                    legendgroup=sp,
                    showlegend=show_legend,
                    line=dict(shape="spline", width=6),
                    opacity=0.65,
                    hovertemplate=f"Species: {sp}<br>Height: %{{y:.1f}} m<br>Volume: %{{x:.3f}} m³<extra></extra>",
                    marker_color=col_hex
                ),
                row=1, col=col
            )

    add_panel(prof_before,  col=1, show_legend=True)
    add_panel(prof_after,   col=2, show_legend=False)
    add_panel(prof_removed, col=3, show_legend=False)

    # layout + axes
    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5)
    )
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Crown volume [m³]", row=1, col=c, rangemode="tozero", range=[0, x_upper])

    fig.update_yaxes(title_text="Height above ground [m]", row=1, col=1,
                     rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=2, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=3, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)

    st.plotly_chart(fig, use_container_width=True)

# --- NEW: render XY heatmaps of crown volume (Before | After | Removed) ---
def render_crown_volume_heatmaps(df_all: pd.DataFrame, nbins: int = HEATMAP_NBINS):
    # ---- requirements ----
    need = {"x", "y", "species", "crownVoxelCountPerMeters", "crownVoxelSize"}
    miss = need - set(df_all.columns)
    if miss:
        st.warning("Missing columns for crown volume heatmaps: " + ", ".join(sorted(miss)))
        return

    # numeric coords
    x_all = pd.to_numeric(df_all["x"], errors="coerce")
    y_all = pd.to_numeric(df_all["y"], errors="coerce")
    valid_xy = x_all.notna() & y_all.notna()
    if not valid_xy.any():
        st.info("No valid (x, y) coordinates for heatmaps.")
        return

    # species universe
    species_all = sorted(df_all.loc[valid_xy, "species"].astype(str).unique().tolist())

    # ✅ DO NOT write to session_state here. Just read a fallback for plotting:
    sel_species = st.session_state.get("heat_species", species_all)
    if not sel_species:  # guard weird empty state
        sel_species = species_all

    # ALWAYS-ON smoothing with fixed σ = 5 m
    sigma_x_m = 5.0
    sigma_y_m = 5.0

    # weights per tree (m³)
    vol_all = crown_volume_per_tree(df_all).fillna(0.0)

    # global grid
    xmin, xmax = float(x_all[valid_xy].min()), float(x_all[valid_xy].max())
    ymin, ymax = float(y_all[valid_xy].min()), float(y_all[valid_xy].max())
    if xmin == xmax: xmin, xmax = xmin - 0.5, xmax + 0.5
    if ymin == ymax: ymin, ymax = ymin - 0.5, ymax + 0.5

    x_edges = np.linspace(xmin, xmax, nbins + 1)
    y_edges = np.linspace(ymin, ymax, nbins + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    bin_wx = (xmax - xmin) / nbins if nbins > 0 else 1.0
    bin_wy = (ymax - ymin) / nbins if nbins > 0 else 1.0

    # colors
    color_map = _species_colors(df_all)
    def _gradient_white_to(hex_color: str):
        target = hex_color if isinstance(hex_color, str) and hex_color.startswith("#") else "#AAAAAA"
        return [(0.0, "#ffffff"), (1.0, target)]
    MULTI_SPECIES_GREEN = [(0.0, "#ffffff"), (1.0, "#228B22")]

    # smoothing helpers
    def _gaussian_kernel_1d(sigma_bins: float) -> np.ndarray:
        if sigma_bins <= 0: return np.array([1.0], dtype=float)
        r = int(max(1, round(3.0 * sigma_bins)))
        x = np.arange(-r, r + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma_bins) ** 2)
        return k / k.sum()

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

    # panel builder
    def build_panel(mask_panel: pd.Series, top_n: int = 5):
        m_panel = (mask_panel.fillna(False)) & valid_xy
        per_sp_raw = {}
        for sp in sel_species:
            m_sp = m_panel & (df_all["species"].astype(str) == sp)
            if not m_sp.any():
                per_sp_raw[sp] = np.zeros((nbins, nbins), dtype=float)
                continue
            H_sp, _, _ = np.histogram2d(
                x_all[m_sp].to_numpy(float),
                y_all[m_sp].to_numpy(float),
                bins=[x_edges, y_edges],
                weights=vol_all[m_sp].to_numpy(float)
            )
            per_sp_raw[sp] = H_sp.T

        sx = (sigma_x_m / bin_wx) if bin_wx > 0 else 0.0
        sy = (sigma_y_m / bin_wy) if bin_wy > 0 else 0.0
        per_sp = {sp: _blur2d(M, sx, sy) for sp, M in per_sp_raw.items()}

        Z = np.zeros((nbins, nbins), dtype=float)
        for sp in sel_species: Z += per_sp[sp]

        hover = np.empty((nbins, nbins), dtype=object)
        if Z.max(initial=0) <= 0:
            hover[:] = "Volume: 0.000 m³"
        else:
            stack = np.stack([per_sp[sp] for sp in sel_species], axis=-1) if sel_species else np.zeros((nbins, nbins, 0))
            order = np.argsort(-stack, axis=-1)
            for iy in range(nbins):
                for ix in range(nbins):
                    total = Z[iy, ix]
                    if total <= 0:
                        hover[iy, ix] = "Volume: 0.000 m³"; continue
                    parts, taken = [], 0
                    for j in order[iy, ix]:
                        v = stack[iy, ix, j]
                        if v <= 0: continue
                        parts.append(f"{sel_species[j]}: {v:.3f} m³")
                        taken += 1
                        if taken >= top_n: break
                    txt = f"Volume: {total:.3f} m³"
                    if parts: txt += "<br>" + "<br>".join(parts)
                    hover[iy, ix] = txt
        return Z, hover

    # masks
    mask_before  = pd.Series(True, index=df_all.index)
    mask_after   = df.get("management_status", pd.Series(False, index=df_all.index)).isin({"Target tree", "Untouched"})
    mask_removed = ~mask_after if "management_status" in df.columns else pd.Series(False, index=df_all.index)

    # build panels
    Z_before,  Htxt_before  = build_panel(mask_before)
    Z_after,   Htxt_after   = build_panel(mask_after)
    Z_removed, Htxt_removed = build_panel(mask_removed)

    # color scale
    z_max = np.nanmax([Z_before.max(initial=0), Z_after.max(initial=0), Z_removed.max(initial=0)]) or 1.0
    if len(sel_species) == 1:
        colorscale = _gradient_white_to(color_map.get(sel_species[0], "#AAAAAA"))
    else:
        colorscale = [(0.0, "#ffffff"), (1.0, "#228B22")]

    # plot 3 heatmaps
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(Before, After, Removed),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.06
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
                font=st.session_state.plot_title_font,
            )
            for ann in fig.layout.annotations
        ]
    )

    def add_heatmap(Z, Htxt, col):
        fig.add_trace(go.Heatmap(x=x_centers, y=y_centers, z=Z, text=Htxt, hoverinfo="text", coloraxis="coloraxis"),
                      row=1, col=col)

    add_heatmap(Z_before,  Htxt_before,  col=1)
    add_heatmap(Z_after,   Htxt_after,   col=2)
    add_heatmap(Z_removed, Htxt_removed, col=3)

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=60),
        coloraxis=dict(colorscale=colorscale, cmin=0.0, cmax=float(z_max), colorbar=dict(title="Crown vol [m³]"))
    )
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="x [m]", row=1, col=c, constrain="domain")
        fig.update_yaxes(title_text="y [m]" if c == 1 else None, row=1, col=c, scaleanchor=f"x{c}", scaleratio=1.0)

    st.plotly_chart(fig, use_container_width=True)

    # ✅ Render the widget BELOW the plots WITHOUT writing to session_state yourself.
    #    Only pass `default` on first render (when key doesn't exist).
    kwargs = {
        "label": "Species included in heatmaps:",
        "options": species_all,
        "key": "heat_species",
        "help": ("Filters all three heatmaps. When a single species is selected, "
                 "the color scale goes from white to its species color.")
    }
    if "heat_species" not in st.session_state:
        kwargs["default"] = species_all  # provide default ONLY if the key doesn't exist
    st.multiselect(**kwargs)

# ========== PAGE CONTENT ==========
iou.heading_centered("Crown volume profiles by height","darkgrey",5)
render_crown_volume_profiles(df)

st.divider()
iou.heading_centered("Crown volume heatmaps","darkgrey",5)
render_crown_volume_heatmaps(df, nbins=HEATMAP_NBINS)
