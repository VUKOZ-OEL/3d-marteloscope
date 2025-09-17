# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: Crown volume profiles by height
# - "Show Values by" = segmented_control (multi): Species, Management
# - Species vždy SOLID, Management vždy DASHED (nezávislé na tom, co je primary/overlay)
# - Barvy: speciesColorHex / managementColorHex
# - Titulky panelů: Σ crown volume na hektar (zaokrouhleno na celé)
# - Tooltipy: celé hodnoty
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.io_utils as iou

st.markdown("#### Explore canopy statistics: Crown volume profiles by height")

# --- Data ---
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ========== SETTINGS ==========
CHART_HEIGHT = 420
Before  = "Original Stand"
After   = "Managed Stand"
Removed = "Removed from Stand"

# --- UI controls ---
c1, c2, c3 = st.columns([2,2,2])
with c2:
    layers_mode = st.segmented_control(
        "**Show Values by:**",
        options=["Species", "Management"],
        default="Species",
        help="Select one or both. When both are selected, both are shown (Species = solid, Management = dashed).",
        selection_mode="multi",
        width="stretch"
    )

# zajistit list a aspoň jednu vrstvu
if not layers_mode:
    layers_mode = ["Species"]
elif isinstance(layers_mode, str):
    layers_mode = [layers_mode]

# pořadí: první považujeme za primární (jen kvůli pořadí vykreslení/legend), ale styl je dán typem
primary = layers_mode[0]
overlay = layers_mode[1] if len(layers_mode) > 1 else None

# --- area (ha) pro přepočet na hektar ---
try:
    area_ha = float(st.session_state.plot_info['size_ha'].iloc[0])
    if not np.isfinite(area_ha) or area_ha <= 0:
        area_ha = 1.0
except Exception:
    area_ha = 1.0  # fallback

# --- masks (Before/After/Removed) ---
keep_status = {"Target tree", "Untouched"}
if "management_status" in df.columns:
    mask_after   = df["management_status"].isin(keep_status)
    mask_removed = ~mask_after
else:
    mask_after   = pd.Series(False, index=df.index)
    mask_removed = pd.Series(False, index=df.index)

# --- color maps ---
def _species_colors(df_all: pd.DataFrame) -> dict:
    if "species" not in df_all.columns or "speciesColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(species=lambda d: d["species"].astype(str))
                 .groupby("species")["speciesColorHex"].first())
    return tmp.to_dict()

def _management_colors(df_all: pd.DataFrame) -> dict:
    if "management_status" not in df_all.columns or "managementColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(mgmt=lambda d: d["management_status"].astype(str))
                 .groupby("mgmt")["managementColorHex"].first())
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

# --- parse crownVoxelCountPerMeters do list[int] ---
def _as_counts(seq) -> list[int]:
    if isinstance(seq, (list, tuple, np.ndarray, pd.Series)):
        return [int(x) if pd.notna(x) else 0 for x in seq]
    if isinstance(seq, str):
        try:
            return _as_counts(ast.literal_eval(seq))
        except Exception:
            return []
    return []

# --- crown volume per tree [m³] ---
def crown_volume_per_tree(df_sub: pd.DataFrame) -> pd.Series:
    if "crownVoxelSize" not in df_sub.columns or "crownVoxelCountPerMeters" not in df_sub.columns:
        return pd.Series(0.0, index=df_sub.index, dtype=float)
    voxel_size_m = pd.to_numeric(df_sub["crownVoxelSize"], errors="coerce")
    voxel_vol = voxel_size_m.pow(3)
    counts_series = df_sub["crownVoxelCountPerMeters"].apply(_as_counts)
    counts_sum = counts_series.apply(lambda lst: float(np.nansum(lst)) if len(lst) else 0.0)
    vol = counts_sum * voxel_vol.fillna(0.0)
    return vol.fillna(0.0).astype(float)

# --- expand do long (height_bin, group, volume) ---
def expand_crown_volume(df_sub: pd.DataFrame, group_col: str) -> pd.DataFrame:
    required = {"crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize", group_col}
    missing = required - set(df_sub.columns)
    if missing:
        st.warning("Missing columns for crown volume profile: " + ", ".join(sorted(missing)))
        return pd.DataFrame(columns=["height_bin", group_col, "volume"])

    out_h, out_g, out_v = [], [], []
    for _, row in df_sub.iterrows():
        grp = str(row.get(group_col))
        h0  = pd.to_numeric(row.get("crownStartHeight"), errors="coerce")
        vs  = pd.to_numeric(row.get("crownVoxelSize"),   errors="coerce")
        seq = row.get("crownVoxelCountPerMeters")
        if not (np.isfinite(h0) and np.isfinite(vs) and vs > 0):
            continue
        counts = _as_counts(seq)
        if not counts:
            continue
        base = int(math.floor(float(h0)))
        voxel_vol = float(vs) ** 3
        for i, c in enumerate(counts):
            try:
                c_int = int(c)
            except Exception:
                continue
            if c_int <= 0:
                continue
            hb = base + i
            out_h.append(hb)
            out_g.append(grp)
            out_v.append(c_int * voxel_vol)

    if not out_h:
        return pd.DataFrame(columns=["height_bin", group_col, "volume"])

    long = pd.DataFrame({"height_bin": out_h, group_col: out_g, "volume": out_v})
    return long.groupby(["height_bin", group_col], as_index=False)["volume"].sum()

# --- build profile 0-filled ---
def profiles_by(df_sub: pd.DataFrame, H: int, group_col: str, color_map: dict) -> pd.DataFrame:
    long = expand_crown_volume(df_sub, group_col)
    if group_col not in df_sub.columns:
        groups = []
    else:
        groups = sorted(df_sub[group_col].astype(str).unique().tolist())
    if not groups:
        full_idx = pd.MultiIndex.from_product([range(0, H), []], names=["height_bin", group_col])
        return pd.DataFrame(index=full_idx).reset_index().assign(volume=0.0, color="#AAAAAA")
    full_idx = pd.MultiIndex.from_product([range(0, H), groups], names=["height_bin", group_col])
    prof = (long.set_index(["height_bin", group_col])["volume"]
                .reindex(full_idx, fill_value=0.0)
                .reset_index())
    prof["color"] = prof[group_col].map(color_map).fillna("#AAAAAA")
    return prof

# --- main render ---
def render_crown_volume_profiles(df_all: pd.DataFrame, primary: str, overlay: str | None):
    need = {"height", "crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize"}
    miss = need - set(df_all.columns)
    if miss:
        st.warning("Missing columns: " + ", ".join(sorted(miss)))
        return

    # Y rozsah
    hmax = pd.to_numeric(df_all["height"], errors="coerce").max()
    if not np.isfinite(hmax):
        st.warning("Invalid values in 'height'.")
        return
    H = max(1, int(math.ceil(float(hmax))))
    y_centers = np.arange(0, H, 1) + 0.5

    # colormapy
    species_cmap = _species_colors(df_all)
    mgmt_cmap    = _management_colors(df_all)

    # subsety
    df_before  = df_all
    df_after   = df_all[mask_after]
    df_removed = df_all[mask_removed]

    # vybrat primary/overlay sloupce (styl bude řízen typem sloupce)
    def _colormap(col: str):
        return species_cmap if col == "species" else mgmt_cmap

    def _resolve_col(name: str) -> str:
        return "species" if name == "Species" else "management_status"

    primary_col  = _resolve_col(primary)
    overlay_col  = _resolve_col(overlay) if overlay else None
    if overlay_col == primary_col:
        overlay_col = None  # žádné duplikace

    # profily (primary)
    prof_before  = profiles_by(df_before,  H, primary_col, _colormap(primary_col))
    prof_after   = profiles_by(df_after,   H, primary_col, _colormap(primary_col))
    prof_removed = profiles_by(df_removed, H, primary_col, _colormap(primary_col))

    # overlay profily
    overlay_before = overlay_after = overlay_removed = None
    if overlay_col is not None:
        overlay_before  = profiles_by(df_before,  H, overlay_col, _colormap(overlay_col))
        overlay_after   = profiles_by(df_after,   H, overlay_col, _colormap(overlay_col))
        overlay_removed = profiles_by(df_removed, H, overlay_col, _colormap(overlay_col))

    # pořadí skupin dle "Before" (primary)
    group_order = (prof_before.groupby(primary_col)["volume"]
                              .sum()
                              .sort_values(ascending=False)
                              .index.tolist())

    # sdílené X maximum
    x_max = max(prof_before["volume"].max(), prof_after["volume"].max(), prof_removed["volume"].max())
    if overlay_before is not None:
        x_max = max(x_max, overlay_before["volume"].max(), overlay_after["volume"].max(), overlay_removed["volume"].max())
    x_upper = _nice_upper(x_max)

    # Σ na hektar (bez dvojího započtu)
    def _sum_per_ha(dfx: pd.DataFrame) -> float:
        return float(crown_volume_per_tree(dfx).sum()) / float(area_ha if area_ha > 0 else 1.0)

    title_before  = f"{Before} · Σ { _sum_per_ha(df_before):.0f} m³/ha"
    title_after   = f"{After} · Σ { _sum_per_ha(df_after):.0f} m³/ha"
    title_removed = f"{Removed} · Σ { _sum_per_ha(df_removed):.0f} m³/ha"

    # subplots
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=(title_before, title_after, title_removed),
        horizontal_spacing=0.06
    )

    if "plot_title_font" in st.session_state:
        fig.update_layout(
            annotations=[
                dict(
                    text=ann.text, x=ann.x, y=ann.y, xref=ann.xref, yref=ann.yref,
                    showarrow=False, font=st.session_state.plot_title_font,
                )
                for ann in fig.layout.annotations
            ]
        )

    # helper: přidání panelu se stylem daným TYPem group_col (Species=solid, Management=dash)
    def add_panel(prof_df: pd.DataFrame, col: int, show_legend: bool, group_col: str):
        is_species = (group_col == "species")
        width   = 6 if is_species else 4
        dash    = None if is_species else "dash"
        opacity = 0.65 if is_species else 0.8

        order = (prof_df.groupby(group_col)["volume"]
                           .sum()
                           .sort_values(ascending=False)
                           .index.tolist())
        for grp in order:
            d = (prof_df[prof_df[group_col] == grp]
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
                    name=str(grp),
                    legendgroup=f"{group_col}-{grp}",
                    showlegend=show_legend,
                    line=dict(shape="spline", width=width, dash=dash),
                    opacity=opacity,
                    hovertemplate=f"{group_col}: {grp}<br>Height: %{{y:.0f}} m<br>Volume: %{{x:.0f}} m³<extra></extra>",
                    marker_color=col_hex
                ),
                row=1, col=col
            )

    # vykreslit primární (legenda v 1. panelu)
    add_panel(prof_before,  col=1, show_legend=True,  group_col=primary_col)
    add_panel(prof_after,   col=2, show_legend=False, group_col=primary_col)
    add_panel(prof_removed, col=3, show_legend=False, group_col=primary_col)

    # vykreslit overlay (pokud je) — také legenda v 1. panelu, aby zobrazila vše
    if overlay_before is not None:
        add_panel(overlay_before,  col=1, show_legend=True,  group_col=overlay_col)
        add_panel(overlay_after,   col=2, show_legend=False, group_col=overlay_col)
        add_panel(overlay_removed, col=3, show_legend=False, group_col=overlay_col)

    # layout + osy
    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        hovermode="y unified"
    )
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Crown volume [m³]", row=1, col=c, rangemode="tozero", range=[0, x_upper])

    fig.update_yaxes(title_text="Height above ground [m]", row=1, col=1,
                     rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=2, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=3, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)

    st.plotly_chart(fig, use_container_width=True)

# --- PAGE ---
#iou.heading_centered("Crown volume profiles by height", "darkgrey", 5)
render_crown_volume_profiles(df, primary=primary, overlay=overlay)
