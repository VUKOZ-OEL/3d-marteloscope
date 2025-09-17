# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: Competition for Light (percent-based)
# 1) Bubble: Avg available light (%)
# 2) Bars by Species: who competes (sum of % shading by neighbors)
# 3) Bars by Management: who competes (sum of % shading by neighbors)
# Filters: Stand State (single) + Height range; Species & Management (multi-select) BELOW charts
# ------------------------------------------------------------

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import src.io_utils as iou

# --- panel names from session ---
Before = st.session_state.Before
After = st.session_state.After
Removed = st.session_state.Removed
stand_stat_all = [Before, After]

st.markdown("##### Competition for Light")

# ---------- DATA ----------
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/"
        "PokojnaHora_3df/PokojnaHora.json"
    )
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()


# ---------- HELPERS ----------
def _is_num(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _to_comp_map(v) -> dict:
    """Parse light_comp; supports dict, JSON-str, or None. Keys -> int(treeId), values -> float(%)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {}
    if isinstance(v, dict):
        return {int(k): float(vv) for k, vv in v.items() if _is_num(vv)}
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return {int(k): float(vv) for k, vv in parsed.items() if _is_num(vv)}
        except Exception:
            return {}
    return {}


def _species_colors(d: pd.DataFrame) -> dict:
    if "species" not in d.columns or "speciesColorHex" not in d.columns:
        return {}
    return (
        d.assign(species=lambda x: x["species"].astype(str))
        .groupby("species")["speciesColorHex"]
        .first()
        .to_dict()
    )


def _management_colors(d: pd.DataFrame) -> dict:
    if "management_status" not in d.columns or "managementColorHex" not in d.columns:
        return {}
    cmap = (
        d.assign(ms=lambda x: x["management_status"].astype(str))
        .groupby("ms")["managementColorHex"]
        .first()
        .to_dict()
    )
    return {
        k: (v if isinstance(v, str) and v.strip() else "#AAAAAA")
        for k, v in cmap.items()
    }


def _make_masks(d: pd.DataFrame):
    """Stand state masks for focal-trees by their own management_status."""
    keep_status = {"Target tree", "Untouched"}
    mask_after = d.get("management_status", pd.Series(False, index=d.index)).isin(
        keep_status
    )
    mask_removed = (
        ~mask_after
        if "management_status" in d.columns
        else pd.Series(False, index=d.index)
    )
    mask_before = pd.Series(True, index=d.index)
    return {Before: mask_before, After: mask_after, Removed: mask_removed}


# ---------- STATIC LISTS ----------
sp_all = iou._unique_sorted(df.get("species", pd.Series(dtype=object)))
mg_all = iou._unique_sorted(df.get("management_status", pd.Series(dtype=object)))

# ---------- CANONICALIZE COLUMNS (do this BEFORE slicing df_focal) ----------
# Fallback pro starší export: 'alight_avail'
if "light_avail" not in df.columns and "alight_avail" in df.columns:
    df["light_avail"] = df["alight_avail"]

# Zajistit existenci a rozsah 0..100
df["light_avail"] = (
    pd.to_numeric(df.get("light_avail", 0.0), errors="coerce")
    .fillna(0.0)
    .clip(lower=0.0, upper=100.0)
)

# Bezpečné sestavení mapy sousedů (light_comp_map)
if "light_comp" in df.columns:
    _src_comp = df["light_comp"]
else:
    _src_comp = pd.Series([{}] * len(df), index=df.index)


def _to_comp_map_safe(v):
    try:
        return _to_comp_map(v)
    except Exception:
        return {}


df["light_comp_map"] = _src_comp.apply(_to_comp_map_safe)

# ---------- HEIGHT FIRST ----------
h_series = pd.to_numeric(df.get("height", np.nan), errors="coerce")
Hmax = int(np.nanmax(h_series)) if np.isfinite(np.nanmax(h_series)) else 0
Hmax = max(0, Hmax)

c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])
with c2:
    stand_state = st.segmented_control(
        "**Select Stand State to Show:**",
        options=stand_stat_all,
        default=Before,
        selection_mode="single",
        width="stretch",
        help="Original / Managed / Removed.",
    )
with c4:
    height_min, height_max = st.slider(
        "**Height filter [m]**",
        min_value=0,
        max_value=Hmax,
        value=(0, Hmax),
        step=1,
        help="Zobrazí pouze stromy v zadaném intervalu výšek (aplikuje se jako první).",
    )

# ---------- READ CURRENT FILTER VALUES FROM SESSION ----------
species_sel = st.session_state.get("species_sel", sp_all if sp_all else ["(none)"])
mgmt_sel = st.session_state.get("mgmt_sel", mg_all if mg_all else ["(none)"])
species_sel = (
    list(species_sel) if isinstance(species_sel, (list, tuple, set)) else [species_sel]
)
mg_sel_set = set(mgmt_sel) if isinstance(mgmt_sel, (list, set, tuple)) else {mgmt_sel}

# ---------- BASE MASKS ----------
state_masks = _make_masks(df)
mask_state = state_masks.get(stand_state, pd.Series(True, index=df.index))

mask_height = pd.Series(True, index=df.index)
if "height" in df.columns:
    h = pd.to_numeric(df["height"], errors="coerce")
    mask_height = (h >= height_min) & (h <= height_max)

mask_species = pd.Series(True, index=df.index)
if species_sel and "(none)" not in species_sel:
    mask_species = df["species"].astype(str).isin(species_sel)

mask_mgmt = pd.Series(True, index=df.index)
if mg_sel_set and "(none)" not in mg_sel_set:
    mask_mgmt = df["management_status"].astype(str).isin(mg_sel_set)

# Výběr stromů (výška → stav → druhy/management)
focal_mask = mask_height & mask_state & mask_species & mask_mgmt
df_focal = df.loc[focal_mask].copy()

# Sada ID odstraněných stromů (globálně podle masky Removed)
mask_removed_global = state_masks[Removed]
removed_ids = set(
    pd.to_numeric(df.loc[mask_removed_global, "id"], errors="coerce")
    .dropna()
    .astype(int)
    .tolist()
)


# ---------- RECOMPUTE light_avail / light_comp for After/Removed ----------
def _adjust_light(row, do_adjust: bool):
    """Vrátí (adj_light_avail, adj_comp_map) pro daný strom podle stavu stand_state."""
    la = float(row.get("light_avail", 0.0))
    cmap = row.get("light_comp_map", {})
    if not isinstance(cmap, dict):
        cmap = {}

    if not do_adjust or not removed_ids:
        return la, cmap

    add_back = 0.0
    kept = {}
    for nid, pct in cmap.items():
        try:
            nid_int = int(nid)
        except Exception:
            continue
        if nid_int in removed_ids:
            add_back += float(pct)  # přičteme kolik stínil
        else:
            kept[nid_int] = float(pct)  # ponecháme v comp mapě
    la_new = min(100.0, max(0.0, la + add_back))
    return la_new, kept


do_adjust = stand_state in (After, Removed)

if not df_focal.empty:
    adj_vals = df_focal.apply(lambda r: _adjust_light(r, do_adjust), axis=1)
    df_focal["light_avail_adj"] = [v[0] for v in adj_vals]
    df_focal["light_comp_adj"] = [v[1] for v in adj_vals]
else:
    df_focal["light_avail_adj"] = []
    df_focal["light_comp_adj"] = []

# ---------- NEIGHBOR ATTRS ----------
neighbor_attrs = df[
    ["id", "species", "management_status", "speciesColorHex", "managementColorHex"]
].copy()
neighbor_attrs["id"] = pd.to_numeric(neighbor_attrs["id"], errors="coerce").astype(
    "Int64"
)

# ---------- EXPLODE adjusted light_comp -> df_comp ----------
if not df_focal.empty:
    df_comp = (
        df_focal[["light_comp_adj"]]
        .assign(tmp=lambda t: t["light_comp_adj"].apply(lambda d: list(d.items())))
        .explode("tmp", ignore_index=True)
    )
    df_comp = df_comp.dropna(subset=["tmp"])
    df_comp[["neighbor_id", "shade_pct"]] = pd.DataFrame(
        df_comp["tmp"].tolist(), index=df_comp.index
    )
    df_comp = df_comp.drop(columns=["tmp"])
    df_comp["neighbor_id"] = pd.to_numeric(
        df_comp["neighbor_id"], errors="coerce"
    ).astype("Int64")
    df_comp["shade_pct"] = (
        pd.to_numeric(df_comp["shade_pct"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )
else:
    df_comp = pd.DataFrame(columns=["neighbor_id", "shade_pct"])

# Join na atributy sousedů
df_comp = df_comp.merge(
    neighbor_attrs, left_on="neighbor_id", right_on="id", how="left"
)

# ---------- AGREGACE PRO GRAFY „WHO COMPETES“ ----------
spec_df = (
    df_comp.groupby("species", as_index=False)["shade_pct"].sum()
    if not df_comp.empty
    else pd.DataFrame(columns=["species", "shade_pct"])
).rename(columns={"shade_pct": "value"})

mgmt_df = (
    df_comp.groupby("management_status", as_index=False)["shade_pct"].sum()
    if not df_comp.empty
    else pd.DataFrame(columns=["management_status", "shade_pct"])
).rename(columns={"shade_pct": "value"})

species_all = (
    sorted(df["species"].astype(str).unique().tolist())
    if "species" in df.columns
    else []
)
mgmt_all = (
    df["management_status"].astype(str).unique().tolist()
    if "management_status" in df.columns
    else []
)

if species_all:
    spec_df = (
        spec_df.set_index("species").reindex(species_all, fill_value=0.0).reset_index()
    )
if mgmt_all:
    mgmt_df = (
        mgmt_df.set_index("management_status")
        .reindex(mgmt_all, fill_value=0.0)
        .reset_index()
    )

# color maps
if not df_comp.empty and "speciesColorHex" in df_comp.columns:
    species_cmap = (
        df_comp.dropna(subset=["species"])
        .groupby("species")["speciesColorHex"]
        .first()
        .to_dict()
    )
    for k, v in _species_colors(df).items():
        species_cmap.setdefault(k, v)
else:
    species_cmap = _species_colors(df)

if not df_comp.empty and "managementColorHex" in df_comp.columns:
    mgmt_cmap = (
        df_comp.dropna(subset=["management_status"])
        .groupby("management_status")["managementColorHex"]
        .first()
        .to_dict()
    )
    for k, v in _management_colors(df).items():
        mgmt_cmap.setdefault(k, v)
else:
    mgmt_cmap = _management_colors(df)

# ---------- BUBBLE: Average available light ----------
avg_light = float(df_focal["light_avail_adj"].mean()) if not df_focal.empty else 0.0
p_light = max(0.0, min(100.0, avg_light)) / 100.0  # 0..1

bubble_title = "Average Available Light"
spec_title = "Who Competes · by Species"
mgmt_title = "Who Competes · by Management"

fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(bubble_title, spec_title, mgmt_title),
    horizontal_spacing=0.06,
)

# === 1) BUBBLE (outer 100%, inner area ~ avg light) ===
R = 1.0
r = math.sqrt(p_light) * R
xg, yg = 0.0, 0.0
xr, yr = 0.0, 0.0

fig.add_shape(
    type="circle",
    xref="x1",
    yref="y1",
    x0=xg - R,
    x1=xg + R,
    y0=yg - R,
    y1=yg + R,
    line=dict(width=0),
    fillcolor="rgba(68, 68, 68, 1.0)",
)
fig.add_shape(
    type="circle",
    xref="x1",
    yref="y1",
    x0=xr - r,
    x1=xr + r,
    y0=yr - r,
    y1=yr + r,
    line=dict(width=0),
    fillcolor="rgba(255, 204, 0, 1.0)",
)

fig.add_annotation(
    x=xg,
    y=yg + R + 0.15,
    xref="x1",
    yref="y1",
    text="Total: 100 %",
    showarrow=False,
    font=dict(size=13, color="#222"),
)
fig.add_annotation(
    x=xr,
    y=yr - r - 0.15,
    xref="x1",
    yref="y1",
    text=f"Available: {avg_light:.0f} %",
    showarrow=False,
    font=dict(size=13, color="#222"),
)

fig.update_xaxes(
    row=1, col=1, visible=False, range=[-1.3, 1.3], scaleanchor="y1", scaleratio=1
)
fig.update_yaxes(row=1, col=1, visible=False, range=[-1.3, 1.3])

# === 2) BARS by SPECIES (values in %) ===
x_species = (
    spec_df["species"].astype(str).tolist()
    if not spec_df.empty
    else (species_all or [])
)
y_species = spec_df["value"].tolist() if not spec_df.empty else ([0.0] * len(x_species))
colors_species = [species_cmap.get(s, "#AAAAAA") for s in x_species]
fig.add_trace(
    go.Bar(
        x=x_species,
        y=y_species,
        marker_color=colors_species,
        hovertemplate="Species: %{x}<br>Shade: %{y:.2f} %<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=2,
)

# === 3) BARS by MANAGEMENT (values in %) ===
x_mgmt = (
    mgmt_df["management_status"].astype(str).tolist()
    if not mgmt_df.empty
    else (mgmt_all or [])
)
y_mgmt = mgmt_df["value"].tolist() if not mgmt_df.empty else ([0.0] * len(x_mgmt))
colors_mgmt = [mgmt_cmap.get(m, "#AAAAAA") for m in x_mgmt]
fig.add_trace(
    go.Bar(
        x=x_mgmt,
        y=y_mgmt,
        marker_color=colors_mgmt,
        hovertemplate="Management: %{x}<br>Shade: %{y:.2f} %<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=3,
)

# Layout
fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=40))

# Axes formatting for bars
for c in (2, 3):
    fig.update_xaxes(title_text=None, tickangle=45, row=1, col=c)
    col_vals = y_species if c == 2 else y_mgmt
    ymax = float(max(col_vals)) if col_vals else 1.0
    if ymax <= 0:
        y_upper = 1.0
    else:
        magnitude = 10 ** int(np.floor(np.log10(ymax)))
        step = magnitude / 2
        y_upper = math.ceil(ymax / step) * step
    fig.update_yaxes(
        title_text="Shade contribution [%]", range=[0, y_upper], row=1, col=c
    )

# ---------- LAYOUT: CHART + FILTERS BELOW ----------
c_bot1, c_bot2 = st.columns([2, 15])
with c_bot1:
    st.pills(
        "**Filter Species:**",
        options=sp_all if sp_all else ["(none)"],
        default=species_sel,
        selection_mode="multi",
        help="Pick one or more species.",
        key="species_sel",
    )
with c_bot2:
    st.plotly_chart(fig, use_container_width=True)

c31, c32 = st.columns([2, 15])
with c32:
    st.pills(
        "**Filter Management:**",
        options=mg_all if mg_all else ["(none)"],
        default=list(mg_sel_set) if mg_sel_set else mg_all,
        selection_mode="multi",
        help="Select one or more management categories.",
        key="mgmt_sel",
    )
