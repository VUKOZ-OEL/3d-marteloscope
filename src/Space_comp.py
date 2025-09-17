# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: Competition for Space (Shared Crown Voxels)
# 1) Bubble: Shared vs Total crown volume
# 2) Bars by Species: who competes (sum of neighbor crown volumes that are shared)
# 3) Bars by Management: who competes (sum of neighbor crown volumes that are shared)
# Filters: Stand State (single) + Show as on top; Species & Management (multi-select) BELOW charts
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

st.markdown("##### Competition for Space")

# ---------- DATA ----------
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/"
        "PokojnaHora_3df/PokojnaHora.json"
    )
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()


# ---------- HELPERS ----------
def _to_list_of_dicts(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict) and "treeId" in x and "count" in x]
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [
                    x
                    for x in parsed
                    if isinstance(x, dict) and "treeId" in x and "count" in x
                ]
        except Exception:
            return []
    return []


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


def _crown_volume_series(d: pd.DataFrame) -> pd.Series:
    cnt = pd.to_numeric(d.get("crownVoxelCount", 0), errors="coerce").fillna(0.0)
    vsz = pd.to_numeric(d.get("crownVoxelSize", 0), errors="coerce").fillna(0.0)
    return (cnt * (vsz**3)).astype(float)


def _shared_volume_for_tree_row(row) -> float:
    vsz = pd.to_numeric(row.get("crownVoxelSize"), errors="coerce")
    if not np.isfinite(vsz) or vsz <= 0:
        return 0.0
    voxel_vol = float(vsz) ** 3
    items = _to_list_of_dicts(row.get("crownVoxelCountShared"))
    total_shared_vox = float(sum(max(0, int(it.get("count", 0))) for it in items))
    return total_shared_vox * voxel_vol


def _make_masks(d: pd.DataFrame):
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

# ---------- TOP CONTROLS (Stand State + Show as) ----------
c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 2, 1])
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
    show_mode = st.segmented_control(
        "**Show Values as:**",
        options=["Percentage (%)", "Volume (m³)"],
        default="Percentage (%)",
        width="stretch",
        help="Percent uses total shared space as 100 %.",
    )

# ---------- READ CURRENT FILTER VALUES FROM SESSION (for use NOW) ----------
species_sel = st.session_state.get("species_sel", sp_all if sp_all else ["(none)"])
mgmt_sel = st.session_state.get("mgmt_sel", mg_all if mg_all else ["(none)"])

# normalize
species_sel = (
    list(species_sel) if isinstance(species_sel, (list, tuple, set)) else [species_sel]
)
mg_sel_set = set(mgmt_sel) if isinstance(mgmt_sel, (list, set, tuple)) else {mgmt_sel}

# ---------- DERIVED COLUMNS ----------
df["crown_volume_ref"] = _crown_volume_series(df)

# Sousedi – atributy podle jejich ID (ID sloupce = 'id' souseda)
neighbor_attrs = df[
    ["id", "species", "management_status", "speciesColorHex", "managementColorHex"]
].copy()
neighbor_attrs["id"] = pd.to_numeric(neighbor_attrs["id"], errors="coerce").astype(
    "Int64"
)

# ---------- BUILD FOCAL SUBSET ----------
mask_species = pd.Series(True, index=df.index)
if species_sel and "(none)" not in species_sel:
    mask_species = df["species"].astype(str).isin(species_sel)

mask_mgmt = pd.Series(True, index=df.index)
if mg_sel_set and "(none)" not in mg_sel_set:
    mask_mgmt = df["management_status"].astype(str).isin(mg_sel_set)

state_masks = _make_masks(df)
mask_state = state_masks.get(stand_state, pd.Series(True, index=df.index))

focal_mask = mask_species & mask_mgmt & mask_state
df_focal = df.loc[focal_mask].copy()

# shared volume per focal tree (baseline, před úpravou masek)
if not df_focal.empty:
    df_focal["shared_volume_base"] = df_focal.apply(_shared_volume_for_tree_row, axis=1)
else:
    df_focal["shared_volume_base"] = []

# ---------- EXPLODE crownVoxelCountShared -> df_shared (s fokálním ID) ----------
# (DŮLEŽITÉ: vezmeme i fokální id, abychom uměli pro každý strom spočítat, kolik mu stínili odstranění sousedi)
df_focal["_shared_list"] = df_focal["crownVoxelCountShared"].apply(_to_list_of_dicts)
df_shared = (
    df_focal[["id", "crownVoxelSize", "_shared_list"]]
    .rename(columns={"id": "focal_id"})
    .explode("_shared_list", ignore_index=True)
)
df_shared = df_shared.dropna(subset=["_shared_list"])


def _safe_get(d, key):
    try:
        return d.get(key, None)
    except Exception:
        return None


df_shared["neighbor_id"] = pd.to_numeric(
    df_shared["_shared_list"].apply(lambda d: _safe_get(d, "treeId")), errors="coerce"
).astype("Int64")
df_shared["count"] = (
    pd.to_numeric(
        df_shared["_shared_list"].apply(lambda d: _safe_get(d, "count")),
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
    .clip(lower=0)
)
df_shared["voxel_vol"] = pd.to_numeric(
    df_shared["crownVoxelSize"], errors="coerce"
).pow(3)
df_shared["shared_volume"] = (df_shared["count"] * df_shared["voxel_vol"]).fillna(0.0)

# Připojíme atributy sousedů (pro „Who competes“)
df_shared = df_shared.merge(
    neighbor_attrs,
    left_on="neighbor_id",
    right_on="id",
    how="left",
    suffixes=("", "_nbr"),
)

# ---------- MASK LOGIC (After/Removed): vyřadit z df_shared odstraněné sousedy a jejich objem přičíst k volnému prostoru ----------
# sada ID odstraněných stromů (globálně podle masky Removed)
removed_ids = set(
    pd.to_numeric(df.loc[state_masks[Removed], "id"], errors="coerce")
    .dropna()
    .astype(int)
    .tolist()
)

# označíme řádky, kde soused je odstraněný
df_shared["neighbor_removed"] = (
    df_shared["neighbor_id"].astype("Int64").isin(list(removed_ids))
)

# kolik sdíleného objemu tvořili odstranění sousedi – po FOKÁLNÍCH stromech
removed_shared_by_focal = (
    df_shared.loc[df_shared["neighbor_removed"]]
    .groupby("focal_id")["shared_volume"]
    .sum()
    .rename("removed_shared")
)

# baseline sdílený objem na strom
df_focal = df_focal.merge(
    removed_shared_by_focal, left_on="id", right_index=True, how="left"
).assign(removed_shared=lambda t: t["removed_shared"].fillna(0.0))

# upravený (adjusted) sdílený objem pro fokální strom po aplikaci masky
do_adjust = stand_state in (After, Removed)
if do_adjust:
    df_focal["shared_volume_adj"] = (
        df_focal["shared_volume_base"] - df_focal["removed_shared"]
    ).clip(lower=0.0)
else:
    df_focal["shared_volume_adj"] = df_focal["shared_volume_base"]

# celkový ref objem fokální koruny a volný objem po úpravě
df_focal["crown_volume_ref"] = pd.to_numeric(
    df_focal["crown_volume_ref"], errors="coerce"
).fillna(0.0)
df_focal["free_volume_adj"] = (
    df_focal["crown_volume_ref"] - df_focal["shared_volume_adj"]
).clip(lower=0.0)

# ---------- AGREGACE PRO BAR CHARTS (po vyřazení odstraněných sousedů) ----------
df_shared_kept = (
    df_shared[~df_shared["neighbor_removed"]] if do_adjust else df_shared.copy()
)

spec_df = (
    df_shared_kept.groupby("species", as_index=False)["shared_volume"].sum()
    if not df_shared_kept.empty
    else pd.DataFrame(columns=["species", "shared_volume"])
).rename(columns={"shared_volume": "value"})

mgmt_df = (
    df_shared_kept.groupby("management_status", as_index=False)["shared_volume"].sum()
    if not df_shared_kept.empty
    else pd.DataFrame(columns=["management_status", "shared_volume"])
).rename(columns={"shared_volume": "value"})

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
if not df_shared_kept.empty and "speciesColorHex" in df_shared_kept.columns:
    species_cmap = (
        df_shared_kept.dropna(subset=["species"])
        .groupby("species")["speciesColorHex"]
        .first()
        .to_dict()
    )
    for k, v in _species_colors(df).items():
        species_cmap.setdefault(k, v)
else:
    species_cmap = _species_colors(df)

if not df_shared_kept.empty and "managementColorHex" in df_shared_kept.columns:
    mgmt_cmap = (
        df_shared_kept.dropna(subset=["management_status"])
        .groupby("management_status")["managementColorHex"]
        .first()
        .to_dict()
    )
    for k, v in _management_colors(df).items():
        mgmt_cmap.setdefault(k, v)
else:
    mgmt_cmap = _management_colors(df)

# ---------- SUMÁŘE PRO BUBLINU ----------
total_ref_vol = float(df_focal["crown_volume_ref"].sum()) if not df_focal.empty else 0.0
total_shared_vol = (
    float(df_focal["shared_volume_adj"].sum()) if not df_focal.empty else 0.0
)
non_shared_vol = max(0.0, total_ref_vol - total_shared_vol)

# ---------- NORMALIZATION (Show as %) ----------
if show_mode == "Percentage (%)":
    denom = total_shared_vol if total_shared_vol > 0 else 1.0
    spec_plot_vals = spec_df.assign(plot_val=lambda t: (t["value"] / denom) * 100.0)
    mgmt_plot_vals = mgmt_df.assign(plot_val=lambda t: (t["value"] / denom) * 100.0)
    y_title_bars = "Share of shared space [%]"
    total_label = "Total crown: 100 %"
    shared_label = f"Shared: { (total_shared_vol / total_ref_vol * 100.0) if total_ref_vol > 0 else 0.0 :.0f} %"
else:
    spec_plot_vals = spec_df.assign(plot_val=lambda t: t["value"])
    mgmt_plot_vals = mgmt_df.assign(plot_val=lambda t: t["value"])
    y_title_bars = "Shared space [m³]"
    total_label = f"Total crown: {total_ref_vol:.0f} m³"
    shared_label = f"Shared: {total_shared_vol:.0f} m³"

# ---------- TITLES ----------
bubble_title = "Shared vs Total Crown Volume"
spec_title = "Who Competes · by Species"
mgmt_title = "Who Competes · by Management"

# ---------- PLOT ----------
fig = make_subplots(
    rows=1,
    cols=3,
    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
    subplot_titles=(bubble_title, spec_title, mgmt_title),
    horizontal_spacing=0.06,
)

# 1) BUBBLE (outer = total crown, inner = shared; dotyk na pravém okraji pro vizuální srovnání)
p_share = (total_shared_vol / total_ref_vol) if total_ref_vol > 0 else 0.0
R = 1.0
r = math.sqrt(max(0.0, min(1.0, p_share))) * R
xg, yg = 0.0, 0.0
xr, yr = (R - r), 0.0

fig.add_shape(
    type="circle",
    xref="x1",
    yref="y1",
    x0=xg - R,
    x1=xg + R,
    y0=yg - R,
    y1=yg + R,
    line=dict(width=0),
    fillcolor="rgba(60, 141, 47, 0.25)",  # light green
)
fig.add_shape(
    type="circle",
    xref="x1",
    yref="y1",
    x0=xr - r,
    x1=xr + r,
    y0=yr - r,
    y1=yr + r,
    line=dict(width=2, color="rgba(255, 40, 40, 1.0)"),
    fillcolor="rgba(255, 40, 40, 0.45)",
)

fig.add_annotation(
    x=xg,
    y=yg + R + 0.15,
    xref="x1",
    yref="y1",
    text=total_label,
    showarrow=False,
    font=dict(size=13),
)
fig.add_annotation(
    x=xr,
    y=yr - r - 0.15,
    xref="x1",
    yref="y1",
    text=shared_label,
    showarrow=False,
    font=dict(size=13),
)

fig.update_xaxes(
    row=1, col=1, visible=False, range=[-1.3, 1.3], scaleanchor="y1", scaleratio=1
)
fig.update_yaxes(row=1, col=1, visible=False, range=[-1.3, 1.3])

# 2) BARS by SPECIES
x_species = (
    spec_plot_vals["species"].tolist()
    if not spec_plot_vals.empty
    else (species_all or [])
)
y_species = (
    spec_plot_vals["plot_val"].tolist()
    if not spec_plot_vals.empty
    else ([0.0] * len(x_species))
)
colors_species = [species_cmap.get(s, "#AAAAAA") for s in x_species]
fig.add_trace(
    go.Bar(
        x=x_species,
        y=y_species,
        marker_color=colors_species,
        hovertemplate="Species: %{x}<br>Value: %{y:.2f}"
        + (" %" if show_mode == "Percentage (%)" else " m³")
        + "<extra></extra>",
        showlegend=False,
    ),
    row=1,
    col=2,
)

# 3) BARS by MANAGEMENT
x_mgmt = (
    mgmt_plot_vals["management_status"].tolist()
    if not mgmt_plot_vals.empty
    else (mgmt_all or [])
)
y_mgmt = (
    mgmt_plot_vals["plot_val"].tolist()
    if not mgmt_plot_vals.empty
    else ([0.0] * len(x_mgmt))
)
colors_mgmt = [mgmt_cmap.get(m, "#AAAAAA") for m in x_mgmt]
fig.add_trace(
    go.Bar(
        x=x_mgmt,
        y=y_mgmt,
        marker_color=colors_mgmt,
        hovertemplate="Management: %{x}<br>Value: %{y:.2f}"
        + (" %" if show_mode == "Percentage (%)" else " m³")
        + "<extra></extra>",
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
        title_text=(
            "Share of shared space [%]"
            if show_mode == "Percentage (%)"
            else "Shared space [m³]"
        ),
        range=[0, y_upper],
        row=1,
        col=c,
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

# one combined Management filter
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
