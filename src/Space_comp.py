# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Streamlit: Competition for Space (Shared Crown Voxels)
# Visualizes:
#   1) Bubble: Shared vs Total crown volume
#   2) Bars by Species: who competes (sum of shared crown volumes)
#   3) Bars by Management: who competes
#
# Filters:
#   • Stand State (Before / After / Removed)
#   • Show as Percentage or Volume
#   • DBH filter
#   • Height filter
#   • Species filter (multi)
#   • Management filter (multi)
#
# SPECIAL LOGIC:
#   Trees filtered by DBH/Height remain included if they share crown space
#   with any tree that *did* pass the filters (neighbors must be preserved).
# -------------------------------------------------------------------

import json
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import src.io_utils as iou

# -------------------------------------------------------------------
# SESSION STATE: stand-state definitions
# -------------------------------------------------------------------
Before  = st.session_state.Before
After   = st.session_state.After
Removed = st.session_state.Removed
stand_stat_all = [Before, After]

st.markdown("### **Competition for Space - Shared Volume of Crowns**")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/"
        "PokojnaHora_3df/PokojnaHora.json"
    )
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# -------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------

def _to_list_of_dicts(v):
    """Safely parse the crownVoxelCountShared field into a list of dicts."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict) and "treeId" in x]
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict) and "treeId" in x]
        except:
            return []
    return []

def _species_colors(d):
    """Map species → speciesColorHex."""
    if "species" not in d.columns or "speciesColorHex" not in d.columns:
        return {}
    return (
        d.assign(species=lambda x: x["species"].astype(str))
        .groupby("species")["speciesColorHex"]
        .first()
        .to_dict()
    )

def _management_colors(d):
    """Map management_status → managementColorHex."""
    if "management_status" not in d.columns or "managementColorHex" not in d.columns:
        return {}
    cmap = (
        d.assign(ms=lambda x: x["management_status"].astype(str))
        .groupby("ms")["managementColorHex"]
        .first()
        .to_dict()
    )
    return {k: (v if isinstance(v, str) and v.strip() else "#AAAAAA") for k, v in cmap.items()}

def _crown_volume_series(d):
    """Compute total crown volume from voxel count and voxel size."""
    cnt = pd.to_numeric(d.get("crownVoxelCount", 0), errors="coerce").fillna(0)
    vsz = pd.to_numeric(d.get("crownVoxelSize", 0), errors="coerce").fillna(0)
    return (cnt * (vsz ** 3)).astype(float)

def _shared_volume_for_tree_row(row):
    """Compute total shared volume for a tree from shared voxel list."""
    vsz = pd.to_numeric(row.get("crownVoxelSize"), errors="coerce")
    if not np.isfinite(vsz) or vsz <= 0:
        return 0.0
    voxel_vol = float(vsz) ** 3
    items = _to_list_of_dicts(row.get("crownVoxelCountShared"))
    total_shared_vox = float(sum(max(0, int(it.get("count", 0))) for it in items))
    return total_shared_vox * voxel_vol

def _make_masks(d):
    """Return masks for stand-state logic: Before / After / Removed."""
    keep = {"Target tree", "Untouched"}
    mask_after = d["management_status"].astype(str).isin(keep)
    mask_removed = ~mask_after
    mask_before = pd.Series(True, index=d.index)
    return {Before: mask_before, After: mask_after, Removed: mask_removed}

# -------------------------------------------------------------------
# STATIC SELECTION LISTS
# -------------------------------------------------------------------
sp_all = iou._unique_sorted(df.get("species", pd.Series(dtype=object)))
mg_all = iou._unique_sorted(df.get("management_status", pd.Series(dtype=object)))

# -------------------------------------------------------------------
# TOP FILTERS (stand state, show mode, DBH, Height)
# -------------------------------------------------------------------
c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([0.5,3,0.5,2,0.5,2,0.5,2,0.5])

with c2:
    stand_state = st.segmented_control(
        "**Select Stand State:**", stand_stat_all,
        default=Before, width="stretch"
    )

with c4:
    show_mode = st.segmented_control(
        "**Show Values as:**",
        ["Percentage (%)", "Volume (m³)"],
        default="Percentage (%)",
        width="stretch"
    )

# --- DBH slider ---
with c6:
    dbh_vals = df["dbh"].dropna()
    dbh_min, dbh_max = float(dbh_vals.min()), float(dbh_vals.max())
    dbh_min = math.floor(dbh_min)
    dbh_max = math.ceil(dbh_max)
    dbh_range = st.slider("**DBH filter (cm):**", dbh_min, dbh_max, (dbh_min, dbh_max))

# --- Height slider ---
with c8:
    h_vals = df["height"].dropna()
    h_min, h_max = float(h_vals.min()), float(h_vals.max())
    h_min = math.floor(h_min)
    h_max = math.ceil(h_max)
    height_range = st.slider("**Height filter (m):**", h_min, h_max, (h_min, h_max))

# -------------------------------------------------------------------
# READ SPECIES & MANAGEMENT SELECTIONS FROM SESSION
# -------------------------------------------------------------------
species_sel = st.session_state.get("species_sel", sp_all)
mg_sel = st.session_state.get("mgmt_sel", mg_all)

species_sel = list(species_sel)
mg_sel_set = set(mg_sel)

# -------------------------------------------------------------------
# DERIVED COLUMNS
# -------------------------------------------------------------------
df["crown_volume_ref"] = _crown_volume_series(df)

neighbor_attrs = df[["id", "species", "management_status",
                     "speciesColorHex", "managementColorHex"]].copy()
neighbor_attrs["id"] = pd.to_numeric(neighbor_attrs["id"], errors="coerce").astype("Int64")

# -------------------------------------------------------------------
# BUILD FOCAL SUBSET (with DBH/Height filtering + neighbor preservation)
# -------------------------------------------------------------------
state_masks = _make_masks(df)

mask_dbh = (df["dbh"] >= dbh_range[0]) & (df["dbh"] <= dbh_range[1])
mask_h   = (df["height"] >= height_range[0]) & (df["height"] <= height_range[1])

mask_species = df["species"].astype(str).isin(species_sel)
mask_mgmt    = df["management_status"].astype(str).isin(mg_sel_set)
mask_state   = state_masks[stand_state]

# Trees that directly pass filters
focal_mask_base = mask_species & mask_mgmt & mask_state & mask_dbh & mask_h
focal_ids_base = set(df.loc[focal_mask_base, "id"].dropna().astype(int).tolist())

# Collect neighbors directly (without df_shared)
neighbor_ids = set()
for lst in df.loc[focal_mask_base, "crownVoxelCountShared"].apply(_to_list_of_dicts):
    for item in lst:
        tid = item.get("treeId")
        if tid is not None:
            neighbor_ids.add(int(tid))

# Final set of trees to keep
all_kept_ids = focal_ids_base.union(neighbor_ids)

# Final focal subset
focal_mask = df["id"].astype(int).isin(all_kept_ids)
df_focal = df.loc[focal_mask].copy()

# Compute baseline shared volume for focal trees
df_focal["shared_volume_base"] = df_focal.apply(_shared_volume_for_tree_row, axis=1)

# -------------------------------------------------------------------
# EXPLODE SHARED LIST → df_shared
# -------------------------------------------------------------------
df_focal["_shared_list"] = df_focal["crownVoxelCountShared"].apply(_to_list_of_dicts)

df_shared = (
    df_focal[["id", "crownVoxelSize", "_shared_list"]]
    .rename(columns={"id": "focal_id"})
    .explode("_shared_list", ignore_index=True)
)

df_shared = df_shared.dropna(subset=["_shared_list"])

df_shared["neighbor_id"] = (
    pd.to_numeric(df_shared["_shared_list"].apply(lambda d: d.get("treeId")), errors="coerce")
    .astype("Int64")
)
df_shared["count"] = (
    pd.to_numeric(
        df_shared["_shared_list"].apply(lambda d: d.get("count")), errors="coerce"
    )
    .fillna(0).astype(int).clip(lower=0)
)
df_shared["voxel_vol"] = pd.to_numeric(df_shared["crownVoxelSize"], errors="coerce").pow(3)
df_shared["shared_volume"] = (df_shared["count"] * df_shared["voxel_vol"]).fillna(0)

# Attach neighbor attributes
df_shared = df_shared.merge(
    neighbor_attrs,
    left_on="neighbor_id",
    right_on="id",
    how="left",
    suffixes=("", "_nbr"),
)

# -------------------------------------------------------------------
# REMOVED TREES HANDLING (adjust shared volume)
# -------------------------------------------------------------------
removed_ids = set(
    pd.to_numeric(df.loc[state_masks[Removed], "id"], errors="coerce")
    .dropna().astype(int).tolist()
)

df_shared["neighbor_removed"] = df_shared["neighbor_id"].isin(removed_ids)

removed_shared_by_focal = (
    df_shared[df_shared["neighbor_removed"]]
    .groupby("focal_id")["shared_volume"]
    .sum()
    .rename("removed_shared")
)

df_focal = df_focal.merge(removed_shared_by_focal, left_on="id", right_index=True, how="left")
df_focal["removed_shared"] = df_focal["removed_shared"].fillna(0)

# Adjust shared volume depending on stand state
if stand_state in (After, Removed):
    df_focal["shared_volume_adj"] = (
        df_focal["shared_volume_base"] - df_focal["removed_shared"]
    ).clip(lower=0)
else:
    df_focal["shared_volume_adj"] = df_focal["shared_volume_base"]

df_focal["free_volume_adj"] = (
    df_focal["crown_volume_ref"] - df_focal["shared_volume_adj"]
).clip(lower=0)

# -------------------------------------------------------------------
# AGGREGATE FOR BAR CHARTS
# -------------------------------------------------------------------
df_shared_kept = (
    df_shared[~df_shared["neighbor_removed"]] if stand_state in (After, Removed)
    else df_shared.copy()
)

spec_df = (df_shared_kept.groupby("species", as_index=False)["shared_volume"].sum()
           if not df_shared_kept.empty else
           pd.DataFrame(columns=["species", "value"])
          ).rename(columns={"shared_volume": "value"})

mgmt_df = (df_shared_kept.groupby("management_status", as_index=False)["shared_volume"].sum()
           if not df_shared_kept.empty else
           pd.DataFrame(columns=["management_status", "value"])
          ).rename(columns={"shared_volume": "value"})

# Reindex to include missing categories
if sp_all:
    spec_df = spec_df.set_index("species").reindex(sp_all, fill_value=0).reset_index()

if mg_all:
    mgmt_df = mgmt_df.set_index("management_status").reindex(mg_all, fill_value=0).reset_index()

# Color maps
species_cmap = _species_colors(df)
mgmt_cmap    = _management_colors(df)

# -------------------------------------------------------------------
# TOTALS FOR BUBBLE
# -------------------------------------------------------------------
total_ref_vol = float(df_focal["crown_volume_ref"].sum())
total_shared_vol = float(df_focal["shared_volume_adj"].sum())
non_shared_vol = max(0, total_ref_vol - total_shared_vol)

# -------------------------------------------------------------------
# NORMALIZATION FOR PLOTS
# -------------------------------------------------------------------
if show_mode == "Percentage (%)":
    denom = total_shared_vol if total_shared_vol > 0 else 1
    spec_plot_vals = spec_df.assign(plot_val=lambda t: (t["value"] / denom) * 100)
    mgmt_plot_vals = mgmt_df.assign(plot_val=lambda t: (t["value"] / denom) * 100)
    y_title_bars = "Share of shared space [%]"
    total_label = "Total crown: 100 %"
    shared_label = f"Shared: {(total_shared_vol / total_ref_vol * 100 if total_ref_vol else 0):.0f} %"
else:
    spec_plot_vals = spec_df.assign(plot_val=lambda t: t["value"])
    mgmt_plot_vals = mgmt_df.assign(plot_val=lambda t: t["value"])
    y_title_bars = "Shared space [m³]"
    total_label = f"Total crown: {total_ref_vol:.0f} m³"
    shared_label = f"Shared: {total_shared_vol:.0f} m³"

# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]],
    subplot_titles=("Shared vs Total Crown Volume", "Who Competes · by Species", "Who Competes · by Management"),
    horizontal_spacing=0.06,
)

# 1) BUBBLE DIAGRAM
p_share = (total_shared_vol / total_ref_vol) if total_ref_vol > 0 else 0.0
R = 1.0
r = math.sqrt(max(0, min(1, p_share))) * R
xg, yg = 0, 0
xr, yr = (R - r), 0

fig.add_shape(
    type="circle",
    xref="x1", yref="y1",
    x0=xg - R, x1=xg + R, y0=yg - R, y1=yg + R,
    fillcolor="rgba(60,141,47,0.25)", line=dict(width=0)
)
fig.add_shape(
    type="circle",
    xref="x1", yref="y1",
    x0=xr - r, x1=xr + r, y0=yr - r, y1=yr + r,
    fillcolor="rgba(255,40,40,0.45)",
    line=dict(width=2, color="rgba(255,40,40,1)")
)

fig.add_annotation(x=xg, y=yg + R + 0.15, xref="x1", yref="y1",
                   text=total_label, showarrow=False)
fig.add_annotation(x=xr, y=yr - r - 0.15, xref="x1", yref="y1",
                   text=shared_label, showarrow=False)

fig.update_xaxes(visible=False, range=[-1.3, 1.3], row=1, col=1, scaleanchor="y1")
fig.update_yaxes(visible=False, range=[-1.3, 1.3], row=1, col=1)

# 2) BAR CHART – SPECIES
fig.add_trace(
    go.Bar(
        x=spec_plot_vals["species"],
        y=spec_plot_vals["plot_val"],
        marker_color=[species_cmap.get(s, "#AAAAAA") for s in spec_plot_vals["species"]],
        hovertemplate="Species: %{x}<br>Value: %{y:.2f}" +
                       (" %" if show_mode == "Percentage (%)" else " m³") +
                       "<extra></extra>",
        showlegend=False,
    ),
    row=1, col=2
)

# 3) BAR CHART – MANAGEMENT
fig.add_trace(
    go.Bar(
        x=mgmt_plot_vals["management_status"],
        y=mgmt_plot_vals["plot_val"],
        marker_color=[mgmt_cmap.get(m, "#AAAAAA") for m in mgmt_plot_vals["management_status"]],
        hovertemplate="Management: %{x}<br>Value: %{y:.2f}" +
                       (" %" if show_mode == "Percentage (%)" else " m³") +
                       "<extra></extra>",
        showlegend=False,
    ),
    row=1, col=3
)

# Axis formatting
for c in (2, 3):
    fig.update_xaxes(tickangle=45, row=1, col=c)
    vals = spec_plot_vals["plot_val"] if c == 2 else mgmt_plot_vals["plot_val"]
    ymax = float(max(vals)) if len(vals) else 1
    if ymax <= 0:
        y_upper = 1
    else:
        magnitude = 10 ** int(np.floor(np.log10(ymax)))
        step = magnitude / 2
        y_upper = math.ceil(ymax / step) * step
    fig.update_yaxes(title_text=y_title_bars, range=[0, y_upper], row=1, col=c)

fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=40))

# -------------------------------------------------------------------
# BOTTOM FILTERS
# -------------------------------------------------------------------
c_bot1, c_bot2 = st.columns([2, 15])

with c_bot1:
    st.pills(
        "**Filter Species:**",
        options=sp_all,
        default=species_sel,
        selection_mode="multi",
        key="species_sel",
    )

with c_bot2:
    st.plotly_chart(fig, use_container_width=True)

c31, c32 = st.columns([2, 15])

with c32:
    st.pills(
        "**Filter Management:**",
        options=mg_all,
        default=mg_sel_set,
        selection_mode="multi",
        key="mgmt_sel",
    )
