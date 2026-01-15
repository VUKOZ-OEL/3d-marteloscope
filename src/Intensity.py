# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Intensity of Silvicultural Intervention — i18n version
# - Uses existing keys from I18N only
# - Before/After/Removed are stored as keys in session_state
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import src.io_utils as iou
import math

from src.i18n import t


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
# NOTE: pro testování nechávám fallback load, ale v produkci budeš mít data už v session_state z app.py
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
    st.session_state.trees = iou.load_project_json(file_path)

df0 = st.session_state.trees.copy()
df0["species"] = df0["species"].astype(str)
df0["management_status"] = df0["management_status"].astype(str)

# numeric fix
for col in ["dbh", "height", "Volume_m3", "BA_m2", "crown_volume"]:
    if col in df0:
        df0[col] = pd.to_numeric(df0[col], errors="coerce")

# basal area if missing
if "BA_m2" not in df0:
    df0["BA_m2"] = np.pi * (df0["dbh"] / 200.0) ** 2

# crown volume placeholder if missing
if "crown_volume" not in df0:
    df0["crown_volume"] = np.nan


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown(f"### **{t('intensity_header')}**")

c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns([0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5])

# --- options jako keys ---
METRIC_TREE_COUNT = "metric_tree_count"
METRIC_VOLUME_M3 = "metric_volume_m3"
METRIC_BASAL_AREA_M2 = "metric_basal_area_m2"
CROWN_VOLUME_M3 = "crown_volume_m3"


PLOT_BY_SPECIES = "species"
PLOT_BY_MANAGEMENT = "management_label"  # v I18N není "Cutting purpose", použijeme Management

with c2:
    metric_id = st.selectbox(
        f"**{t('intensity_based_on')}**",
        options=[METRIC_TREE_COUNT, METRIC_VOLUME_M3, METRIC_BASAL_AREA_M2, CROWN_VOLUME_M3],
        format_func=lambda k: t(k),
    )

with c4:
    group_by_id = st.segmented_control(
        f"**{t('plot_by')}**",
        options=[PLOT_BY_SPECIES, PLOT_BY_MANAGEMENT],
        format_func=lambda k: t(k),
        default=PLOT_BY_SPECIES,
        width="stretch",
    )

with c6:
    dbh_vals = df0["dbh"].dropna()
    dbh_min, dbh_max = float(dbh_vals.min()), float(dbh_vals.max())
    dbh_min = math.floor(dbh_min)
    dbh_max = math.ceil(dbh_max)
    dbh_range = st.slider(f"**{t('dbh_filter')}**", dbh_min, dbh_max, (dbh_min, dbh_max))

with c8:
    h_vals = df0["height"].dropna()
    h_min, h_max = float(h_vals.min()), float(h_vals.max())
    h_min = math.floor(h_min)
    h_max = math.ceil(h_max)
    height_range = st.slider(f"**{t('height_filter')}**", h_min, h_max, (h_min, h_max))


# ------------------------------------------------------------
# FILTER
# ------------------------------------------------------------
df = df0.copy()
df = df[(df["dbh"] >= dbh_range[0]) & (df["dbh"] <= dbh_range[1])]
df = df[(df["height"] >= height_range[0]) & (df["height"] <= height_range[1])]


# ------------------------------------------------------------
# METRIC COLUMN
# ------------------------------------------------------------
if metric_id == METRIC_TREE_COUNT:
    df["metric_value"] = 1.0
elif metric_id == METRIC_VOLUME_M3:
    df["metric_value"] = df["Volume_m3"].fillna(0.0)
elif metric_id == METRIC_BASAL_AREA_M2:
    df["metric_value"] = df["BA_m2"].fillna(0.0)
else:  # Crown Volume
    df["metric_value"] = df["crown_volume"].fillna(0.0)


# ------------------------------------------------------------
# REMOVAL MASK
# ------------------------------------------------------------
keep_status = {"Target tree", "Untouched"}
df["is_removed"] = ~df["management_status"].isin(keep_status)


# ------------------------------------------------------------
# GROUPING LOGIC
# ------------------------------------------------------------
if group_by_id == PLOT_BY_SPECIES:
    main_group = "species"
    stack_group = "management_status"
    color_col = "managementColorHex"
else:  # Management
    main_group = "management_status"
    stack_group = "species"
    color_col = "speciesColorHex"

if color_col not in df.columns:
    df[color_col] = "#AAAAAA"
else:
    df[color_col] = df[color_col].fillna("#AAAAAA").astype(str)


# ------------------------------------------------------------
# PIVOTS (FULL)
# ------------------------------------------------------------
pivot_total_all = df.pivot_table(
    index=main_group,
    columns=stack_group,
    values="metric_value",
    aggfunc="sum",
    fill_value=0.0,
)

pivot_removed_all = df[df["is_removed"]].pivot_table(
    index=main_group,
    columns=stack_group,
    values="metric_value",
    aggfunc="sum",
    fill_value=0.0,
)

# hlavní skupiny
if group_by_id == PLOT_BY_SPECIES:
    all_main_groups = sorted(pivot_total_all.index.tolist())
else:
    all_main_groups = sorted([g for g in pivot_total_all.index if g not in keep_status])

# stack skupiny
if group_by_id == PLOT_BY_SPECIES:
    stack_groups = [c for c in pivot_total_all.columns if c not in keep_status and pivot_total_all[c].sum() > 0]
else:
    stack_groups = [c for c in pivot_total_all.columns if pivot_total_all[c].sum() > 0]

# bezpečné pivoty pro vybrané skupiny
pivot_total = pivot_total_all.reindex(index=all_main_groups, columns=stack_groups, fill_value=0.0)
pivot_removed = pivot_removed_all.reindex(index=all_main_groups, columns=stack_groups, fill_value=0.0)


# ------------------------------------------------------------
# PROCENTA
# ------------------------------------------------------------
total_all = float(pivot_total_all.values.sum())
pct_from_total = (pivot_removed / total_all * 100.0) if total_all > 0 else (pivot_removed * 0.0)

# pro inside-group intensity
group_sums = pivot_total_all.reindex(all_main_groups).sum(axis=1)
group_sums_safe = group_sums.replace(0.0, np.nan)
pct_in_group = (pivot_removed.div(group_sums_safe, axis=0) * 100.0).fillna(0.0)


# ------------------------------------------------------------
# TOTAL řádek (Sum) — NOTE: Sum je v I18N jako t('sum_label')
# ------------------------------------------------------------
sum_label = t("sum_label")

if len(stack_groups) and total_all > 0:
    total_row_stack = (
        pivot_removed_all.reindex(index=pivot_total_all.index, columns=stack_groups, fill_value=0.0).sum(axis=0)
        / total_all
        * 100.0
    )
else:
    total_row_stack = pd.Series(0.0, index=stack_groups)

pct_from_total_plot = pct_from_total.copy()
pct_from_total_plot.loc[sum_label] = total_row_stack

summary_intensity = float(pct_from_total_plot.loc[sum_label].sum())
summary_title = t("summary_total_selection_intensity", value=f"{summary_intensity:.1f}")


# ------------------------------------------------------------
# BARVY PRO STACK
# ------------------------------------------------------------
stack_colors = {}
for sg in stack_groups:
    row = df[df[stack_group] == sg]
    stack_colors[sg] = row[color_col].iloc[0] if len(row) else "#777777"


# ------------------------------------------------------------
# LAYOUT – DVA GRAFY
# ------------------------------------------------------------
left, right = st.columns([1, 1])

# ------------------------------------------------------------
# 🔵 GRAF 1 — REMOVAL FROM TOTAL
# ------------------------------------------------------------
with left:
    fig1 = go.Figure()

    for sg in stack_groups:
        cd_rows = []
        for g in pct_from_total_plot.index:
            if g == sum_label:
                removed_val = pivot_removed_all.reindex(columns=[sg], fill_value=0.0)[sg].sum()
                total_val = pivot_total_all.values.sum()
            else:
                removed_val = pivot_removed_all.reindex(index=[g], columns=[sg], fill_value=0.0).iloc[0, 0]
                total_val = pivot_total_all.reindex(index=[g], fill_value=0.0).loc[g, :].sum()
            cd_rows.append([removed_val, total_val])

        custom = np.array(cd_rows)

        fig1.add_trace(
            go.Bar(
                y=pct_from_total_plot.index,
                x=pct_from_total_plot[sg],
                name=str(sg),
                orientation="h",
                marker_color=stack_colors[sg],
                customdata=custom,
                hovertemplate=(
                    f"{t('group_label')}: %{{y}}<br>"
                    f"{t('removed_label')}: %{{customdata[0]:.1f}}<br>"
                    f"{t('total_label')}: %{{customdata[1]:.1f}}<br>"
                    f"{t('percent_label')}: %{{x:.1f}}%<extra></extra>"
                ),
            )
        )

    fig1.update_layout(
        height=400,
        barmode="stack",
        title={"text": summary_title, "x": 0.5, "xanchor": "center"},
        xaxis_title=t("removal_percent_axis_title"),
        yaxis_title="",
        legend_title=t("legend_title"),
    )
    fig1.update_xaxes(range=[0, 100])
    st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------------------
# 🔵 GRAF 2 — INSIDE GROUP INTENSITY
# ------------------------------------------------------------
with right:
    fig2 = go.Figure()

    for sg in stack_groups:
        cd_rows = []
        for g in pct_in_group.index:
            removed_val = pivot_removed_all.reindex(index=[g], columns=[sg], fill_value=0.0).iloc[0, 0]
            total_val = pivot_total_all.reindex(index=[g], fill_value=0.0).loc[g, :].sum()
            cd_rows.append([removed_val, total_val])

        custom = np.array(cd_rows)

        fig2.add_trace(
            go.Bar(
                y=pct_in_group.index,
                x=pct_in_group[sg],
                name=str(sg),
                orientation="h",
                marker_color=stack_colors[sg],
                customdata=custom,
                hovertemplate=(
                    f"{t('group_label')}: %{{y}}<br>"
                    f"{t('removed_label')}: %{{customdata[0]:.1f}}<br>"
                    f"{t('total_label')}: %{{customdata[1]:.1f}}<br>"
                    f"{t('percent_label')}: %{{x:.1f}}%<extra></extra>"
                ),
            )
        )

    fig2.update_layout(
        height=400,
        barmode="stack",
        title={"text": t("intensity_in_group_title"), "x": 0.5, "xanchor": "center"},
        xaxis_title=t("removal_percent_axis_title"),
        yaxis_title="",
        legend_title=t("legend_title"),
    )
    fig2.update_xaxes(range=[0, 100])
    st.plotly_chart(fig2, use_container_width=True)
