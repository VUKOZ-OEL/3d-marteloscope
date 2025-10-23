import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import src.io_utils as iou
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Dict, List, Union
import json

print("summary")
print(st.session_state.trees["speciesColorHex"])

# ---------- DATA ----------
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- HLAVIČKA ----------
st.markdown(f"### Plot summary:")

# ---------- SPOLEČNÉ ----------
CHART_HEIGHT = 360
Before = st.session_state.Before
After = st.session_state.After
Removed = st.session_state.Removed

colorBySpp = "Species"
colorByMgmt = "Management"

df = df.copy()

# bezpečné vytvoření 'volume'
if "Volume_m3" in df.columns:
    df["volume"] = pd.to_numeric(df["Volume_m3"], errors="coerce")
else:
    df["volume"] = np.nan

# dopočet bazální plochy [m²] z DBH [cm]  -> BA = π * (dbh_cm / 200)^2
if "dbh" in df.columns:
    dbh_cm = pd.to_numeric(df["dbh"], errors="coerce")
    df["basal_area_m2"] = np.pi * (dbh_cm / 200.0) ** 2
else:
    df["basal_area_m2"] = np.nan

# standardizace typů
if "species" in df.columns:
    df["species"] = df["species"].astype(str)
if "speciesColorHex" in df.columns:
    df["speciesColorHex"] = df["speciesColorHex"].astype(str)
if "management_status" in df.columns:
    df["management_status"] = df["management_status"].astype(str)
if "managementColorHex" in df.columns:
    df["managementColorHex"] = df["managementColorHex"].astype(str)

# plocha (ha) pro přepočet na hektar
try:
    area_ha = float(plot_info['size_ha'].iloc[0])
    if not np.isfinite(area_ha) or area_ha <= 0:
        area_ha = 1.0
except Exception:
    area_ha = 1.0

def _make_masks(d: pd.DataFrame):
    keep_status = {"Target tree", "Untouched"}
    mask_after = d.get("management_status", pd.Series(False, index=d.index)).isin(keep_status)
    mask_removed = ~mask_after if "management_status" in d.columns else pd.Series(False, index=d.index)
    mask_before = pd.Series(True, index=d.index)  # vše
    return {Before: mask_before, After: mask_after, Removed: mask_removed}

def _species_colors(d: pd.DataFrame) -> dict:
    if "species" not in d.columns or "speciesColorHex" not in d.columns:
        return {}
    return (d.assign(species=lambda x: x["species"].astype(str))
             .groupby("species")["speciesColorHex"].first().to_dict())

def _management_colors(d: pd.DataFrame) -> dict:
    """ Barvy ze sloupce managementColorHex pro všechny management_status,
        chybějící -> šedá. """
    if "management_status" not in d.columns or "managementColorHex" not in d.columns:
        return {}
    t = d.assign(
        management_status=lambda x: x["management_status"].astype(str),
        managementColorHex=lambda x: x["managementColorHex"].astype(str),
    )
    cmap = t.groupby("management_status")["managementColorHex"].first().to_dict()
    for k, v in list(cmap.items()):
        if not isinstance(v, str) or not v.strip():
            cmap[k] = "#AAAAAA"
    return cmap

def _make_bins_labels(df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str):
    vals = pd.to_numeric(df_all.get(value_col), errors="coerce").dropna()
    if vals.empty:
        return None, None
    vmin = float(np.floor(vals.min() / bin_size) * bin_size)
    vmax = float(np.ceil (vals.max() / bin_size) * bin_size)
    if vmax <= vmin:
        vmax = vmin + bin_size
    bins = np.arange(vmin, vmax + bin_size, bin_size, dtype=float)
    labels = [f"{int(b)}–{int(b + bin_size)} {unit_label}" for b in bins[:-1]]
    return bins, labels

def _y_upper_for(df_all: pd.DataFrame, value_col: str, bins: np.ndarray, labels: list[str]) -> int:
    vals = pd.to_numeric(df_all.get(value_col), errors="coerce")
    cats = pd.cut(vals, bins=bins, labels=labels, include_lowest=True, right=False, ordered=True)
    vc = cats.value_counts()
    if vc.empty:
        return 10
    return int(max(10, np.ceil(vc.max() / 10.0) * 10))

# ---------- OVLÁDÁNÍ ----------
c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5,4,0.25,2,0.25,2,1])

with c2:
    dist_mode = st.segmented_control(
        "**Show Data for:**",
        options=[Before, After, Removed],
        default=Before,
        width="stretch",
    )

with c4:
    sum_metric_label = st.segmented_control(
        "**Sum values by:**",
        options=["Tree count", "Volume (m³)", "Basal area (m²)"],
        default="Tree count",
        width="stretch",
    )

with c6:
    color_mode = st.segmented_control(
        "**Color by:**",
        options=[colorBySpp, colorByMgmt],
        default=colorBySpp,
        width="stretch",
    )

def _metric_meta(label: str):
    """Vrátí (value_col | None pro počty, y_title, unit_suffix, pie_value_label)."""
    if label == "Tree count":
        return None, "Trees", "trees/ha", "Trees"
    if label == "Volume (m³)":
        return "volume", "Volume (m³)", "m³/ha", "Volume (m³)"
    if label == "Basal area (m²)":
        return "basal_area_m2", "Basal area (m²)", "m²/ha", "Basal area (m²)"
    # fallback
    return None, "Trees", "trees/ha", "Trees"

value_col, y_title, unit_suffix, pie_value_label = _metric_meta(sum_metric_label)

masks = _make_masks(df)
mask = masks.get(dist_mode, pd.Series(True, index=df.index))
df_sel = df[mask].copy()

# ---------- 1) PIE + DBH + HEIGHT (sdílená legenda) ----------
def render_three_panel_with_shared_legend(df_all: pd.DataFrame, df_sub: pd.DataFrame,
                                          value_col: str | None, color_mode: str,
                                          y_title: str, unit_suffix: str, pie_value_label: str):

    # --- Nastavení hue / kategorií / barev
    colorBySpp = "Species"
    colorByMgmt = "Management"
    if color_mode == colorByMgmt:
        hue_col = "management_status"
        if hue_col in df_all.columns:
            categories = pd.Index(df_all[hue_col].astype(str).dropna().unique()).tolist()
        else:
            categories = []
        color_map = _management_colors(df_all)
        color_map = {c: color_map.get(c, "#AAAAAA") for c in categories}
    else:
        hue_col = "species"
        categories = sorted(df_all.get("species", pd.Series([], dtype=str)).astype(str).dropna().unique().tolist())
        color_map = _species_colors(df_all)
        color_map = {c: color_map.get(c, "#AAAAAA") for c in categories}

    d = df_sub.copy()
    if hue_col not in d.columns:
        st.warning(f"Chybí sloupec '{hue_col}' – nelze vykreslit grafy.")
        return
    d[hue_col] = d[hue_col].astype(str).str.strip()

    # --- PIE agregace + celkový součet pro střed
    if value_col is None:
        pie_agg = d.groupby(hue_col, as_index=False).agg(value=(hue_col, "size"))
        total_raw  = int(pie_agg["value"].sum())
        unit_disp  = "trees"
        total_text = f"Σ =<br><b>{total_raw}</b><br>{unit_disp}"
        hover_value_token = "%{value:.0f}"
    else:
        if value_col not in d.columns:
            st.warning(f"Chybí sloupec '{value_col}' – nelze spočítat grafy pro vybranou metriku.")
            return
        pie_agg = d.groupby(hue_col, as_index=False).agg(value=(value_col, "sum"))
        total_raw  = float(pie_agg["value"].sum())
        unit_disp  = "m³" if value_col == "volume" else ("m²" if value_col == "basal_area_m2" else "")
        total_text = f"Σ =<br><b>{total_raw:,.1f}</b><br>{unit_disp}".replace(",", " ")
        hover_value_token = "%{value:.1f}"

    # respektuj pořadí kategorií a přidej 0 pro chybějící
    if categories:
        pie_agg = (pie_agg.set_index(hue_col).reindex(categories).fillna(0).reset_index())
    else:
        categories = pie_agg[hue_col].tolist()

    # --- DBH / HEIGHT biny
    dbh_bins, dbh_labels = _make_bins_labels(df_all, "dbh", 10, "cm")
    if dbh_bins is None:
        dbh_bins, dbh_labels = np.array([0, 10]), ["0–10 cm"]
    if "height" in df_all.columns:
        h_bins, h_labels = _make_bins_labels(df_all, "height", 5, "m")
        if h_bins is None:
            h_bins, h_labels = np.array([0, 5]), ["0–5 m"]
    else:
        h_bins, h_labels = np.array([0, 5]), ["0–5 m"]

    # --- Long tabulky
    def long_binned(df_in: pd.DataFrame, base_col: str, bins: np.ndarray, labels: list[str],
                    hue: str, value_col: str | None) -> pd.DataFrame:
        t = df_in.copy()
        t[hue] = t[hue].astype(str)
        vals = pd.to_numeric(t[base_col], errors="coerce")
        cats = pd.Categorical(pd.cut(vals, bins=bins, labels=labels, include_lowest=True, right=False, ordered=True),
                              categories=labels, ordered=True)
        t = t.assign(bin=cats).dropna(subset=["bin"])
        if t.empty:
            return pd.DataFrame(columns=["bin", hue, "value"])
        if value_col is None:
            pv = t.pivot_table(index="bin", columns=hue, aggfunc="size", fill_value=0)
        else:
            t["weight"] = pd.to_numeric(t[value_col], errors="coerce").fillna(0.0)
            pv = t.pivot_table(index="bin", columns=hue, values="weight", aggfunc="sum", fill_value=0.0)
        long = pv.stack().rename("value").reset_index()
        long["bin"] = long["bin"].astype(str)
        return long

    dbh_long    = long_binned(df_sub, "dbh",    dbh_bins, dbh_labels, hue_col, value_col)
    height_long = long_binned(df_sub, "height", h_bins,   h_labels,   hue_col, value_col) if "height" in df_sub.columns else pd.DataFrame(columns=["bin", hue_col, "value"])

    # --- Y-osa: horní hranice
    def upper_from_long(long_df: pd.DataFrame, labels: list[str]) -> float:
        if long_df.empty:
            return 10
        totals = (long_df.groupby("bin")["value"].sum().reindex(labels).fillna(0))
        maxv = float(totals.max())
        if maxv <= 0:
            return 10
        magnitude = 10 ** int(np.floor(np.log10(maxv)))
        step = magnitude / 2
        upper = math.ceil(maxv / step) * step
        return upper

    dbh_y_upper = upper_from_long(dbh_long, dbh_labels)
    h_y_upper   = upper_from_long(height_long, h_labels)

    # --- Subplots: pie | dbh | height
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Stand Composition", "In DBH class", "In Height class"),
        horizontal_spacing=0.06
    )

    if hasattr(st.session_state, "plot_title_font"):
        fig.update_layout(annotations=[
            dict(text=ann.text, x=ann.x, y=ann.y, xref=ann.xref, yref=ann.yref,
                 showarrow=False, font=st.session_state.plot_title_font)
            for ann in fig.layout.annotations
        ])

    # --- 1) PIE (hole=0.50) + filtrace nulových kategorií
    pie_plot = pie_agg[pie_agg["value"] > 0].copy()
    no_data = pie_plot.empty

    if no_data:
        pie_labels = ["No data"]
        pie_values = [1]
        pie_colors = ["#EEEEEE"]
        hovertemplate = ""
        textinfo = "none"
        texttemplate = None
    else:
        pie_labels = pie_plot[hue_col]   # pro hover
        pie_values = pie_plot["value"]
        pie_colors = [color_map.get(c, "#AAAAAA") for c in pie_plot[hue_col]]
        # text na výsečích = vždy procenta
        textinfo = "text"
        texttemplate = "%{percent:.1%}"
        # hover: jméno kategorie + skutečná hodnota + jednotka
        hovertemplate = (
            f"{hue_col}: %{{label}}<br>"
            f"{pie_value_label}: {hover_value_token} {unit_disp}"
            "<extra></extra>"
        )

    fig.add_trace(
        go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.50,
            marker=dict(colors=pie_colors),
            textinfo=textinfo,
            texttemplate=texttemplate,
            textposition="inside",
            insidetextorientation="radial",
            textfont=dict(size=11),
            hovertemplate=hovertemplate,
            showlegend=False,
            sort=False
        ),
        row=1, col=1
    )

    # --- středový Σ = ...
    pie_trace = fig.data[-1]
    if hasattr(pie_trace, "domain") and pie_trace.domain:
        cx = (pie_trace.domain.x[0] + pie_trace.domain.x[1]) / 2
        cy = (pie_trace.domain.y[0] + pie_trace.domain.y[1]) / 2
    else:
        cx, cy = 0.17, 0.5  # fallback

    fig.add_annotation(
        x=cx, y=cy, xref="paper", yref="paper",
        text=(total_text if not no_data else "Σ = 0 " + unit_disp),
        showarrow=False,
        font=dict(size=26, color="black"),  # ← uprav velikost středového textu tady
        xanchor="center", yanchor="middle"
    )

    # --- 2) DBH (stack podle hue)
    for cat in categories:
        y_vals = (dbh_long[dbh_long[hue_col] == cat]
                  .set_index("bin").reindex(dbh_labels)["value"].fillna(0).tolist())
        fig.add_trace(
            go.Bar(
                x=dbh_labels, y=y_vals, name=cat,
                marker_color=color_map.get(cat, "#AAAAAA"),
                legendgroup=cat, showlegend=True,
                hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y}}<extra></extra>"
            ),
            row=1, col=2
        )

    # --- 3) HEIGHT (stack)
    if not height_long.empty:
        for cat in categories:
            y_vals = (height_long[height_long[hue_col] == cat]
                      .set_index("bin").reindex(h_labels)["value"].fillna(0).tolist())
            fig.add_trace(
                go.Bar(
                    x=h_labels, y=y_vals, name=cat,
                    marker_color=color_map.get(cat, "#AAAAAA"),
                    legendgroup=cat, showlegend=False,
                    hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y}}<extra></extra>"
                ),
                row=1, col=3
            )

    # --- layout
    fig.update_layout(
        barmode="stack",
        height=CHART_HEIGHT + 80,
        margin=dict(l=10, r=10, t=60, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # --- Osy
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array",
                     categoryarray=dbh_labels, row=1, col=2)
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array",
                     categoryarray=h_labels, row=1, col=3)

    if value_col is None:
        fig.update_yaxes(title_text=y_title, row=1, col=2, tick0=0, dtick=25, range=[0, dbh_y_upper])
        fig.update_yaxes(title_text=None,   row=1, col=3, tick0=0, dtick=25, range=[0, h_y_upper])
    else:
        fig.update_yaxes(title_text=y_title, row=1, col=2, range=[0, dbh_y_upper])
        fig.update_yaxes(title_text=None,   row=1, col=3, range=[0, h_y_upper])

    st.plotly_chart(fig, use_container_width=True)

# RENDER PLOTS
render_three_panel_with_shared_legend(
    df_all=df,
    df_sub=df_sel,
    value_col=value_col,
    color_mode=color_mode,
    y_title=y_title,
    unit_suffix=unit_suffix,
    pie_value_label=pie_value_label,
)
