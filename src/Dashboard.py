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

# ---------- DATA ----------
def load_plot_info(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)
    pi = data.get("plot_info") or []
    return pd.DataFrame(pi)

#file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v12.json")
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- HLAVIČKA ----------
st.markdown(f"#### **{plot_info['name'].iloc[0]}**")

# ---------- SPOLEČNÉ ----------
CHART_HEIGHT = 360
Before = "Original Stand"
After = "Managed Stand"
Removed = "Removed from Stand"

df = df.copy()
df["volume"] = df["Volume_m3"]
if "species" in df.columns:
    df["species"] = df["species"].astype(str)
if "speciesColorHex" in df.columns:
    df["speciesColorHex"] = df["speciesColorHex"].astype(str)

def _make_masks(d: pd.DataFrame):
    keep_status = {"Target tree", "Untouched"}
    mask_after   = d.get("management_status", pd.Series(False, index=d.index)).isin(keep_status)
    mask_removed = ~mask_after if "management_status" in d.columns else pd.Series(False, index=d.index)
    mask_before  = pd.Series(True, index=d.index)  # vše
    return {Before: mask_before, After: mask_after, Removed: mask_removed}

def _species_colors(d: pd.DataFrame) -> dict:
    if "species" not in d.columns or "speciesColorHex" not in d.columns:
        return {}
    return (d.assign(species=lambda x: x["species"].astype(str))
              .groupby("species")["speciesColorHex"].first().to_dict())

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

# ---------- OVLÁDÁNÍ (nahrazeno: Before/After/Removed) ----------
c_left, c_mid, c_right = st.columns([3, 4, 3])

with c_mid:
    dist_mode = st.segmented_control("", options=[Before, After, Removed], default=Before)


masks = _make_masks(df)
mask = masks.get(dist_mode, pd.Series(True, index=df.index))
df_sel = df[mask].copy()


# ---------- 1) PIE – Species composition ----------
# ---------- SPOLEČNÁ FIGURA: 1 řádek, 3 panely, jedna legenda dole ----------
def render_three_panel_with_shared_legend(df_all: pd.DataFrame, df_sub: pd.DataFrame, by_volume: bool):
    # --- Příprava barev/species
    color_map = _species_colors(df_all)
    species_all = sorted(df_all["species"].astype(str).unique().tolist())

    # --- PIE data (species -> value)
    d = df_sub.copy()
    d["species"] = d["species"].astype(str).str.strip()
    if by_volume:
        if "volume" not in d.columns:
            st.warning("Chybí sloupec 'volume' – nelze spočítat koláč podle objemu.")
            return
        pie_agg = (d.groupby("species", as_index=False)
                     .agg(value=("volume", "sum")))
        pie_value_label = "Volume (m³)"
    else:
        pie_agg = (d.groupby("species", as_index=False)
                     .agg(value=("species", "size")))
        pie_value_label = "Trees"
    pie_agg = pie_agg.sort_values("value", ascending=False)

    # --- DBH biny (podle celé populace pro stabilní osu)
    dbh_bins, dbh_labels = _make_bins_labels(df_all, "dbh", 10, "cm")
    if dbh_bins is None:
        dbh_bins, dbh_labels = np.array([0, 10]), ["0–10 cm"]
    dbh_y_upper = _y_upper_for(df_all, "dbh", dbh_bins, dbh_labels)

    # --- HEIGHT biny
    if "height" in df_all.columns:
        h_bins, h_labels = _make_bins_labels(df_all, "height", 5, "m")
        if h_bins is None:
            h_bins, h_labels = np.array([0, 5]), ["0–5 m"]
        h_y_upper = _y_upper_for(df_all, "height", h_bins, h_labels)
    else:
        h_bins, h_labels, h_y_upper = np.array([0, 5]), ["0–5 m"], 10

    # --- Převody na long pro DBH / HEIGHT (jen pro vybraný subset df_sub)
    def long_binned(df_in: pd.DataFrame, value_col: str, bins: np.ndarray, labels: list[str]) -> pd.DataFrame:
        t = df_in.copy()
        t["species"] = t["species"].astype(str)
        cats = pd.Categorical(pd.cut(pd.to_numeric(t[value_col], errors="coerce"),
                                     bins=bins, labels=labels,
                                     include_lowest=True, right=False, ordered=True),
                              categories=labels, ordered=True)
        t = t.assign(bin=cats).dropna(subset=["bin"])
        pv = t.pivot_table(index="bin", columns="species", aggfunc="size", fill_value=0)
        long = pv.stack().rename("count").reset_index()
        long["bin"] = long["bin"].astype(str)
        return long

    dbh_long = long_binned(df_sub, "dbh", dbh_bins, dbh_labels)
    height_long = long_binned(df_sub, "height", h_bins, h_labels) if "height" in df_sub.columns else pd.DataFrame(columns=["bin","species","count"])

    # --- Subplots: pie | dbh | height
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Species Composition", "DBH Distribution", "Height Distribution"),
        horizontal_spacing=0.06
    )

    # --- 1) PIE (bez legendy, ať se nezdvojuje)
    fig.add_trace(
        go.Pie(
            labels=pie_agg["species"],
            values=pie_agg["value"],
            hole=0.35,
            marker=dict(colors=[color_map.get(s, "#AAAAAA") for s in pie_agg["species"]]),
            hovertemplate=f"Species: %{{label}}<br>{pie_value_label}: %{{value}}<br>%: %{{percent}}<extra></extra>",
            showlegend=False
        ),
        row=1, col=1
    )

    # --- 2) DBH (stack podle druhu) – JEN TADY zapneme legendu pro každý druh
    for sp in species_all:
        y_vals = (dbh_long[dbh_long["species"] == sp]
                  .set_index("bin").reindex(dbh_labels)["count"].fillna(0).tolist())
        fig.add_trace(
            go.Bar(
                x=dbh_labels, y=y_vals, name=sp,
                marker_color=color_map.get(sp, "#AAAAAA"),
                legendgroup=sp, showlegend=True,
                hovertemplate=f"%{{x}}<br>Species: {sp}<br>Trees: %{{y}}<extra></extra>"
            ),
            row=1, col=2
        )

    # --- 3) HEIGHT (stack) – legendu VYPNEME, ale necháme stejné legendgroup
    if not height_long.empty:
        for sp in species_all:
            y_vals = (height_long[height_long["species"] == sp]
                      .set_index("bin").reindex(h_labels)["count"].fillna(0).tolist())
            fig.add_trace(
                go.Bar(
                    x=h_labels, y=y_vals, name=sp,
                    marker_color=color_map.get(sp, "#AAAAAA"),
                    legendgroup=sp, showlegend=False,
                    hovertemplate=f"%{{x}}<br>Species: {sp}<br>Trees: %{{y}}<extra></extra>"
                ),
                row=1, col=3
            )

    # --- layout: jedna společná legenda dole
    fig.update_layout(
        barmode="stack",
        height=CHART_HEIGHT + 80,
        margin=dict(l=10, r=10, t=60, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5
        )
    )

    # osy
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=dbh_labels, row=1, col=2)
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=h_labels,   row=1, col=3)
    fig.update_yaxes(title_text="Trees", row=1, col=2, tick0=0, dtick=25, range=[0, dbh_y_upper])
    fig.update_yaxes(title_text=None,  row=1, col=3, tick0=0, dtick=25, range=[0, h_y_upper])

    st.plotly_chart(fig, use_container_width=True)


# ---------- UI: volba pro PIE (zůstává pod figurou, můžeš přesunout nad) ----------
# vykreslit figurku (3 panely + 1 legenda)
c_left2, c_right2 = st.columns([1,10])

with c_left2:
    st.markdown("###")
    st.markdown("###")
    pie_metric = st.radio(
        "Percentage by:",
        options=["Volume (m³)", "Tree count"],
        index=0 if st.session_state.get("pie_metric", "Volume (m³)") == "Volume (m³)" else 1,
        horizontal=False,
        key="pie_metric"
    )

with c_right2:
    render_three_panel_with_shared_legend(
        df_all=df,
        df_sub=df_sel,
        by_volume=(st.session_state.get("pie_metric", "Volume (m³)") == "Volume (m³)")
    )



with st.expander("### Plot Details"):
    st.markdown(f"""
- **Forest type:** {plot_info['forest_type'].iloc[0]}
- **Number of trees:** {plot_info['no_trees'].iloc[0]}
- **Wood volume:** {plot_info['volume'].iloc[0]} m³
- **Area:** {plot_info['size_ha'].iloc[0]} ha
- **Altitude:** {plot_info['altitude'].iloc[0]} m
- **Precipitation:** {plot_info['precipitation'].iloc[0]} mm/year
- **Average temperature:** {plot_info['temperature'].iloc[0]} °C
- **Established:** {plot_info['established'].iloc[0]}
- **Location:** {plot_info['state'].iloc[0]}
- **Owner:** {plot_info['owner'].iloc[0]}
""")
