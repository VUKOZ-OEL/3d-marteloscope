import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import src.io_utils as iou
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Data ---
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v11.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ========== NASTAVENÍ ==========
CHART_HEIGHT = 350  # výška každého grafu

# Sloupce schované z menu volby Y
exclude_list = {
    "species", "speciesColorHex", "management_status",  # typické
    # případně přidejte další podle potřeby
}

# --- masky After/Removed ---
keep_status = {"Target tree", "Untouched"}
mask_after   = df.get("management_status", pd.Series(False, index=df.index)).isin(keep_status)
mask_removed = ~mask_after if "management_status" in df.columns else pd.Series(False, index=df.index)

# ========== HELPERY SPOLEČNÉ ==========
def _species_order_and_colors(df_all: pd.DataFrame):
    """Pořadí druhů a barvy z 'Before' (celé df)."""
    if "species" not in df_all.columns:
        return [], {}
    order = (df_all["species"].astype(str)
             .value_counts(dropna=False)
             .sort_values(ascending=False)
             .index.astype(str).str.strip().tolist())
    if "speciesColorHex" in df_all.columns:
        col_lookup = (df_all.groupby("species", as_index=False)["speciesColorHex"]
                          .first()
                          .assign(species=lambda d: d["species"].astype(str).str.strip()))
        colors = dict(zip(col_lookup["species"], col_lookup["speciesColorHex"]))
    else:
        colors = {}
    return order, colors

def _y_upper_nice(vmax: float, step_base: int = 10, min_upper: int = 10) -> int:
    """Zaokrouhlení horní meze osy Y na 'pěkné' číslo."""
    if vmax <= 0 or np.isneginf(vmax) or np.isnan(vmax):
        return min_upper
    if vmax <= 100:
        step = step_base
    elif vmax <= 250:
        step = 25
    else:
        step = 50
    return int(max(min_upper, math.ceil(vmax / step) * step))

def _species_colors(df_all: pd.DataFrame) -> dict:
    if "species" not in df_all.columns or "speciesColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(species=lambda d: d["species"].astype(str),
                         speciesColorHex=lambda d: d["speciesColorHex"])
                 .groupby("species")["speciesColorHex"].first())
    return tmp.to_dict()

def _make_bins_labels(df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str):
    vals = pd.to_numeric(df_all.get(value_col), errors="coerce")
    vals_ok = vals.dropna()
    if vals_ok.empty:
        return None, None
    vmin = float(np.floor(vals_ok.min() / bin_size) * bin_size)
    vmax = float(np.ceil (vals_ok.max()  / bin_size) * bin_size)
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
    return int(max(10, math.ceil(vc.max() / 10.0) * 10))

def _compute_binned_counts(df_sub: pd.DataFrame,
                           value_col: str,
                           bins: np.ndarray,
                           labels: list[str]) -> pd.DataFrame:
    d = df_sub.copy()
    d["species"] = d["species"].astype(str)
    cat = pd.Categorical(pd.cut(pd.to_numeric(d[value_col], errors="coerce"),
                                bins=bins, labels=labels,
                                include_lowest=True, right=False, ordered=True),
                         categories=labels, ordered=True)
    d = d.assign(bin=cat)
    pv = (d.pivot_table(index="bin", columns="species", aggfunc="size", fill_value=0))
    long = pv.stack().rename("count").reset_index()
    long["count"] = long["count"].astype(int)
    return long

def _build_three_panel_shared_legend(df_all: pd.DataFrame,
                                     value_col: str,
                                     bin_size: float,
                                     unit_label: str,
                                     color_map: dict,
                                     m_after: pd.Series,
                                     m_removed: pd.Series,
                                     y_upper: int):
    bins, labels = _make_bins_labels(df_all, value_col, bin_size, unit_label)
    if bins is None:
        return None

    long_before  = _compute_binned_counts(df_all,            value_col, bins, labels)
    long_after   = _compute_binned_counts(df_all[m_after],   value_col, bins, labels)
    long_removed = _compute_binned_counts(df_all[m_removed], value_col, bins, labels)

    species_all = (long_before["species"].astype(str).str.strip().drop_duplicates().tolist())
    def color_for(sp): return color_map.get(sp, "#AAAAAA")

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=("Before", "After", "Removed"),
        horizontal_spacing=0.06
    )

    def add_traces(long_df: pd.DataFrame, col: int, show_legend_for_species: set[str]):
        # pořadí X (labels) jednotné pro všechny panely
        for sp in species_all:
            y_vals = (long_df[long_df["species"] == sp]
                      .set_index("bin")
                      .reindex(labels)["count"]
                      .fillna(0).tolist())
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=y_vals,
                    name=sp,
                    marker_color=color_for(sp),
                    legendgroup=sp,
                    showlegend=(sp in show_legend_for_species),
                    hovertemplate=f"%{{x}}<br>Species: {sp}<br>Trees: %{{y}}<extra></extra>"
                ),
                row=1, col=col
            )

    add_traces(long_before,  col=1, show_legend_for_species=set(species_all))
    add_traces(long_after,   col=2, show_legend_for_species=set())
    add_traces(long_removed, col=3, show_legend_for_species=set())

    fig.update_layout(
        barmode="stack",
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.45,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=labels)
    fig.update_yaxes(title_text="Trees", row=1, col=1, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None,    row=1, col=2, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None,    row=1, col=3, tick0=0, range=[0, y_upper])

    return fig

# ========== BY SPECIES (tvá logika s volbou Y) ==========
def _aggregate_by_species(sub: pd.DataFrame, y_col: str) -> pd.DataFrame:
    """Agregace podle typu sloupce y_col:
       - numeric -> sum (NaN=0)
       - bool -> sum (počet True)
       - other -> count řádků
    """
    if y_col not in sub.columns:
        return pd.DataFrame(columns=["species", "value"])

    s = sub[y_col]
    if pd.api.types.is_bool_dtype(s):
        agg_df = sub.groupby("species", as_index=False).agg(value=(y_col, "sum"))
    elif pd.api.types.is_numeric_dtype(pd.to_numeric(s, errors="coerce")):
        s_num = pd.to_numeric(s, errors="coerce").fillna(0.0)
        tmp = sub.copy()
        tmp["_y_"] = s_num
        agg_df = tmp.groupby("species", as_index=False).agg(value=("_y_", "sum"))
    else:
        agg_df = sub.groupby("species", as_index=False).agg(value=("species", "size"))

    return agg_df

def _ensure_all_species(sub_counts: pd.DataFrame, species_order: list[str], color_lookup: dict) -> pd.DataFrame:
    base = pd.DataFrame({"species": species_order})
    sub  = base.merge(
        sub_counts.assign(species=sub_counts["species"].astype(str).str.strip()),
        on="species", how="left"
    )
    sub["value"] = sub["value"].fillna(0).astype(float)
    sub["speciesColorHex"] = sub["species"].map(color_lookup).fillna("#AAAAAA")
    return sub

def render_triple_by_species(df_all: pd.DataFrame, y_col: str):
    required = {"species", "management_status"}
    missing = required - set(df_all.columns)
    if missing:
        st.warning("Chybí sloupce pro vykreslení: " + ", ".join(sorted(missing)))
        return

    species_order, colors = _species_order_and_colors(df_all)
    if not species_order:
        st.info("Nenalezeny žádné druhy.")
        return

    before_df  = _aggregate_by_species(df_all, y_col)
    after_df   = _aggregate_by_species(df_all[mask_after], y_col)
    removed_df = _aggregate_by_species(df_all[mask_removed], y_col)

    before_df  = _ensure_all_species(before_df,  species_order, colors)
    after_df   = _ensure_all_species(after_df,   species_order, colors)
    removed_df = _ensure_all_species(removed_df, species_order, colors)

    y_max = max(before_df["value"].max(), after_df["value"].max(), removed_df["value"].max())
    y_upper = _y_upper_nice(float(y_max))

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=("Before", "After", "Removed"),
        horizontal_spacing=0.06
    )

    def _add_panel(panel_df: pd.DataFrame, col: int):
        for sp, row in panel_df.set_index("species").loc[species_order].iterrows():
            fig.add_trace(
                go.Bar(
                    x=[sp],
                    y=[row["value"]],
                    name=str(sp),
                    marker_color=row["speciesColorHex"],
                    hovertemplate=f"Species: {sp}<br>{y_col}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=col
            )

    _add_panel(before_df, 1)
    _add_panel(after_df, 2)
    _add_panel(removed_df, 3)

    fig.update_layout(
        barmode="group",
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        showlegend=False
    )

    for c in (1, 2, 3):
        fig.update_xaxes(
            title_text=None, tickangle=45,
            categoryorder="array", categoryarray=species_order,
            row=1, col=c
        )

    # Popisek Y podle typu sloupce
    if pd.api.types.is_bool_dtype(df_all[y_col]):
        y_label = f"Count of True ({y_col})"
    elif pd.api.types.is_numeric_dtype(pd.to_numeric(df_all[y_col], errors="coerce")):
        y_label = f"Sum of {y_col}"
    else:
        y_label = f"Count (rows) by species"

    fig.update_yaxes(title_text=y_label, row=1, col=1, tick0=0, range=[0, _y_upper_nice(float(y_max))])
    fig.update_yaxes(title_text=None,    row=1, col=2, tick0=0, range=[0, _y_upper_nice(float(y_max))])
    fig.update_yaxes(title_text=None,    row=1, col=3, tick0=0, range=[0, _y_upper_nice(float(y_max))])

    st.plotly_chart(fig, use_container_width=True)

# ========== BY DBH / HEIGHT CLASS (stackované počty) ==========
def render_triple_by_class(df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str):
    if value_col not in df_all.columns:
        st.warning(f"Chybí sloupec '{value_col}'.")
        return

    bins, labels = _make_bins_labels(df_all, value_col, bin_size, unit_label)
    if bins is None:
        st.info(f"Žádné platné hodnoty pro {value_col}.")
        return

    color_map = _species_colors(df_all)
    y_upper = _y_upper_for(df_all, value_col, bins, labels)

    fig = _build_three_panel_shared_legend(
        df_all=df_all,
        value_col=value_col,
        bin_size=bin_size,
        unit_label=unit_label,
        color_map=color_map,
        m_after=mask_after,
        m_removed=mask_removed,
        y_upper=y_upper
    )
    if fig is None:
        st.info(f"Žádná data pro {value_col} graf.")
        return

    st.plotly_chart(fig, use_container_width=True)

# ========== UI: VOLBY ==========
#st.markdown("**Choose values for plotting**")

# Přepínač režimu X
by_species = "Species Only"
by_dbh = "DBH & Species"
by_heigth = "Height & Species"
mode_options = [by_species, by_dbh, by_heigth]
avail_cols = [c for c in df.columns if c not in exclude_list]

# velikost binu: 5/10/20 cm
dbh_bins = [5, 10, 20]
# velikost binu: 2/5/10 m
h_bins = [2, 5, 10]

c_left, c_left_empty, c_mid,c_right_empty, c_right = st.columns([3, 1, 2, 1, 3])

with c_left:
    #st.markdown("<small><b>Plot Values:<small><b>", unsafe_allow_html=True)
    y_col = st.selectbox("**Select Values to Plot:**", options=avail_cols, index=0)

with c_mid:
    #st.markdown("<small><b>Plot by:<small><b>", unsafe_allow_html=True)
    x_mode = st.pills("**Plot by:**", options=mode_options, default=by_species)

with c_right:
    ##st.markdown("<small><b>Class size:<small><b>", unsafe_allow_html=True)
    bin_size = st.select_slider("**DBH class range [cm]**", options=dbh_bins, value=10)
    bin_size_h = st.select_slider("**Height class range [m]**", options=h_bins, value=5)

# --- vykreslení podle režimu ---
if x_mode == by_species:
    render_triple_by_species(df, y_col)

elif x_mode == by_dbh:
    render_triple_by_class(df_all=df, value_col="dbh", bin_size=float(bin_size), unit_label="cm")

else:  # By Height class
    render_triple_by_class(df_all=df, value_col="height", bin_size=float(bin_size_h), unit_label="m")

with st.expander("See help"):
    st.write('''
        Write help how to use interface.
    ''')