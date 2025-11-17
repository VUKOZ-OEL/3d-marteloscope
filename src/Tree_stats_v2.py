import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import src.io_utils as iou
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.markdown("### Explore tree statistics:")


# --- Data ---
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ========== SETTINGS ==========
CHART_HEIGHT = 350  # chart height

# Columns excluded from Y-variable list (these are categorical/style drivers)
exclude_list = {
    "species", "speciesColorHex", "management_status", "managementColorHex"
}

# --- After/Removed masks ---
keep_status = {"Target tree", "Untouched"}
mask_after   = df.get("management_status", pd.Series(False, index=df.index)).isin(keep_status)
mask_removed = ~mask_after if "management_status" in df.columns else pd.Series(False, index=df.index)

# Standardize dtypes
df = df.copy()
if "species" in df.columns: df["species"] = df["species"].astype(str)
if "speciesColorHex" in df.columns: df["speciesColorHex"] = df["speciesColorHex"].astype(str)
if "management_status" in df.columns: df["management_status"] = df["management_status"].astype(str)
if "managementColorHex" in df.columns: df["managementColorHex"] = df["managementColorHex"].astype(str)

# ========== HELPERS ==========
def _y_upper_nice(vmax: float, step_base: int = 10, min_upper: int = 10) -> int:
    if vmax is None or np.isnan(vmax) or vmax <= 0:
        return min_upper
    if vmax <= 100:
        step = step_base
    elif vmax <= 250:
        step = 25
    else:
        step = 50
    return int(max(min_upper, math.ceil(vmax / step) * step))

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

# --- colors & categories by mode ---
colorBySpp  = "Tree Species"
colorByMgmt = "Tree Management"

def _species_categories_and_colors(df_all: pd.DataFrame):
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
    colors = {c: colors.get(c, "#AAAAAA") for c in order}
    return order, colors

def _management_categories_and_colors(df_all: pd.DataFrame):
    if "management_status" not in df_all.columns:
        return [], {}
    order = pd.Index(df_all["management_status"].astype(str).dropna().unique()).tolist()
    if "managementColorHex" in df_all.columns:
        cmap = (df_all.assign(management_status=lambda x: x["management_status"].astype(str),
                              managementColorHex=lambda x: x["managementColorHex"].astype(str))
                      .groupby("management_status")["managementColorHex"]
                      .first().to_dict())
    else:
        cmap = {}
    cmap = {c: (cmap.get(c) if isinstance(cmap.get(c, ""), str) and cmap.get(c, "").strip() else "#AAAAAA")
            for c in order}
    return order, cmap

def _hue_setup(df_all: pd.DataFrame, color_mode: str):
    """Return (hue_col, categories, color_map, title_suffix)."""
    if color_mode == colorByMgmt:
        cats, colors = _management_categories_and_colors(df_all)
        return "management_status", cats, colors, "(by Management)"
    else:
        cats, colors = _species_categories_and_colors(df_all)
        return "species", cats, colors, "(by Species)"

# ---------- Stats helpers ----------
def _is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False

def _aggfunc_name_for_hover(stat: str, y_col: str) -> str:
    return "Count (rows)" if stat == "Count" else f"{stat} of {y_col}"

# --- binned aggregation by hue + y_col/y_stats ---
def _compute_binned_agg(df_sub: pd.DataFrame,
                        value_col: str,         # DBH / height (bins on X)
                        bins: np.ndarray,
                        labels: list[str],
                        hue_col: str,
                        y_col: str,
                        y_stats: str) -> pd.DataFrame:
    d = df_sub.copy()
    d[hue_col] = d[hue_col].astype(str)

    # assign bin
    cats = pd.Categorical(
        pd.cut(pd.to_numeric(d[value_col], errors="coerce"),
               bins=bins, labels=labels, include_lowest=True, right=False, ordered=True),
        categories=labels, ordered=True
    )
    d = d.assign(bin=cats).dropna(subset=["bin"])

    if y_stats == "Count":
        pv = d.pivot_table(index="bin", columns=hue_col, aggfunc="size", fill_value=0)
        long = pv.stack().rename("value").reset_index()
        long["value"] = long["value"].astype(float)
        return long

    if y_col not in d.columns:
        return pd.DataFrame(columns=["bin", hue_col, "value", "__error__"])
    s = d[y_col]
    if not _is_numeric_like(s):
        return pd.DataFrame(columns=["bin", hue_col, "value", "__error__"]).assign(__error__=True)

    d["_y_"] = pd.to_numeric(s, errors="coerce")

    if y_stats == "Sum":
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].sum().rename(columns={"_y_": "value"})
    elif y_stats == "Mean":
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].mean().rename(columns={"_y_": "value"})
    elif y_stats == "Median":
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].median().rename(columns={"_y_": "value"})
    elif y_stats == "Max":
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].max().rename(columns={"_y_": "value"})
    elif y_stats == "Min":
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].min().rename(columns={"_y_": "value"})
    else:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].size().rename(columns={"_y_": "value"})

    return agg

def _frame_max_value(df_long: pd.DataFrame) -> float:
    if df_long is None or df_long.empty or "value" not in df_long.columns:
        return 0.0
    v = pd.to_numeric(df_long["value"], errors="coerce")
    if v.empty:
        return 0.0
    return float(v.max())

# ========== CATEGORY AGGREGATION (species / management) ==========
def _aggregate_by_hue(sub: pd.DataFrame, hue_col: str, y_col: str, y_stats: str) -> pd.DataFrame:
    if hue_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value"])

    if y_stats == "Count":
        return sub.groupby(hue_col, as_index=False).agg(value=(hue_col, "size")).assign(value=lambda x: x["value"].astype(float))

    if y_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value", "__error__"])

    s = sub[y_col]
    if not _is_numeric_like(s):
        return pd.DataFrame(columns=[hue_col, "value", "__error__"]).assign(__error__=True)

    x = pd.to_numeric(s, errors="coerce")
    tmp = sub.copy()
    tmp["_y_"] = x

    if y_stats == "Sum":
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "sum"))
    elif y_stats == "Mean":
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "mean"))
    elif y_stats == "Median":
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "median"))
    elif y_stats == "Max":
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "max"))
    elif y_stats == "Min":
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "min"))
    else:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=(hue_col, "size"))

    return agg_df

def _ensure_all_categories(sub_counts: pd.DataFrame, hue_col: str, categories: list[str], color_lookup: dict) -> pd.DataFrame:
    base = pd.DataFrame({hue_col: categories})
    sub  = base.merge(
        sub_counts.assign(**{hue_col: sub_counts[hue_col].astype(str).str.strip()}) if not sub_counts.empty else base.assign(value=np.nan),
        on=hue_col, how="left"
    )
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub["value"] = sub["value"].fillna(0.0).astype(float)
    sub["__color__"] = sub[hue_col].map(color_lookup).fillna("#AAAAAA")
    sub["__error__"] = sub_counts["__error__"].iloc[0] if ("__error__" in sub_counts.columns and not sub_counts.empty) else False
    return sub

# ========== BY CATEGORY (bars) ==========
def render_triple_by_category(df_all: pd.DataFrame, y_col: str, y_stats: str, color_mode: str, stacked: bool):
    """Note: stacking/grouping has no visual effect here because each category is its own X tick,
    but we keep the toggle for UI consistency."""
    hue_col, categories, color_map, title_suffix = _hue_setup(df_all, color_mode)
    if not categories:
        st.info("No categories found for the selected color mode.")
        return

    before_df  = _aggregate_by_hue(df_all,               hue_col, y_col, y_stats)
    after_df   = _aggregate_by_hue(df_all[mask_after],   hue_col, y_col, y_stats)
    removed_df = _aggregate_by_hue(df_all[mask_removed], hue_col, y_col, y_stats)

    any_error = any(("__error__" in d.columns and not d.empty) for d in [before_df, after_df, removed_df])
    if any_error and y_stats != "Count":
        st.warning(f"Cannot {y_stats.lower()} non-numeric variable '{y_col}'.")

    before_df  = _ensure_all_categories(before_df,  hue_col, categories, color_map)
    after_df   = _ensure_all_categories(after_df,   hue_col, categories, color_map)
    removed_df = _ensure_all_categories(removed_df, hue_col, categories, color_map)

    y_max = max(before_df["value"].max(), after_df["value"].max(), removed_df["value"].max())
    y_upper = _y_upper_nice(float(y_max))

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=(f"Before {title_suffix}", f"After {title_suffix}", f"Removed {title_suffix}"),
        horizontal_spacing=0.06
    )

    def _add_panel(panel_df: pd.DataFrame, col: int):
        for cat, row in panel_df.set_index(hue_col).loc[categories].iterrows():
            fig.add_trace(
                go.Bar(
                    x=[str(cat)],
                    y=[row["value"]],
                    name=str(cat),
                    marker_color=row["__color__"],
                    legendgroup=str(cat),
                    showlegend=False,
                    hovertemplate=f"{hue_col}: {cat}<br>{_aggfunc_name_for_hover(y_stats, y_col)}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=col
            )

    _add_panel(before_df, 1)
    _add_panel(after_df, 2)
    _add_panel(removed_df, 3)

    fig.update_layout(
        barmode="stack" if stacked else "group",
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        showlegend=False
    )

    for c in (1, 2, 3):
        fig.update_xaxes(
            title_text=None, tickangle=45,
            categoryorder="array", categoryarray=categories,
            row=1, col=c
        )

    y_label = "Count (rows)" if y_stats == "Count" else f"{y_stats} of {y_col}"
    fig.update_yaxes(title_text=y_label, row=1, col=1, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None,    row=1, col=2, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None,    row=1, col=3, tick0=0, range=[0, y_upper])

    st.plotly_chart(fig, use_container_width=True)

# ========== BY DBH / HEIGHT CLASS (stacked/grouped) ==========
def _panel_y_upper_for_long(long_df: pd.DataFrame, stacked: bool) -> float:
    if long_df is None or long_df.empty:
        return 0.0
    if stacked:
        s = long_df.groupby("bin")["value"].sum()
        return float(s.max()) if not s.empty else 0.0
    else:
        return float(long_df["value"].max())

def render_triple_by_class(df_all: pd.DataFrame,
                           value_col: str,
                           bin_size: float,
                           unit_label: str,
                           color_mode: str,
                           y_col: str,
                           y_stats: str,
                           stacked: bool):
    if value_col not in df_all.columns:
        st.warning(f"Missing column '{value_col}'.")
        return

    hue_col, categories, color_map, title_suffix = _hue_setup(df_all, color_mode)
    if not categories:
        st.info("No categories found for the selected color mode.")
        return

    bins, labels = _make_bins_labels(df_all, value_col, bin_size, unit_label)
    if bins is None:
        st.info(f"No valid values for '{value_col}'.")
        return

    # compute three long tables according to the requested stat
    long_before  = _compute_binned_agg(df_all,               value_col, bins, labels, hue_col, y_col, y_stats)
    long_after   = _compute_binned_agg(df_all[mask_after],   value_col, bins, labels, hue_col, y_col, y_stats)
    long_removed = _compute_binned_agg(df_all[mask_removed], value_col, bins, labels, hue_col, y_col, y_stats)

    def has_error(df_long):
        return ("__error__" in df_long.columns) and (df_long["__error__"].any() if not df_long.empty else True)

    if y_stats != "Count" and (has_error(long_before) or has_error(long_after) or has_error(long_removed)):
        st.warning(f"Cannot {y_stats.lower()} non-numeric variable '{y_col}'.")

    # Y-range according to stacked/grouped
    y_max = max(
        _panel_y_upper_for_long(long_before, stacked),
        _panel_y_upper_for_long(long_after,  stacked),
        _panel_y_upper_for_long(long_removed,stacked),
    )
    y_upper = _y_upper_nice(y_max)

    def color_for(cat): return color_map.get(cat, "#AAAAAA")

    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        subplot_titles=(st.session_state.Before, st.session_state.After, st.session_state.Removed),
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
            font=st.session_state.plot_title_font
            )
        for ann in fig.layout.annotations
        ]
    )

    # traces
    def add_panel_traces(long_df: pd.DataFrame, col: int, show_legend_for: set[str]):
        for cat in categories:
            if long_df.empty:
                y_vals = [0]*len(labels)
            else:
                y_vals = (long_df[long_df.get(hue_col, "") == cat]
                          .set_index("bin")
                          .reindex(labels)["value"]
                          .fillna(0).tolist())
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=y_vals,
                    name=str(cat),
                    marker_color=color_for(cat),
                    legendgroup=str(cat),
                    showlegend=(cat in show_legend_for),
                    hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{_aggfunc_name_for_hover(y_stats, y_col)}: %{{y:.2f}}<extra></extra>"
                ),
                row=1, col=col
            )

    add_panel_traces(long_before,  col=1, show_legend_for=set(categories))
    add_panel_traces(long_after,   col=2, show_legend_for=set())
    add_panel_traces(long_removed, col=3, show_legend_for=set())

    fig.update_layout(
        barmode="stack" if stacked else "group",
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
    fig.update_yaxes(title_text=_aggfunc_name_for_hover(y_stats, y_col), row=1, col=1, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=2, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, range=[0, y_upper])

    st.plotly_chart(fig, use_container_width=True)

# ========== UI ==========
by_category = "Category"
by_dbh = "DBH"
by_height = "Height"
mode_options = [by_dbh, by_height, by_category]

avail_cols = [c for c in df.columns if c not in exclude_list]
avail_stats = ["Count","Sum","Mean","Median","Max","Min"]

# DBH bin sizes [cm] and Height bin sizes [m]
dbh_bins = [5, 10, 20]
h_bins = [2, 5, 10]

c_left, c_left_empty, c_mid, c_right_empty, c_right = st.columns([3, 1, 4, 1, 3])

with c_left:
    y_col = st.selectbox(
        "**Select Values to Plot:**",
        options=avail_cols, index=0,
        help="Choose the column to aggregate on Y-axis (e.g., Volume_m3, CrownWidth, etc.). "
             "Text columns can only be used with 'Count'."
    )
    y_stats = st.selectbox(
        "**How to Plot (Y statistic):**",
        options=avail_stats, index=0,
        help="Pick the descriptive statistic for Y-axis: "
             "• Count = number of rows • Sum/Mean/Median/Max/Min = numeric-only. "
             "If a non-numeric column is selected for these, you'll see a warning."
    )

with c_mid:
    x_mode = st.segmented_control(
        "**Plot by:**",
        options=mode_options, default=by_dbh, width = "stretch",
        help="Choose the X-axis structure: "
             "• DBH = distribution by diameter classes • Height = distribution by height classes • Category = totals per category."
    )
    color_mode = st.segmented_control(
        "**Color by:**",
        options=[colorBySpp, colorByMgmt], default=colorBySpp, width = "stretch",
        help="Controls coloring AND grouping"
    )
    stacked = st.toggle(
        "**Stacked bars**",
        value=True,
        help="Switch between stacked and grouped bars. "
             "While stacked mode shows also 'Count'/'Sum' in class, grouped is better for direct comparison of categories. "
             "For 'Mean', 'Median', 'Max' and 'Min' is strongly suggested to disable stacked mode."
    )

with c_right:
    bin_size = st.select_slider(
        "**DBH class range [cm]**",
        options=dbh_bins, value=10,
        help="Class width for DBH histograms (only used when 'Plot by' = DBH)."
    )
    bin_size_h = st.select_slider(
        "**Height class range [m]**",
        options=h_bins, value=5,
        help="Class width for Height histograms (only used when 'Plot by' = Height)."
    )

# --- Render by mode ---
if x_mode == by_category:
    render_triple_by_category(df, y_col, y_stats, color_mode, stacked)

elif x_mode == by_dbh:
    render_triple_by_class(df_all=df, value_col="dbh", bin_size=float(bin_size), unit_label="cm",
                           color_mode=color_mode, y_col=y_col, y_stats=y_stats, stacked=stacked)

else:  # by_height
    render_triple_by_class(df_all=df, value_col="height", bin_size=float(bin_size_h), unit_label="m",
                           color_mode=color_mode, y_col=y_col, y_stats=y_stats, stacked=stacked)

# --- See help ---
with st.expander("See help"):
    st.markdown("""
## How to use this interface

### 1) Select Values to Plot
Choose the **data column** that will be aggregated on the **Y-axis**.  
- Examples: `Volume_m3`, `BasalArea_m2`, `CrownWidth`, etc.  
- Text columns are allowed only with **Count**.

### 2) How to Plot (Y statistic)
Pick a **descriptive statistic** for the Y-axis:
- **Count**: number of records (works for any column).
- **Sum / Mean / Median / Max / Min**: require a **numeric** (or boolean) column.
  - If you pick a non-numeric column here, you'll see a warning like:
    - *“Cannot sum non-numeric variable 'species'.”*
  - Tip: If your numeric column contains text placeholders (e.g., `"NA"`), they are treated as missing.

### 3) Plot by
Defines the **X-axis**:
- **DBH**: histogram-like bars by **DBH classes** (in cm). Uses *DBH class range*.
- **Height**: histogram-like bars by **height classes** (in m). Uses *Height class range*.
- **Category**: one bar **per category** (Species or Management), i.e., totals by group.

Each mode always shows three panels:
- **Before** (all trees),
- **After** (Target tree + Untouched),
- **Removed** (all the remaining trees).

### 4) Color by
Controls **both coloring and grouping**:
- **Tree Species**: groups by `species` and uses `speciesColorHex`.
- **Tree Management**: groups by `management_status` and uses `managementColorHex`.
If a color is missing for a category, a neutral gray `#AAAAAA` is used.

### 5) Stacked bars
- **ON (stacked)**: categories are summed within each X-bin.  
  - Great for totals; the Y-axis top is derived from **bin sums**.
- **OFF (grouped)**: categories are side-by-side within each X-bin.  
  - Best for **Max** or direct category comparison; the Y-axis top is derived from **max single bar**.

### 6) Class ranges
- **DBH class range [cm]**: width of DBH bins. Smaller bins show finer structure but may be noisy.
- **Height class range [m]**: width of height bins. Same trade-offs.

---

## Best practices & examples

- Want species composition by **volume** in the managed stand?  
  **Color by** = *Tree Species*, **Plot by** = *DBH* or *Height*, **How to Plot** = *Sum*, **Select Values** = *Volume_m3*, then toggle **Stacked** as desired.

- Compare **Max height** per management category:  
  **Color by** = *Tree Management*, **Plot by** = *Height*, **How to Plot** = *Max*, **Select Values** = *height*, set **Stacked** = OFF (grouped) to see per-category peaks.

- Count of trees by DBH class and species:  
  **How to Plot** = *Count*, **Plot by** = *DBH*, **Color by** = *Tree Species*. Works regardless of column types.

---

## Troubleshooting

- **Empty chart / very low bars**: check that your **class ranges** make sense (too narrow bins can be sparse).
- **Warning “Cannot … non-numeric variable”**: change **How to Plot** to **Count** or pick a **numeric** column in **Select Values**.
- **Colors look the same**: ensure the data contains `speciesColorHex` / `managementColorHex` for all categories; otherwise a default gray is used.
- **Removed panel is empty**: verify that `management_status` is present and that some trees are actually outside `Target tree` / `Untouched`.

---

## Notes

- Booleans count as numeric for statistics (e.g., Sum = count of `True`).
- In **Category** mode, the *Stacked* toggle is kept for consistency but has no visible effect (each category is its own X tick).
- Y-axis scaling adapts to **Stacked** vs **Grouped** automatically to keep bars readable.
""")
