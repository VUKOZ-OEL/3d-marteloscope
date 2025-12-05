import streamlit as st
import pandas as pd
import numpy as np
import math
import src.io_utils as iou  # if unused, you can safely remove
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Session-scoped helpers
# --------------------------------------------------------------------------------------

if "plot_uid" not in st.session_state:
    st.session_state.plot_uid = 0


def get_uid():
    """Return a unique plot key for Streamlit charts."""
    st.session_state.plot_uid += 1
    return st.session_state.plot_uid


# --------------------------------------------------------------------------------------
# Global color mode setup (from session_state)
# --------------------------------------------------------------------------------------

colorBySpp = st.session_state.Species
colorByMgmt = st.session_state.Management

st.markdown("### Explore tree statistics:")

df_raw: pd.DataFrame = st.session_state.trees.copy()

# Fixed chart height for all plots
CHART_HEIGHT = 350

# Columns excluded from some Y-variable lists (categorical / styling helpers)
exclude_list = {"species", "speciesColorHex", "management_status", "managementColorHex"}

# --------------------------------------------------------------------------------------
# Masks for "after management" vs "removed"
# --------------------------------------------------------------------------------------

df = df_raw.copy()
keep_status = {"Target tree", "Untouched"}

mask_after = df.get("management_status", pd.Series(False, index=df.index)).isin(
    keep_status
)
mask_removed = (
    ~mask_after
    if "management_status" in df.columns
    else pd.Series(False, index=df.index)
)

# Standardize dtypes for categorical columns
for col in ["species", "speciesColorHex", "management_status", "managementColorHex"]:
    if col in df.columns:
        df[col] = df[col].astype(str)


# --------------------------------------------------------------------------------------
# Numeric helpers and "nice" axis scaling
# --------------------------------------------------------------------------------------

def _y_upper_nice(vmax: float, y_label: str = "", min_upper: float = 1.0) -> float:
    """
    Unified Y-axis rounding:

    - Basal Area:
        if vmax < 5 → round up to nearest 0.1 (e.g., 1.62 → 1.7)
        else → normal ceil

    - Everything else → round UP to nearest 10
    """

    if vmax is None or np.isnan(vmax) or vmax <= 0:
        return min_upper

    if vmax < 5:
        return math.ceil(vmax * 10) / 10.0  # 1 desetina nahoru
    else:
        return math.ceil(vmax)  # běžné chování

    # ---- All other variables: round UP to nearest 10 ----
    return math.ceil(vmax / 10) * 10



def auto_round_step(vmax: float) -> float:
    """
    Choose a reasonable round step for Y-axis based on data scale.
    Used for violin plots with slider-controlled range.
    """
    if vmax <= 0:
        return 1.0
    if vmax <= 10:
        return 1.0
    if vmax <= 50:
        return 5.0
    if vmax <= 200:
        return 10.0
    if vmax <= 1000:
        return 50.0
    return 100.0


def _make_bins_labels(
    df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str
):
    """
    Compute bin edges and labels for class-based bar charts (DBH / height).
    """
    vals = pd.to_numeric(df_all.get(value_col), errors="coerce")
    vals_ok = vals.dropna()
    if vals_ok.empty:
        return None, None
    vmin = float(np.floor(vals_ok.min() / bin_size) * bin_size)
    vmax = float(np.ceil(vals_ok.max() / bin_size) * bin_size)
    if vmax <= vmin:
        vmax = vmin + bin_size
    bins = np.arange(vmin, vmax + bin_size, bin_size, dtype=float)
    labels = [f"{int(b)}–{int(b + bin_size)} {unit_label}" for b in bins[:-1]]
    return bins, labels


# --------------------------------------------------------------------------------------
# Category + color helpers (species / management)
# --------------------------------------------------------------------------------------

def _species_categories_and_colors(df_all: pd.DataFrame):
    """
    Return (ordered_species_list, species_color_dict).
    Order is based on frequency (most frequent first).
    """
    if "species" not in df_all.columns:
        return [], {}

    order = (
        df_all["species"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_values(ascending=False)
        .index.astype(str)
        .str.strip()
        .tolist()
    )

    if "speciesColorHex" in df_all.columns:
        col_lookup = (
            df_all.groupby("species", as_index=False)["speciesColorHex"]
            .first()
            .assign(species=lambda d: d["species"].astype(str).str.strip())
        )
        colors = dict(zip(col_lookup["species"], col_lookup["speciesColorHex"]))
    else:
        colors = {}

    colors = {c: colors.get(c, "#AAAAAA") for c in order}
    return order, colors


def _management_categories_and_colors(df_all: pd.DataFrame):
    """
    Return (ordered_management_status_list, status_color_dict).
    """
    if "management_status" not in df_all.columns:
        return [], {}

    order = pd.Index(df_all["management_status"].astype(str).dropna().unique()).tolist()

    if "managementColorHex" in df_all.columns:
        cmap = (
            df_all.assign(
                management_status=lambda x: x["management_status"].astype(str),
                managementColorHex=lambda x: x["managementColorHex"].astype(str),
            )
            .groupby("management_status")["managementColorHex"]
            .first()
            .to_dict()
        )
    else:
        cmap = {}

    cmap = {
        c: (
            cmap.get(c)
            if isinstance(cmap.get(c, ""), str) and cmap.get(c, "").strip()
            else "#AAAAAA"
        )
        for c in order
    }
    return order, cmap


def _hue_setup(df_all: pd.DataFrame, color_mode: str):
    """
    Decide which column is used for color/hue and return:

        (hue_col_name, categories_list, color_map_dict, title_suffix_str)
    """
    if color_mode == colorByMgmt:
        cats, colors = _management_categories_and_colors(df_all)
        return "management_status", cats, colors, "(by Management)"
    else:
        cats, colors = _species_categories_and_colors(df_all)
        return "species", cats, colors, "(by Species)"


# --------------------------------------------------------------------------------------
# Generic stats helpers
# --------------------------------------------------------------------------------------

def _is_numeric_like(s: pd.Series) -> bool:
    """Return True if Series can be treated as numeric (including bool)."""
    if pd.api.types.is_bool_dtype(s):
        return True
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False


def _aggfunc_name_for_hover(stat: str, y_col: str) -> str:
    """Return human-readable label for hover text."""
    return "Tree Count" if stat == "Count" else f"{stat} of {y_col}"


# --------------------------------------------------------------------------------------
# Binned aggregation (for class-based bar plots)
# --------------------------------------------------------------------------------------

def _compute_binned_agg(
    df_sub: pd.DataFrame,
    value_col: str,  # e.g. DBH / height (bins on X)
    bins: np.ndarray,
    labels: list[str],
    hue_col: str,
    y_col: str,
    y_stats: str,
) -> pd.DataFrame:
    """
    Compute aggregated values for each bin and hue category.

    Returns long-format DataFrame with columns: ["bin", hue_col, "value", (optional)__error__]
    """
    d = df_sub.copy()
    d[hue_col] = d[hue_col].astype(str)

    cats = pd.Categorical(
        pd.cut(
            pd.to_numeric(d[value_col], errors="coerce"),
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
            ordered=True,
        ),
        categories=labels,
        ordered=True,
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
        return pd.DataFrame(columns=["bin", hue_col, "value", "__error__"]).assign(
            __error__=True
        )

    d["_y_"] = pd.to_numeric(s, errors="coerce")

    if y_stats == "Sum":
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .sum()
            .rename(columns={"_y_": "value"})
        )
    elif y_stats == "Mean":
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .mean()
            .rename(columns={"_y_": "value"})
        )
    elif y_stats == "Median":
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .median()
            .rename(columns={"_y_": "value"})
        )
    elif y_stats == "Max":
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .max()
            .rename(columns={"_y_": "value"})
        )
    elif y_stats == "Min":
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .min()
            .rename(columns={"_y_": "value"})
        )
    else:
        agg = (
            d.groupby(["bin", hue_col], as_index=False)["_y_"]
            .size()
            .rename(columns={"_y_": "value"})
        )

    return agg


def _frame_max_value(df_long: pd.DataFrame) -> float:
    """Return max value from 'value' column of long-format DF."""
    if df_long is None or df_long.empty or "value" not in df_long.columns:
        return 0.0
    v = pd.to_numeric(df_long["value"], errors="coerce")
    if v.empty:
        return 0.0
    return float(v.max())


# --------------------------------------------------------------------------------------
# Aggregation by category (species / management) for bar plots
# --------------------------------------------------------------------------------------

def _aggregate_by_hue(
    sub: pd.DataFrame, hue_col: str, y_col: str, y_stats: str
) -> pd.DataFrame:
    """
    Aggregate per category in hue_col.

    For Count → frequency.
    For numeric y_col → Sum / Mean / Median / Min / Max accordingly.
    """
    if hue_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value"])

    if y_stats == "Count":
        return (
            sub.groupby(hue_col, as_index=False)
            .agg(value=(hue_col, "size"))
            .assign(value=lambda x: x["value"].astype(float))
        )

    if y_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value", "__error__"])

    s = sub[y_col]
    if not _is_numeric_like(s):
        return pd.DataFrame(columns=[hue_col, "value", "__error__"]).assign(
            __error__=True
        )

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


def _ensure_all_categories(
    sub_counts: pd.DataFrame, hue_col: str, categories: list[str], color_lookup: dict
) -> pd.DataFrame:
    """
    Ensure output has all categories in correct order, with zero-filled missing rows.
    """
    base = pd.DataFrame({hue_col: categories})
    sub = base.merge(
        sub_counts.assign(**{hue_col: sub_counts[hue_col].astype(str).str.strip()})
        if not sub_counts.empty
        else base.assign(value=np.nan),
        on=hue_col,
        how="left",
    )
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub["value"] = sub["value"].fillna(0.0).astype(float)
    sub["__color__"] = sub[hue_col].map(color_lookup).fillna("#AAAAAA")
    sub["__error__"] = (
        sub_counts["__error__"].iloc[0]
        if ("__error__" in sub_counts.columns and not sub_counts.empty)
        else False
    )
    return sub


# --------------------------------------------------------------------------------------
# Triple bar plot by category (Before / After / Removed)
# --------------------------------------------------------------------------------------

def render_triple_by_category(
    df_all: pd.DataFrame, y_col: str, y_stats: str, color_mode: str, stacked: bool
):
    """
    Render 3-panel bar chart (Before / After / Removed) by category (species/management).
    Used for Tree Count and other aggregated metrics by category.
    """
    hue_col, categories, color_map, title_suffix = _hue_setup(df_all, color_mode)
    if not categories:
        st.info("No categories found for the selected color mode.")
        return

    before_df = _aggregate_by_hue(df_all, hue_col, y_col, y_stats)
    after_df = _aggregate_by_hue(df_all[mask_after], hue_col, y_col, y_stats)
    removed_df = _aggregate_by_hue(df_all[mask_removed], hue_col, y_col, y_stats)

    any_error = any(
        ("__error__" in d.columns and not d.empty)
        for d in [before_df, after_df, removed_df]
    )
    if any_error and y_stats != "Count":
        st.warning(f"Cannot {y_stats.lower()} non-numeric variable '{y_col}'.")

    before_df = _ensure_all_categories(before_df, hue_col, categories, color_map)
    after_df = _ensure_all_categories(after_df, hue_col, categories, color_map)
    removed_df = _ensure_all_categories(removed_df, hue_col, categories, color_map)

    y_max = max(
        before_df["value"].max(), after_df["value"].max(), removed_df["value"].max()
    )
    y_upper = _y_upper_nice(float(y_max), "Tree Count")

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=("Before", "After", "Removed"),
        horizontal_spacing=0.06,
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
                    hovertemplate=(
                        f"{hue_col}: {cat}<br>"
                        "value: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    _add_panel(before_df, 1)
    _add_panel(after_df, 2)
    _add_panel(removed_df, 3)

    fig.update_layout(
        barmode="stack" if stacked else "group",
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        showlegend=False,
    )

    for c in (1, 2, 3):
        fig.update_xaxes(
            title_text=None,
            tickangle=45,
            categoryorder="array",
            categoryarray=categories,
            row=1,
            col=c,
        )

    y_label = "Tree Count" if y_stats == "Count" else f"{y_stats} of {y_col}"
    fig.update_yaxes(
        title_text=y_label,
        row=1,
        col=1,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=2,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=3,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"category_{get_uid()}")


# --------------------------------------------------------------------------------------
# Triple bar plot by DBH / height classes
# --------------------------------------------------------------------------------------

def _panel_y_upper_for_long(long_df: pd.DataFrame, stacked: bool) -> float:
    """Return suitable Y-max for stacked/grouped bar from long-format data."""
    if long_df is None or long_df.empty:
        return 0.0
    if stacked:
        s = long_df.groupby("bin")["value"].sum()
        return float(s.max()) if not s.empty else 0.0
    else:
        return float(long_df["value"].max())


def render_triple_by_class(
    df_all: pd.DataFrame,
    value_col: str,
    bin_size: float,
    unit_label: str,
    color_mode: str,
    y_col: str,
    y_stats: str,
    stacked: bool,
):
    """
    Render 3-panel bar chart for DBH / height classes (Before / After / Removed).
    Used for Tree Count per class.
    """
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

    # Compute three long tables for the requested statistic
    long_before = _compute_binned_agg(
        df_all, value_col, bins, labels, hue_col, y_col, y_stats
    )
    long_after = _compute_binned_agg(
        df_all[mask_after], value_col, bins, labels, hue_col, y_col, y_stats
    )
    long_removed = _compute_binned_agg(
        df_all[mask_removed], value_col, bins, labels, hue_col, y_col, y_stats
    )

    def has_error(df_long):
        return ("__error__" in df_long.columns) and (
            df_long["__error__"].any() if not df_long.empty else True
        )

    if y_stats != "Count" and (
        has_error(long_before) or has_error(long_after) or has_error(long_removed)
    ):
        st.warning(f"Cannot {y_stats.lower()} non-numeric variable '{y_col}'.")

    y_max = max(
        _panel_y_upper_for_long(long_before, stacked),
        _panel_y_upper_for_long(long_after, stacked),
        _panel_y_upper_for_long(long_removed, stacked),
    )
    y_upper = _y_upper_nice(y_max,y_label)

    def color_for(cat):
        return color_map.get(cat, "#AAAAAA")

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=(
            getattr(st.session_state, "Before", "Before"),
            getattr(st.session_state, "After", "After"),
            getattr(st.session_state, "Removed", "Removed"),
        ),
        horizontal_spacing=0.06,
    )

    # Optional: custom styling from session_state (disabled by default)
    if "plot_title_font" in st.session_state:
        fig.update_layout(
            annotations=[
                dict(
                    text=ann.text,
                    x=ann.x,
                    y=ann.y,
                    xref=ann.xref,
                    yref=ann.yref,
                    showarrow=False,
                )
                for ann in fig.layout.annotations
            ]
        )

    def add_panel_traces(long_df: pd.DataFrame, col: int, show_legend_for: set[str]):
        for cat in categories:
            if long_df.empty:
                y_vals = [0] * len(labels)
            else:
                y_vals = (
                    long_df[long_df.get(hue_col, "") == cat]
                    .set_index("bin")
                    .reindex(labels)["value"]
                    .fillna(0)
                    .tolist()
                )
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=y_vals,
                    name=str(cat),
                    marker_color=color_for(cat),
                    legendgroup=str(cat),
                    showlegend=(cat in show_legend_for),
                    hovertemplate=(
                        "%{x}<br>"
                        f"{hue_col}: {cat}<br>"
                        "value: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    add_panel_traces(long_before, col=1, show_legend_for=set(categories))
    add_panel_traces(long_after, col=2, show_legend_for=set())
    add_panel_traces(long_removed, col=3, show_legend_for=set())

    fig.update_layout(
        barmode="stack" if stacked else "group",
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5),
    )
    fig.update_xaxes(
        title_text=None, tickangle=45, categoryorder="array", categoryarray=labels
    )
    fig.update_yaxes(
        title_text=_aggfunc_name_for_hover(y_stats, y_col),
        row=1,
        col=1,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=2,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=3,
        tick0=0,
        range=[0, y_upper],
        showticklabels=True,
        tickmode="auto",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"class_{get_uid()}")


# --------------------------------------------------------------------------------------
# Triple violin (generic metric, Before / After / Removed)
# --------------------------------------------------------------------------------------

def render_triple_violin(
    df_all: pd.DataFrame,
    value_col: str,
    y_label: str,
    color_mode: str,
    slider_range: tuple[float, float] | None = None,
):
    """
    Render 3-panel violin plot (Before / After / Removed) for any numeric metric.
    Coloured by species / management.

    If slider_range is provided, filter values to the given min/max range,
    and scale Y-axis accordingly in a fixed step.
    """
    if value_col not in df_all.columns:
        st.warning(f"Missing column '{value_col}'.")
        return

    hue_col, categories, color_map, title_suffix = _hue_setup(df_all, color_mode)
    if not categories:
        st.info("No categories found for selected color mode.")
        return

    def _clean_numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # Apply slider filter (if provided)
    df_work = df_all.copy()
    df_work[value_col] = _clean_numeric(df_work[value_col])
    df_work = df_work.dropna(subset=[value_col])

    if slider_range is not None:
        lo, hi = slider_range
        df_work = df_work[
            (df_work[value_col] >= lo) & (df_work[value_col] <= hi)
        ].copy()

    if df_work.empty:
        st.info("No data after applying value filter.")
        return

    panels = [
        (df_work, getattr(st.session_state, "Before", "Before")),
        (df_work[mask_after], getattr(st.session_state, "After", "After")),
        (df_work[mask_removed], getattr(st.session_state, "Removed", "Removed")),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.06,
    )

    max_val = 0.0
    for col_idx, (sub, title) in enumerate(panels, start=1):
        if sub.empty:
            continue

        sub = sub.copy()
        sub[hue_col] = sub[hue_col].astype(str)
        sub[value_col] = _clean_numeric(sub[value_col])
        sub = sub.dropna(subset=[value_col])

        for cat in categories:
            vals = sub.loc[sub[hue_col] == cat, value_col].dropna()
            if vals.empty:
                continue

            max_val = max(max_val, float(vals.max()))

            fig.add_trace(
                go.Violin(
                    x=[cat] * len(vals),
                    y=vals,
                    name=str(cat),
                    legendgroup=str(cat),
                    line_color="black",
                    line_width=1,
                    fillcolor=color_map.get(cat, "#AAAAAA"),
                    opacity=0.7,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,  # hide individual points
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"{hue_col}: {cat}<br>"
                        "value: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )

    if max_val <= 0:
        max_val = 1.0

    round_step = auto_round_step(max_val)
    # ---- Basal Area: pokud maximum < 5, zaokrouhli nahoru na 1 desetinu ----

    y_upper = _y_upper_nice(max_val, y_label)

    final_ylabel = make_y_label(y_label)

    # spodní hranice Y = spodní hodnota slideru (pokud existuje)
    y_min = float(slider_range[0]) if slider_range is not None else 0.0

    fig.update_yaxes(
        title_text=final_ylabel,
        row=1,
        col=1,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=2,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=3,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"violin_{get_uid()}")


# --------------------------------------------------------------------------------------
# NEW: Triple violin for Projection Exposure (special case)
# --------------------------------------------------------------------------------------

def render_projection_exposure_page(
    df: pd.DataFrame,
    color_mode: str,
    slider_range: tuple[float, float] | None = None,
    y_label: str | None = None,
):
    """
    Render three violin plots for Projection Exposure:

    1) Before cut:
       - All trees using 'projection_exposure'
    2) After mgmt:
       - Only trees with management_status in {Target tree, Untouched}
         using 'projection_exposure_after_mgmt'
    3) Removed:
       - Trees with management_status NOT in {Target tree, Untouched}
         using 'projection_exposure'
    """
    required_cols = {"projection_exposure", "management_status"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(
            f"Missing required columns for Projection Exposure: {', '.join(missing)}"
        )
        return

    hue_col, categories, color_map, title_suffix = _hue_setup(df, color_mode)
    if not categories:
        st.info("No categories found for selected color mode.")
        return

    keep_status = {"Target tree", "Untouched"}

    def clean_vals(s):
        return pd.to_numeric(s, errors="coerce")

    # Panel 1: BEFORE CUT – all trees, projection_exposure
    df_before = df.copy()
    df_before["projection_exposure"] = clean_vals(df_before["projection_exposure"])
    df_before = df_before.dropna(subset=["projection_exposure"])

    # Panel 2: AFTER MGMT – only Target/Untouched, projection_exposure_after_mgmt
    if "projection_exposure_after_mgmt" in df.columns:
        df_after = df[df["management_status"].isin(keep_status)].copy()
        df_after["projection_exposure_after_mgmt"] = clean_vals(
            df_after["projection_exposure_after_mgmt"]
        )
        df_after = df_after.dropna(subset=["projection_exposure_after_mgmt"])
    else:
        df_after = pd.DataFrame()
        st.info(
            "Column 'projection_exposure_after_mgmt' not found, skipping 'After mgmt' panel."
        )

    # Panel 3: REMOVED – all other statuses, projection_exposure
    df_removed = df[~df["management_status"].isin(keep_status)].copy()
    df_removed["projection_exposure_removed"] = clean_vals(
        df_removed["projection_exposure"]
    )
    df_removed = df_removed.dropna(subset=["projection_exposure_removed"])

    # Apply slider filter if provided
    if slider_range is not None:
        lo, hi = slider_range
        if not df_before.empty:
            df_before = df_before[
                (df_before["projection_exposure"] >= lo)
                & (df_before["projection_exposure"] <= hi)
            ]
        if not df_after.empty and "projection_exposure_after_mgmt" in df_after.columns:
            df_after = df_after[
                (df_after["projection_exposure_after_mgmt"] >= lo)
                & (df_after["projection_exposure_after_mgmt"] <= hi)
            ]
        if not df_removed.empty:
            df_removed = df_removed[
                (df_removed["projection_exposure_removed"] >= lo)
                & (df_removed["projection_exposure_removed"] <= hi)
            ]

    if df_before.empty and df_after.empty and df_removed.empty:
        st.info("No data for Projection Exposure after applying value filter.")
        return

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=["Before cut", "After mgmt", "Removed"],
        horizontal_spacing=0.06,
    )

    panels = [
        ("projection_exposure", df_before, 1),
        ("projection_exposure_after_mgmt", df_after, 2),
        ("projection_exposure_removed", df_removed, 3),
    ]

    max_val = 0.0

    for value_col, sub, col_idx in panels:
        if sub.empty or value_col not in sub.columns:
            continue

        sub = sub.copy()
        sub[hue_col] = sub[hue_col].astype(str)
        sub[value_col] = clean_vals(sub[value_col])
        sub = sub.dropna(subset=[value_col])

        for cat in categories:
            vals = sub.loc[sub[hue_col] == cat, value_col].dropna()
            if vals.empty:
                continue

            max_val = max(max_val, float(vals.max()))

            fig.add_trace(
                go.Violin(
                    x=[cat] * len(vals),
                    y=vals,
                    name=str(cat),
                    legendgroup=str(cat),
                    line_color="black",
                    line_width=1,
                    fillcolor=color_map.get(cat, "#AAAAAA"),
                    opacity=0.7,
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"{hue_col}: {cat}<br>"
                        "value: %{y:.2f}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )

    if max_val <= 0:
        max_val = 1.0

    round_step = auto_round_step(max_val)
    y_upper = _y_upper_nice(max_val, y_label)

    label_for_axis = y_label or "Projection Exposure"
    final_ylabel = make_y_label(label_for_axis)

    # spodní hranice Y = spodní hodnota slideru (pokud existuje)
    y_min = float(slider_range[0]) if slider_range is not None else 0.0

    fig.update_yaxes(
        title_text=final_ylabel,
        row=1,
        col=1,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=2,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )
    fig.update_yaxes(
        title_text=None,
        row=1,
        col=3,
        range=[y_min, y_upper],
        showticklabels=True,
        tickmode="auto",
    )

    st.plotly_chart(fig, use_container_width=True, key=f"projexp_{get_uid()}")


# --------------------------------------------------------------------------------------
# UI: metric mapping and controls
# --------------------------------------------------------------------------------------

VALUE_TREE_COUNT = "Tree Count"
value_mapping = {
    VALUE_TREE_COUNT: None,  # special case (Count)
    "DBH": "dbh",
    "Basal Area": "basal_area_m2",
    "Volume": "Volume_m3",
    "Tree Height": "height",
    "Crown Base Height": "crown_base_height",  # renamed in backend to crownStartHeight
    "Crown Centroid Height": "crown_centroid_height",
    "Crown Volume": "crown_volume",
    "Crown Surface Area": "crown_surface",
    "Horizontal Crown Projection Area": "horizontal_crown_proj",
    "Vertical Crown Projection Area": "vertical_crown_proj",
    "Crown Eccentricity": "crown_eccentricity",
    "Height-DBH Ratio": "heightXdbh",
    "Projection Exposure": "projection_exposure",  # NEW special metric
}

# Units for Y-axis based on metric label
y_units = {
    "DBH": "cm",
    "Basal Area": "m²",
    "Volume": "m³",
    "Tree Height": "m",
    "Crown Base Height": "m",
    "Crown Centroid Height": "m",
    "Crown Volume": "m³",
    "Crown Surface Area": "m²",
    "Horizontal Crown Projection Area": "m²",
    "Vertical Crown Projection Area": "m²",
    "Crown Eccentricity": "m",
    "Height-DBH Ratio": "",
    "Projection Exposure": "%",
}


def make_y_label(label: str) -> str:
    """Construct final Y-axis label from metric name + unit."""
    unit = y_units.get(label, "")
    if unit:
        return f"{label} [{unit}]"
    return label


# Only show metrics actually present in the DataFrame
value_options = []
for label, col in value_mapping.items():
    if label == VALUE_TREE_COUNT:
        value_options.append(label)
    else:
        if col in df.columns:
            value_options.append(label)

# "Plot by" modes
by_category = "Category"
by_dbh = "DBH"
by_height = "Height"
mode_options = [by_dbh, by_height, by_category]

color_mode_default = colorBySpp

# Layout columns
c_left, c_left_empty, c_mid, c_right_empty, c_right = st.columns([3, 1, 4, 1, 3])

# --------------------------------------------------------------------------------------
# LEFT: Value selection
# --------------------------------------------------------------------------------------

with c_left:
    default_index = 0
    if VALUE_TREE_COUNT in value_options:
        default_index = value_options.index(VALUE_TREE_COUNT)
    y_label = st.selectbox(
        "**Values to plot:**",
        options=value_options,
        index=default_index,
        help=(
            "Choose variable to display:\n"
            "- **Tree count** is plotted as barplot.\n"
            "- Other variables are shown as violin plot.\n"
            "- **Projection Exposure** uses a dedicated triple violin layout."
        ),
    )

# --------------------------------------------------------------------------------------
# MID: Plot by / Color by / Stacked toggle
# --------------------------------------------------------------------------------------

with c_mid:
    is_tree_count = y_label == VALUE_TREE_COUNT

    if is_tree_count:
        x_mode = st.segmented_control(
            "**Plot by:**",
            options=mode_options,
            default=by_dbh,
            width="stretch",
            help=(
                "Show tree count according to:\n"
                "- **DBH** class\n"
                "- **Height** class\n"
                "- **Category** (Species/Management)."
            ),
        )
    else:
        # For all other metrics, including Projection Exposure, we always plot by category.
        x_mode = by_category
        st.segmented_control(
            "**Plot by:**",
            options=[by_category],
            default=by_category,
            disabled=True,
            width="stretch",
            help="For selected variable only display by Category is allowed.",
        )

    color_mode = st.segmented_control(
        "**Color by:**",
        options=[colorBySpp, colorByMgmt],
        default=color_mode_default,
        width="stretch",
        help="Select category (Species / Management) to color plots by.",
    )

    # Stacked makes sense only for barplots (Tree count)
    stacked = (
        st.toggle(
            "**Stacked bars**",
            value=True,
            help=("Only for Tree Count, switch between stacked and grouped mode."),
        )
        if is_tree_count
        else False
    )

# --------------------------------------------------------------------------------------
# RIGHT: DBH / Height controls (either bin size for Tree Count, or filters for others)
# --------------------------------------------------------------------------------------

with c_right:
    if is_tree_count:
        # Class width controls (used only for Tree Count)
        dbh_bins = [5, 10, 20]
        h_bins = [2, 5, 10]

        bin_size = st.select_slider(
            "**DBH class range [cm]**",
            options=dbh_bins,
            value=10,
            help="Width of DBH bands.",
        )
        bin_size_h = st.select_slider(
            "**Height class range [m]**",
            options=h_bins,
            value=5,
            help="Width of Height bands.",
        )

        dbh_range = None
        height_range = None

    else:
        # Numeric filters (for all non-Tree-Count metrics, including Projection Exposure)

        # DBH FILTER
        if "dbh" in df.columns:
            dbh_vals = pd.to_numeric(df["dbh"], errors="coerce").dropna()
            if not dbh_vals.empty:
                min_dbh = int(np.floor(dbh_vals.min()))
                max_dbh = int(np.ceil(dbh_vals.max()))

                dbh_range = st.slider(
                    "**DBH filter [cm]**",
                    min_value=min_dbh,
                    max_value=max_dbh,
                    value=(min_dbh, max_dbh),
                    step=1,
                    help="Filter trees by DBH.",
                )
            else:
                dbh_range = None
        else:
            dbh_range = None

        # HEIGHT FILTER
        if "height" in df.columns:
            h_vals = pd.to_numeric(df["height"], errors="coerce").dropna()
            if not h_vals.empty:
                min_h = int(np.floor(h_vals.min()))
                max_h = int(np.ceil(h_vals.max()))

                height_range = st.slider(
                    "**Height filter [m]**",
                    min_value=min_h,
                    max_value=max_h,
                    value=(min_h, max_h),
                    step=1,
                    help="Filter trees by height.",
                )
            else:
                height_range = None
        else:
            height_range = None

        bin_size = None
        bin_size_h = None

# --------------------------------------------------------------------------------------
# Apply DBH / Height filters
# --------------------------------------------------------------------------------------

df_filt = df.copy()

# DBH filter
if dbh_range is not None and "dbh" in df_filt.columns:
    vals = pd.to_numeric(df_filt["dbh"], errors="coerce")
    df_filt = df_filt[(vals >= dbh_range[0]) & (vals <= dbh_range[1])]

# Height filter
if height_range is not None and "height" in df_filt.columns:
    vals_h = pd.to_numeric(df_filt["height"], errors="coerce")
    df_filt = df_filt[(vals_h >= height_range[0]) & (vals_h <= height_range[1])]

# --------------------------------------------------------------------------------------
# Slider for metric values (only for non-Tree-Count metrics)
# --------------------------------------------------------------------------------------

metric_range = None

if not is_tree_count and not df_filt.empty:
    metric_col_for_slider = value_mapping.get(y_label)

    with c_left:
        vals_all = None

        if y_label == "Projection Exposure":
            cols = []
            for c in ["projection_exposure", "projection_exposure_after_mgmt"]:
                if c in df_filt.columns:
                    cols.append(c)
            if cols:
                vals_all = pd.to_numeric(
                    df_filt[cols].values.ravel(), errors="coerce"
                )
        else:
            if metric_col_for_slider in df_filt.columns:
                vals_all = pd.to_numeric(
                    df_filt[metric_col_for_slider], errors="coerce"
                ).dropna()

        if vals_all is not None and len(vals_all) > 0:
            vmin = float(np.nanmin(vals_all))
            vmax = float(np.nanmax(vals_all))

            if vmin < vmax:
                # Projection Exposure: omezení na 0–100
                if y_label == "Projection Exposure":
                    vmin = max(0.0, vmin)
                    vmax = min(100.0, vmax)

                # Basal Area: 1 desetinné místo, krok 0.1
                if y_label == "Basal Area":
                    round_min = round(vmin, 1)
                    round_max = round(vmax, 1)
                    slider_step = 0.1
                    default_min = round_min
                    default_max = round_max
                else:
                    round_min = int(np.floor(vmin))
                    round_max = int(np.ceil(vmax))
                    slider_step = 1
                    default_min = round_min
                    default_max = round_max

                metric_range = st.slider(
                    "**Filter values:**",
                    min_value=round_min,
                    max_value=round_max,
                    value=(default_min, default_max),
                    step=slider_step,
                    help=(
                        "Filter range of metric values. "
                        "This affects which data are shown and how Y-axis is scaled."
                    ),
                )

# --------------------------------------------------------------------------------------
# Main render logic
# --------------------------------------------------------------------------------------

if df_filt.empty:
    st.info("No data after applying DBH/Height filters.")
else:
    if is_tree_count:
        # Tree Count always uses "Count" statistic
        y_stats = "Count"
        dummy_y_col = "dbh"  # any numeric placeholder – actual values are not used

        if x_mode == by_category:
            render_triple_by_category(
                df_filt, dummy_y_col, y_stats, color_mode, stacked
            )

        elif x_mode == by_dbh:
            render_triple_by_class(
                df_all=df_filt,
                value_col="dbh",
                bin_size=bin_size,
                unit_label="cm",
                color_mode=color_mode,
                y_col=dummy_y_col,
                y_stats=y_stats,
                stacked=stacked,
            )

        else:  # by_height
            render_triple_by_class(
                df_all=df_filt,
                value_col="height",
                bin_size=bin_size_h,
                unit_label="m",
                color_mode=color_mode,
                y_col=dummy_y_col,
                y_stats=y_stats,
                stacked=stacked,
            )

    else:
        metric_col = value_mapping[y_label]

        # Special handling for Projection Exposure – three dedicated violin plots
        if y_label == "Projection Exposure":
            render_projection_exposure_page(
                df=df_filt,
                color_mode=color_mode,
                slider_range=metric_range,
                y_label=y_label,
            )

        # Generic 3-panel violin for all other numeric metrics
        else:
            render_triple_violin(
                df_all=df_filt,
                value_col=metric_col,
                y_label=y_label,
                color_mode=color_mode,
                slider_range=metric_range,
            )
