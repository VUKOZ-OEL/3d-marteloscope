import streamlit as st
import pandas as pd
import numpy as np
import math
import src.io_utils as iou  # if unused, you can safely remove
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.i18n import t


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
# Stable UI IDs (do NOT use translated strings as IDs)
# --------------------------------------------------------------------------------------
COLOR_BY_SPECIES = "species"            # existing i18n key
COLOR_BY_MANAGEMENT = "management_label"  # existing i18n key

MODE_BY_CATEGORY = "mode_by_category"
MODE_BY_DBH = "mode_by_dbh"
MODE_BY_HEIGHT = "mode_by_height"

STAT_COUNT = "Count"
STAT_SUM = "Sum"
STAT_MEAN = "Mean"
STAT_MEDIAN = "Median"
STAT_MAX = "Max"
STAT_MIN = "Min"


# --------------------------------------------------------------------------------------
# Page header
# --------------------------------------------------------------------------------------
st.markdown(f"### {t('explore_tree_statistics')}")  # NEW KEY (návrh níže)

df_raw: pd.DataFrame = st.session_state.trees.copy()

CHART_HEIGHT = 350
exclude_list = {"species", "speciesColorHex", "management_status", "managementColorHex"}


# --------------------------------------------------------------------------------------
# Masks for "after management" vs "removed"
# --------------------------------------------------------------------------------------
df = df_raw.copy()
keep_status = {"Target tree", "Untouched"}

mask_after = df.get("management_status", pd.Series(False, index=df.index)).isin(keep_status)
mask_removed = (
    ~mask_after
    if "management_status" in df.columns
    else pd.Series(False, index=df.index)
)

for col in ["species", "speciesColorHex", "management_status", "managementColorHex"]:
    if col in df.columns:
        df[col] = df[col].astype(str)


# --------------------------------------------------------------------------------------
# Numeric helpers and "nice" axis scaling
# --------------------------------------------------------------------------------------
def _y_upper_nice(vmax: float, y_label: str = "", min_upper: float = 1.0) -> float:
    """
    Unified Y-axis rounding:
    - if vmax < 5 → round up to nearest 0.1
    - else → ceil
    """
    if vmax is None or np.isnan(vmax) or vmax <= 0:
        return min_upper

    if vmax < 5:
        return math.ceil(vmax * 10) / 10.0
    else:
        return math.ceil(vmax)


def auto_round_step(vmax: float) -> float:
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


def _make_bins_labels(df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str):
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


def _hue_setup(df_all: pd.DataFrame, color_mode_id: str):
    if color_mode_id == COLOR_BY_MANAGEMENT:
        cats, colors = _management_categories_and_colors(df_all)
        return "management_status", cats, colors, "(by Management)"
    else:
        cats, colors = _species_categories_and_colors(df_all)
        return "species", cats, colors, "(by Species)"


# --------------------------------------------------------------------------------------
# Generic stats helpers
# --------------------------------------------------------------------------------------
def _is_numeric_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    try:
        pd.to_numeric(s, errors="coerce")
        return True
    except Exception:
        return False


def _aggfunc_name_for_hover(stat: str, y_col: str) -> str:
    return t("tree_count") if stat == STAT_COUNT else t("stat_of_variable", stat=stat, var=y_col)  # NEW KEY


# --------------------------------------------------------------------------------------
# Binned aggregation (for class-based bar plots)
# --------------------------------------------------------------------------------------
def _compute_binned_agg(
    df_sub: pd.DataFrame,
    value_col: str,
    bins: np.ndarray,
    labels: list[str],
    hue_col: str,
    y_col: str,
    y_stats: str,
) -> pd.DataFrame:
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

    if y_stats == STAT_COUNT:
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

    if y_stats == STAT_SUM:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].sum().rename(columns={"_y_": "value"})
    elif y_stats == STAT_MEAN:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].mean().rename(columns={"_y_": "value"})
    elif y_stats == STAT_MEDIAN:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].median().rename(columns={"_y_": "value"})
    elif y_stats == STAT_MAX:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].max().rename(columns={"_y_": "value"})
    elif y_stats == STAT_MIN:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].min().rename(columns={"_y_": "value"})
    else:
        agg = d.groupby(["bin", hue_col], as_index=False)["_y_"].size().rename(columns={"_y_": "value"})

    return agg


def _panel_y_upper_for_long(long_df: pd.DataFrame, stacked: bool) -> float:
    if long_df is None or long_df.empty:
        return 0.0
    if stacked:
        s = long_df.groupby("bin")["value"].sum()
        return float(s.max()) if not s.empty else 0.0
    return float(pd.to_numeric(long_df["value"], errors="coerce").max())


# --------------------------------------------------------------------------------------
# Aggregation by category (species / management) for bar plots
# --------------------------------------------------------------------------------------
def _aggregate_by_hue(sub: pd.DataFrame, hue_col: str, y_col: str, y_stats: str) -> pd.DataFrame:
    if hue_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value"])

    if y_stats == STAT_COUNT:
        return (
            sub.groupby(hue_col, as_index=False)
            .agg(value=(hue_col, "size"))
            .assign(value=lambda x: x["value"].astype(float))
        )

    if y_col not in sub.columns:
        return pd.DataFrame(columns=[hue_col, "value", "__error__"])

    s = sub[y_col]
    if not _is_numeric_like(s):
        return pd.DataFrame(columns=[hue_col, "value", "__error__"]).assign(__error__=True)

    x = pd.to_numeric(s, errors="coerce")
    tmp = sub.copy()
    tmp["_y_"] = x

    if y_stats == STAT_SUM:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "sum"))
    elif y_stats == STAT_MEAN:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "mean"))
    elif y_stats == STAT_MEDIAN:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "median"))
    elif y_stats == STAT_MAX:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "max"))
    elif y_stats == STAT_MIN:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=("_y_", "min"))
    else:
        agg_df = tmp.groupby(hue_col, as_index=False).agg(value=(hue_col, "size"))

    return agg_df


def _ensure_all_categories(sub_counts: pd.DataFrame, hue_col: str, categories: list[str], color_lookup: dict) -> pd.DataFrame:
    base = pd.DataFrame({hue_col: categories})
    sub = base.merge(
        sub_counts.assign(**{hue_col: sub_counts[hue_col].astype(str).str.strip()})
        if not sub_counts.empty
        else base.assign(value=np.nan),
        on=hue_col,
        how="left",
    )
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce").fillna(0.0).astype(float)
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
def render_triple_by_category(df_all: pd.DataFrame, y_col: str, y_stats: str, color_mode_id: str, stacked: bool):
    hue_col, categories, color_map, _ = _hue_setup(df_all, color_mode_id)
    if not categories:
        st.info(t("warn_no_categories"))
        return

    before_df = _aggregate_by_hue(df_all, hue_col, y_col, y_stats)
    after_df = _aggregate_by_hue(df_all[mask_after], hue_col, y_col, y_stats)
    removed_df = _aggregate_by_hue(df_all[mask_removed], hue_col, y_col, y_stats)

    any_error = any(("__error__" in d.columns and not d.empty) for d in [before_df, after_df, removed_df])
    if any_error and y_stats != STAT_COUNT:
        st.warning(t("warn_non_numeric_stat", stat=y_stats, var=y_col))  # NEW KEY

    before_df = _ensure_all_categories(before_df, hue_col, categories, color_map)
    after_df = _ensure_all_categories(after_df, hue_col, categories, color_map)
    removed_df = _ensure_all_categories(removed_df, hue_col, categories, color_map)

    y_max = max(before_df["value"].max(), after_df["value"].max(), removed_df["value"].max())
    y_upper = _y_upper_nice(float(y_max), t("tree_count"))

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=(st.session_state.Before, st.session_state.After, st.session_state.Removed),
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
                        f"{t('group_label')}: {cat}<br>"
                        f"{t('value_label')}: %{{y:.2f}}"
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
        fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=categories, row=1, col=c)

    y_label = t("tree_count") if y_stats == STAT_COUNT else t("stat_of_variable", stat=y_stats, var=y_col)
    fig.update_yaxes(title_text=y_label, row=1, col=1, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=2, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")

    st.plotly_chart(fig, use_container_width=True, key=f"category_{get_uid()}")


# --------------------------------------------------------------------------------------
# Triple bar plot by DBH / height classes
# --------------------------------------------------------------------------------------
def render_triple_by_class(
    df_all: pd.DataFrame,
    value_col: str,
    bin_size: float,
    unit_label: str,
    color_mode_id: str,
    y_col: str,
    y_stats: str,
    stacked: bool,
):
    if value_col not in df_all.columns:
        st.warning(t("warn_missing_column", column=value_col))
        return

    hue_col, categories, color_map, _ = _hue_setup(df_all, color_mode_id)
    if not categories:
        st.info(t("warn_no_categories"))
        return

    bins, labels = _make_bins_labels(df_all, value_col, bin_size, unit_label)
    if bins is None:
        st.info(t("warn_no_valid_values", column=value_col))  # NEW KEY
        return

    long_before = _compute_binned_agg(df_all, value_col, bins, labels, hue_col, y_col, y_stats)
    long_after = _compute_binned_agg(df_all[mask_after], value_col, bins, labels, hue_col, y_col, y_stats)
    long_removed = _compute_binned_agg(df_all[mask_removed], value_col, bins, labels, hue_col, y_col, y_stats)

    def has_error(df_long):
        return ("__error__" in df_long.columns) and (df_long["__error__"].any() if not df_long.empty else True)

    if y_stats != STAT_COUNT and (has_error(long_before) or has_error(long_after) or has_error(long_removed)):
        st.warning(t("warn_non_numeric_stat", stat=y_stats, var=y_col))

    y_max = max(
        _panel_y_upper_for_long(long_before, stacked),
        _panel_y_upper_for_long(long_after, stacked),
        _panel_y_upper_for_long(long_removed, stacked),
    )
    y_upper = _y_upper_nice(y_max, t("tree_count"))

    def color_for(cat):
        return color_map.get(cat, "#AAAAAA")

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=(st.session_state.Before, st.session_state.After, st.session_state.Removed),
        horizontal_spacing=0.06,
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
                        f"{t('group_label')}: {cat}<br>"
                        f"{t('value_label')}: %{{y:.2f}}"
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
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=labels)

    fig.update_yaxes(title_text=_aggfunc_name_for_hover(y_stats, y_col), row=1, col=1, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=2, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, range=[0, y_upper], showticklabels=True, tickmode="auto")

    st.plotly_chart(fig, use_container_width=True, key=f"class_{get_uid()}")



# --------------------------------------------------------------------------------------
# Triple violin plot (Before / After / Removed) for continuous metrics
# --------------------------------------------------------------------------------------
def render_triple_violin(
    df_all: pd.DataFrame,
    value_col: str,
    y_label: str,
    color_mode: str,
    slider_range=None,
):
    """Render 3 side-by-side violin plots (Before/After/Removed) split by species/management."""
    if value_col not in df_all.columns:
        st.warning(t("warn_missing_column", column=value_col))
        return

    hue_col, categories, color_map, _ = _hue_setup(df_all, color_mode)
    if not categories:
        st.info(t("warn_no_categories"))
        return

    # Prepare numeric values + optional slider filtering
    def _prep(sub: pd.DataFrame) -> pd.DataFrame:
        d = sub.copy()
        d[hue_col] = d[hue_col].astype(str)
        d["_val_"] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna(subset=["_val_"])
        if slider_range is not None:
            lo, hi = slider_range
            d = d[(d["_val_"] >= float(lo)) & (d["_val_"] <= float(hi))]
        return d

    before = _prep(df_all)
    after = _prep(df_all[mask_after])
    removed = _prep(df_all[mask_removed])

    if before.empty and after.empty and removed.empty:
        st.warning(t("warn_no_data_for_filters"))
        return

    # Axis range
    all_vals = pd.concat([before["_val_"], after["_val_"], removed["_val_"]], ignore_index=True)
    all_vals = pd.to_numeric(all_vals, errors="coerce").dropna()

    if all_vals.empty:
        st.warning(t("warn_no_valid_values", column=value_col))
        return

    if slider_range is not None:
        y_lower, y_upper = float(slider_range[0]), float(slider_range[1])
    else:
        y_lower = float(all_vals.min())
        y_upper = float(all_vals.max())
        # keep some room / nicer rounding
        y_upper = _y_upper_nice(y_upper, y_label, min_upper=y_upper if y_upper > 0 else 1.0)

        # for typical forestry metrics (>=0), start from 0 when all values are positive
        if y_lower >= 0:
            y_lower = 0.0
        else:
            y_lower = math.floor(y_lower)

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=(st.session_state.Before, st.session_state.After, st.session_state.Removed),
        horizontal_spacing=0.06,
    )

    def _color_for(cat: str) -> str:
        c = color_map.get(cat, "#AAAAAA")
        return c if isinstance(c, str) and c.strip() else "#AAAAAA"

    def _add_panel(panel_df: pd.DataFrame, col: int, show_legend_for: set[str]):
        if panel_df.empty:
            # plotly doesn't like completely empty subplots with categorical axes; add a hidden dummy trace
            fig.add_trace(
                go.Violin(x=[""], y=[np.nan], showlegend=False, hoverinfo="skip", visible=False),
                row=1,
                col=col,
            )
            return

        for cat in categories:
            vals = panel_df.loc[panel_df[hue_col] == cat, "_val_"]
            if vals.empty:
                continue
            fig.add_trace(
                go.Violin(
                    x=[cat] * len(vals),
                    y=vals,
                    name=str(cat),
                    legendgroup=str(cat),
                    showlegend=(cat in show_legend_for),
                    box_visible=True,
                    meanline_visible=True,
                    points="outliers",
                    marker_color=_color_for(cat),
                    line_color=_color_for(cat),
                    hovertemplate=(
                        f"{t('group_label')}: {cat}<br>"
                        f"{t('value_label')}: %{{y:.2f}}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    _add_panel(before, col=1, show_legend_for=set(categories))
    _add_panel(after, col=2, show_legend_for=set())
    _add_panel(removed, col=3, show_legend_for=set())

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=95),
        violinmode="group",
        legend=dict(orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5),
        showlegend=True,
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

    fig.update_yaxes(title_text=y_label, row=1, col=1, range=[y_lower, y_upper], tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=2, range=[y_lower, y_upper], tickmode="auto")
    fig.update_yaxes(title_text=None, row=1, col=3, range=[y_lower, y_upper], tickmode="auto")

    st.plotly_chart(fig, use_container_width=True, key=f"violin_{get_uid()}")


# --------------------------------------------------------------------------------------
# UI: metric mapping and controls (use i18n keys as stable IDs)
# --------------------------------------------------------------------------------------
VALUE_TREE_COUNT = "tree_count"  # exists

METRIC_DBH = "dbh"
METRIC_BASAL_AREA = "basal_area_m2"
METRIC_VOLUME = "volume"
METRIC_TREE_HEIGHT = "tree_height"
METRIC_CROWN_BASE_HEIGHT = "crown_base_height"
METRIC_CROWN_CENTROID_HEIGHT = "crown_centroid_height"
METRIC_CROWN_VOLUME = "crown_volume_m3"
METRIC_CROWN_SURFACE = "crown_surface"
METRIC_HORIZONTAL_CROWN_PROJ = "horizontal_crown_proj"
METRIC_VERTICAL_CROWN_PROJ = "vertical_crown_proj"
METRIC_CROWN_ECC = "crown_eccentricity"
METRIC_HEIGHT_DBH = "height_dbh_ratio"
METRIC_PROJ_EXP = "projection_exposure"

value_mapping = {
    VALUE_TREE_COUNT: None,
    METRIC_DBH: "dbh",
    METRIC_BASAL_AREA: "basal_area_m2",
    METRIC_VOLUME: "Volume_m3",
    METRIC_TREE_HEIGHT: "height",
    METRIC_CROWN_BASE_HEIGHT: "crown_base_height",
    METRIC_CROWN_CENTROID_HEIGHT: "crown_centroid_height",
    METRIC_CROWN_VOLUME: "crown_volume",
    METRIC_CROWN_SURFACE: "crown_surface",
    METRIC_HORIZONTAL_CROWN_PROJ: "horizontal_crown_proj",
    METRIC_VERTICAL_CROWN_PROJ: "vertical_crown_proj",
    METRIC_CROWN_ECC: "crown_eccentricity",
    METRIC_HEIGHT_DBH: "heightXdbh",
    METRIC_PROJ_EXP: "projection_exposure",
}

# units
y_units = {
    METRIC_DBH: t("unit_cm"),
    METRIC_BASAL_AREA: "m²",
    METRIC_VOLUME: "m³",
    METRIC_TREE_HEIGHT: t("unit_m"),
    METRIC_CROWN_BASE_HEIGHT: t("unit_m"),
    METRIC_CROWN_CENTROID_HEIGHT: t("unit_m"),
    METRIC_CROWN_VOLUME: "m³",
    METRIC_CROWN_SURFACE: "m²",
    METRIC_HORIZONTAL_CROWN_PROJ: "m²",
    METRIC_VERTICAL_CROWN_PROJ: "m²",
    METRIC_CROWN_ECC: t("unit_m"),
    METRIC_HEIGHT_DBH: "",
    METRIC_PROJ_EXP: "%",
}

def make_y_label(metric_id: str) -> str:
    unit = y_units.get(metric_id, "")
    base = t(metric_id) if isinstance(metric_id, str) else str(metric_id)
    return f"{base} [{unit}]" if unit else base


# Only show metrics actually present in df
value_options = []
for mid, col in value_mapping.items():
    if mid == VALUE_TREE_COUNT:
        value_options.append(mid)
    else:
        if col in df.columns:
            value_options.append(mid)

mode_options = [MODE_BY_DBH, MODE_BY_HEIGHT, MODE_BY_CATEGORY]

c_left, c_left_empty, c_mid, c_right_empty, c_right = st.columns([3, 1, 4, 1, 3])

# LEFT: Value selection
with c_left:
    default_index = value_options.index(VALUE_TREE_COUNT) if VALUE_TREE_COUNT in value_options else 0
    metric_id = st.selectbox(
        f"**{t('values_to_plot')}**",  # NEW KEY
        options=value_options,
        index=default_index,
        format_func=lambda k: t(k),
        help=t("values_to_plot_help"),  # NEW KEY
    )

# MID: Plot by / Color by / Stacked toggle
with c_mid:
    is_tree_count = metric_id == VALUE_TREE_COUNT

    if is_tree_count:
        x_mode = st.segmented_control(
            f"**{t('plot_by')}**",  # exists
            options=mode_options,
            default=MODE_BY_DBH,
            width="stretch",
            format_func=lambda k: {
                MODE_BY_DBH: "DBH",
                MODE_BY_HEIGHT: t("tree_height"),
                MODE_BY_CATEGORY: t("category"),
            }.get(k, k),
            help=t("plot_by_help_count"),  # NEW KEY
        )
    else:
        x_mode = MODE_BY_CATEGORY
        st.segmented_control(
            f"**{t('plot_by')}**",
            options=[MODE_BY_CATEGORY],
            default=MODE_BY_CATEGORY,
            disabled=True,
            width="stretch",
            format_func=lambda k: t("category"),
            help=t("plot_by_help_category_only"),  # NEW KEY
        )

    color_mode_id = st.segmented_control(
        f"**{t('color_by')}**",  # exists
        options=[COLOR_BY_SPECIES, COLOR_BY_MANAGEMENT],
        default=COLOR_BY_SPECIES,
        width="stretch",
        format_func=lambda k: t(k),
        help=t("color_by_help"),  # NEW KEY
    )

    stacked = (
        st.toggle(
            f"**{t('stacked_bars')}**",  # NEW KEY
            value=True,
            help=t("stacked_bars_help"),  # NEW KEY
        )
        if is_tree_count
        else False
    )

# RIGHT: DBH / Height controls
with c_right:
    if is_tree_count:
        dbh_bins = [5, 10, 20]
        h_bins = [2, 5, 10]

        bin_size = st.select_slider(
            f"**{t('dbh_class_range')}**",  # NEW KEY
            options=dbh_bins,
            value=10,
            help=t("dbh_class_range_help"),  # NEW KEY
        )
        bin_size_h = st.select_slider(
            f"**{t('height_class_range')}**",  # NEW KEY
            options=h_bins,
            value=5,
            help=t("height_class_range_help"),  # NEW KEY
        )

        dbh_range = None
        height_range = None
    else:
        # DBH FILTER
        dbh_range = None
        if "dbh" in df.columns:
            dbh_vals = pd.to_numeric(df["dbh"], errors="coerce").dropna()
            if not dbh_vals.empty:
                min_dbh = int(np.floor(dbh_vals.min()))
                max_dbh = int(np.ceil(dbh_vals.max()))
                dbh_range = st.slider(
                    f"**{t('dbh_filter')}**",  # exists
                    min_value=min_dbh,
                    max_value=max_dbh,
                    value=(min_dbh, max_dbh),
                    step=1,
                    help=t("dbh_filter_help"),  # NEW KEY
                )

        # HEIGHT FILTER
        height_range = None
        if "height" in df.columns:
            h_vals = pd.to_numeric(df["height"], errors="coerce").dropna()
            if not h_vals.empty:
                min_h = int(np.floor(h_vals.min()))
                max_h = int(np.ceil(h_vals.max()))
                height_range = st.slider(
                    f"**{t('height_filter')}**",
                    min_value=min_h,
                    max_value=max_h,
                    value=(min_h, max_h),
                    step=1,
                    help=t("height_filter_help"),  # NEW KEY
                )

        bin_size = None
        bin_size_h = None


# --------------------------------------------------------------------------------------
# Apply DBH / Height filters
# --------------------------------------------------------------------------------------
df_filt = df.copy()

if dbh_range is not None and "dbh" in df_filt.columns:
    vals = pd.to_numeric(df_filt["dbh"], errors="coerce")
    df_filt = df_filt[(vals >= dbh_range[0]) & (vals <= dbh_range[1])]

if height_range is not None and "height" in df_filt.columns:
    vals_h = pd.to_numeric(df_filt["height"], errors="coerce")
    df_filt = df_filt[(vals_h >= height_range[0]) & (vals_h <= height_range[1])]


# --------------------------------------------------------------------------------------
# Slider for metric values (only for non-Tree-Count metrics)
# --------------------------------------------------------------------------------------
metric_range = None

if (not is_tree_count) and (not df_filt.empty):
    metric_col_for_slider = value_mapping.get(metric_id)

    with c_left:
        vals_all = None

        if metric_id == METRIC_PROJ_EXP:
            cols = [c for c in ["projection_exposure", "projection_exposure_after_mgmt"] if c in df_filt.columns]
            if cols:
                vals_all = pd.to_numeric(df_filt[cols].values.ravel(), errors="coerce")
        else:
            if metric_col_for_slider in df_filt.columns:
                vals_all = pd.to_numeric(df_filt[metric_col_for_slider], errors="coerce").dropna()

        if vals_all is not None and len(vals_all) > 0:
            vmin = float(np.nanmin(vals_all))
            vmax = float(np.nanmax(vals_all))

            if vmin < vmax:
                if metric_id == METRIC_PROJ_EXP:
                    vmin = max(0.0, vmin)
                    vmax = min(100.0, vmax)

                if metric_id == METRIC_BASAL_AREA:
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
                    f"**{t('filter_values')}**",  # NEW KEY
                    min_value=round_min,
                    max_value=round_max,
                    value=(default_min, default_max),
                    step=slider_step,
                    help=t("filter_values_help"),  # NEW KEY
                )


# --------------------------------------------------------------------------------------
# Main render logic
# --------------------------------------------------------------------------------------
if df_filt.empty:
    st.warning(t("warn_no_data_for_filters"))  # exists
else:
    if is_tree_count:
        y_stats = STAT_COUNT
        dummy_y_col = "dbh"

        if x_mode == MODE_BY_CATEGORY:
            render_triple_by_category(df_filt, dummy_y_col, y_stats, color_mode_id, stacked)

        elif x_mode == MODE_BY_DBH:
            render_triple_by_class(
                df_all=df_filt,
                value_col="dbh",
                bin_size=bin_size,
                unit_label=t("unit_cm"),
                color_mode_id=color_mode_id,
                y_col=dummy_y_col,
                y_stats=y_stats,
                stacked=stacked,
            )

        else:  # MODE_BY_HEIGHT
            render_triple_by_class(
                df_all=df_filt,
                value_col="height",
                bin_size=bin_size_h,
                unit_label=t("unit_m"),
                color_mode_id=color_mode_id,
                y_col=dummy_y_col,
                y_stats=y_stats,
                stacked=stacked,
            )

    else:
        metric_col = value_mapping[metric_id]

        if metric_id == METRIC_PROJ_EXP:
            # keep your existing special renderer if you want; not shown here to keep changes scoped
            st.info(t("warn_projection_exposure_special"))  # NEW KEY placeholder
        else:
            # NOTE: you still need your render_triple_violin function here (unchanged),
            # but it must accept metric_id/y_label -> we keep your existing signature
            render_triple_violin(
                df_all=df_filt,
                value_col=metric_col,
                y_label=t(metric_id),
                color_mode=color_mode_id,
                slider_range=metric_range,
            )
