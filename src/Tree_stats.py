import streamlit as st
import pandas as pd
import numpy as np
import math
import src.io_utils as iou
from plotly.subplots import make_subplots
import plotly.graph_objects as go

if "plot_uid" not in st.session_state:
    st.session_state.plot_uid = 0


def get_uid():
    st.session_state.plot_uid += 1
    return st.session_state.plot_uid


# --- colors & categories by mode ---
colorBySpp = st.session_state.Species
colorByMgmt = st.session_state.Management

st.markdown("### Explore tree statistics:")

# --- Data ---
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
    st.session_state.trees = iou.load_project_json(file_path)

df_raw: pd.DataFrame = st.session_state.trees.copy()

# ========== SETTINGS ==========
CHART_HEIGHT = 350  # chart height

# Columns excluded from Y-variable list (these are categorical/style drivers)
exclude_list = {"species", "speciesColorHex", "management_status", "managementColorHex"}

# --- After/Removed masks ---
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

# Standardize dtypes
if "species" in df.columns:
    df["species"] = df["species"].astype(str)
if "speciesColorHex" in df.columns:
    df["speciesColorHex"] = df["speciesColorHex"].astype(str)
if "management_status" in df.columns:
    df["management_status"] = df["management_status"].astype(str)
if "managementColorHex" in df.columns:
    df["managementColorHex"] = df["managementColorHex"].astype(str)


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


def _make_bins_labels(
    df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str
):
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
def _compute_binned_agg(
    df_sub: pd.DataFrame,
    value_col: str,  # DBH / height (bins on X)
    bins: np.ndarray,
    labels: list[str],
    hue_col: str,
    y_col: str,
    y_stats: str,
) -> pd.DataFrame:
    d = df_sub.copy()
    d[hue_col] = d[hue_col].astype(str)

    # assign bin
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
    if df_long is None or df_long.empty or "value" not in df_long.columns:
        return 0.0
    v = pd.to_numeric(df_long["value"], errors="coerce")
    if v.empty:
        return 0.0
    return float(v.max())


# ========== CATEGORY AGGREGATION (species / management) ==========
def _aggregate_by_hue(
    sub: pd.DataFrame, hue_col: str, y_col: str, y_stats: str
) -> pd.DataFrame:
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


# ========== BY CATEGORY (bars – pro Tree count) ==========
def render_triple_by_category(
    df_all: pd.DataFrame, y_col: str, y_stats: str, color_mode: str, stacked: bool
):
    """Kategorie na X; sloupce – Count nebo další agregace (používáme pro Tree count)."""
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
    y_upper = _y_upper_nice(float(y_max))

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
                    hovertemplate=f"{hue_col}: {cat}<br>{_aggfunc_name_for_hover(y_stats, y_col)}: %{{y:.2f}}<extra></extra>",
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

    y_label = "Count (rows)" if y_stats == "Count" else f"{y_stats} of {y_col}"
    fig.update_yaxes(title_text=y_label, row=1, col=1, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=2, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, range=[0, y_upper])

    st.plotly_chart(fig, use_container_width=True, key=f"category_{get_uid()}")


# ========== BY DBH / HEIGHT CLASS (bars – pro Tree count) ==========
def _panel_y_upper_for_long(long_df: pd.DataFrame, stacked: bool) -> float:
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

    # Y-range according to stacked/grouped
    y_max = max(
        _panel_y_upper_for_long(long_before, stacked),
        _panel_y_upper_for_long(long_after, stacked),
        _panel_y_upper_for_long(long_removed, stacked),
    )
    y_upper = _y_upper_nice(y_max)

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

    # volitelný styling titulků, pokud máš v session_state definovaný font
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
                    font=st.session_state.plot_title_font,
                )
                for ann in fig.layout.annotations
            ]
        )

    # traces
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
                    hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{_aggfunc_name_for_hover(y_stats, y_col)}: %{{y:.2f}}<extra></extra>",
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
    )
    fig.update_yaxes(title_text=None, row=1, col=2, tick0=0, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, range=[0, y_upper])

    st.plotly_chart(fig, use_container_width=True, key=f"class_{get_uid()}")


# ========== TRIPLE VIOLIN PLOT (pro všechny metriky kromě Tree count) ==========
def render_triple_violin(df_all: pd.DataFrame, value_col: str, color_mode: str):
    if value_col not in df_all.columns:
        st.warning(f"Missing column '{value_col}'.")
        return

    hue_col, categories, color_map, title_suffix = _hue_setup(df_all, color_mode)
    if not categories:
        st.info("No categories found for selected color mode.")
        return

    def _clean_numeric(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    panels = [
        (df_all, "Before"),
        (df_all[mask_after], "After"),
        (df_all[mask_removed], "Removed"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[p[1] for p in panels],
        horizontal_spacing=0.06,
    )

    max_val = 0
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
                    points=False,  # <<< odstranění bodů mimo violin
                    showlegend=(col_idx == 1),
                    hovertemplate=f"{hue_col}: {cat}<br>{value_col}: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )

    y_upper = _y_upper_nice(max_val)
    fig.update_yaxes(title_text=value_col, row=1, col=1, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=2, range=[0, y_upper])
    fig.update_yaxes(title_text=None, row=1, col=3, range=[0, y_upper])

    st.plotly_chart(fig, use_container_width=True, key=f"violin_{get_uid()}")


# ========== UI ==========

# Mapování názvů v UI -> sloupců v datech
VALUE_TREE_COUNT = "Tree Count"
value_mapping = {
    VALUE_TREE_COUNT: None,  # speciální hodnota (Count)
    "DBH": "dbh",
    "BA": "BasalArea_m2",
    "Volume": "Volume_m3",
    "Tree Height": "height",
    "Crown Base Height": "crown_base_height",
    "Crown Centroid Height": "crown_centroid_height",
    #    "crown diameter": "crown_diameter",
    "Crown Volume": "crown_volume",
    "Crown Surface Area": "crown_surface",
    "Horizontal Crown Projection Area": "horizontal_crown_proj",
    "Vertical Crown Projection Area": "vertical_crown_proj",
    "Crown Eccentricity": "crown_eccentricity",
    "Height-DBH Ratio": "heightXdbh",
}

# dostupné volby podle toho, co je v datech
value_options = []
for label, col in value_mapping.items():
    if label == VALUE_TREE_COUNT:
        value_options.append(label)
    else:
        if col in df.columns:
            value_options.append(label)

# "Plot by"
by_category = "Category"
by_dbh = "DBH"
by_height = "Height"
mode_options = [by_dbh, by_height, by_category]

# Defaultní bin velikosti (fixní, třídy jsou stále po pevné šířce)
DBH_BIN_DEFAULT = 5.0  # cm
H_BIN_DEFAULT = 2.0  # m

color_mode_default = colorBySpp

# Sloupce pro layout
c_left, c_left_empty, c_mid, c_right_empty, c_right = st.columns([3, 1, 4, 1, 3])

# --- LEFT: výběr proměnné ---
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
            "- **Tree count** is plotted as barblot.\n"
            "- Other variables are showed as violin plot."
        ),
    )

# --- MID: Plot by + Color by + Stacked ---
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
        # pro ostatní proměnné je graf vždy podle kategorie, DBH/Height jsou jen vizuálně „zakázané“
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
        help="Select Category (Species / Management) to color plots by.",
    )

    # Stacked dává smysl hlavně pro barploty (Tree count)
    stacked = (
        st.toggle(
            "**Stacked bars**",
            value=True,
            help=("Only for Tree count, switch between stacked and grouped mode."),
        )
        if is_tree_count
        else False
    )

# --- RIGHT: DBH / Height filtry ---
# --- RIGHT: DBH / Height filtry ---
with c_right:
    if is_tree_count:
        # -----------------------------
        # CLASS RANGE (Tree count ONLY)
        # -----------------------------

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
        # ---------------------------------
        # MIN–MAX FILTER (NON Tree count)
        # ---------------------------------

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

# --- aplikace filtrů na data ---
df_filt = df.copy()

# DBH filter
if dbh_range is not None and "dbh" in df_filt.columns:
    vals = pd.to_numeric(df_filt["dbh"], errors="coerce")
    df_filt = df_filt[(vals >= dbh_range[0]) & (vals <= dbh_range[1])]

# Height filter
if height_range is not None and "height" in df_filt.columns:
    vals_h = pd.to_numeric(df_filt["height"], errors="coerce")
    df_filt = df_filt[(vals_h >= height_range[0]) & (vals_h <= height_range[1])]

# --- Render podle vybrané proměnné a režimu ---
if df_filt.empty:
    st.info("No data after applying DBH/Height filters.")
else:
    if is_tree_count:
        y_stats = "Count"
        dummy_y_col = "dbh"

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
        render_triple_violin(
            df_all=df_filt, value_col=metric_col, color_mode=color_mode
        )
