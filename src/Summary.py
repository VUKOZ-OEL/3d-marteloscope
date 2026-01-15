import streamlit as st
import pandas as pd
import numpy as np
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.i18n import t

# ---------- DATA ----------
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- HLAVIČKA ----------
st.markdown(f"### {t('plot_summary')}")

# ---------- SPOLEČNÉ ----------
CHART_HEIGHT = 360
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
    area_ha = float(plot_info["size_ha"].iloc[0])
    if not np.isfinite(area_ha) or area_ha <= 0:
        area_ha = 1.0
except Exception:
    area_ha = 1.0

# canopy cover (%) z projekce korun (surfaceAreaProjection v m2)
if "surfaceAreaProjection" in df.columns:
    sap = pd.to_numeric(df["surfaceAreaProjection"], errors="coerce").fillna(0.0)
    df["canopy_cover_pct"] = (sap / (area_ha * 10_000.0)) * 100.0
else:
    df["canopy_cover_pct"] = np.nan


# ---------- STABILNÍ IDs (options jako keys) ----------
STATUS_BEFORE = "label_before"
STATUS_AFTER = "label_after"
STATUS_REMOVED = "label_removed"

COLOR_BY_SPECIES = "species"          # existující key ve slovníku
COLOR_BY_MANAGEMENT = "management_label"  # existující key ve slovníku

METRIC_TREE_COUNT = "metric_tree_count"
METRIC_VOLUME = "metric_volume_m3"
METRIC_BASAL_AREA = "metric_basal_area_m2"
METRIC_CANOPY_COVER = "metric_canopy_cover_pct"


def _make_masks(d: pd.DataFrame):
    keep_status = {"Target tree", "Untouched"}
    mask_after = d.get("management_status", pd.Series(False, index=d.index)).isin(keep_status)
    mask_removed = ~mask_after if "management_status" in d.columns else pd.Series(False, index=d.index)
    mask_before = pd.Series(True, index=d.index)
    return {STATUS_BEFORE: mask_before, STATUS_AFTER: mask_after, STATUS_REMOVED: mask_removed}


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
    """Barvy ze sloupce managementColorHex pro všechny management_status, chybějící -> šedá."""
    if "management_status" not in d.columns or "managementColorHex" not in d.columns:
        return {}
    tt = d.assign(
        management_status=lambda x: x["management_status"].astype(str),
        managementColorHex=lambda x: x["managementColorHex"].astype(str),
    )
    cmap = tt.groupby("management_status")["managementColorHex"].first().to_dict()
    for k, v in list(cmap.items()):
        if not isinstance(v, str) or not v.strip():
            cmap[k] = "#AAAAAA"
    return cmap


def _make_bins_labels(df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str):
    vals = pd.to_numeric(df_all.get(value_col), errors="coerce").dropna()
    if vals.empty:
        return None, None
    vmin = float(np.floor(vals.min() / bin_size) * bin_size)
    vmax = float(np.ceil(vals.max() / bin_size) * bin_size)
    if vmax <= vmin:
        vmax = vmin + bin_size
    bins = np.arange(vmin, vmax + bin_size, bin_size, dtype=float)
    labels = [f"{int(b)}–{int(b + bin_size)} {unit_label}" for b in bins[:-1]]
    return bins, labels


# ---------- OVLÁDÁNÍ ----------
c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 3, 0.25, 4, 0.25, 2, 0.5])

with c2:
    dist_mode = st.segmented_control(
        f"**{t('show_data_for')}**",
        options=[STATUS_BEFORE, STATUS_AFTER, STATUS_REMOVED],
        format_func=lambda k: t(k),
        default=STATUS_BEFORE,
        width="stretch",
    )

with c4:
    sum_metric_id = st.segmented_control(
        f"**{t('sum_values_by')}**",
        options=[METRIC_TREE_COUNT, METRIC_VOLUME, METRIC_BASAL_AREA, METRIC_CANOPY_COVER],
        format_func=lambda k: t(k),
        default=METRIC_TREE_COUNT,
        width="stretch",
    )

with c6:
    color_mode = st.segmented_control(
        f"**{t('color_by')}**",
        options=[COLOR_BY_SPECIES, COLOR_BY_MANAGEMENT],
        format_func=lambda k: t(k),
        default=COLOR_BY_SPECIES,
        width="stretch",
    )


def _metric_meta(metric_id: str):
    """
    Vrátí:
      (value_col | None pro počty,
       y_title,
       unit_suffix,
       pie_value_label,
       unit_disp)
    """
    if metric_id == METRIC_TREE_COUNT:
        return None, t("trees"), t("trees_per_ha"), t("trees"), "trees"

    if metric_id == METRIC_VOLUME:
        return "volume", f"{t('value_volume')} (m³)", t("m3_per_ha"), f"{t('value_volume')} (m³)", "m³"

    if metric_id == METRIC_BASAL_AREA:
        return "basal_area_m2", f"{t('basal_area')} (m²)", t("m2_per_ha"), f"{t('basal_area')} (m²)", "m²"

    if metric_id == METRIC_CANOPY_COVER:
        return "canopy_cover_pct", t("metric_canopy_cover_pct"), t("unit_percent"), t("metric_canopy_cover"), "%"

    return None, t("trees"), t("trees_per_ha"), t("trees"), "trees"


value_col, y_title, unit_suffix, pie_value_label, unit_disp = _metric_meta(sum_metric_id)

masks = _make_masks(df)
mask = masks.get(dist_mode, pd.Series(True, index=df.index))
df_sel = df[mask].copy()


# ---------- 1) PIE + DBH + HEIGHT (sdílená legenda) ----------
def render_three_panel_with_shared_legend(
    df_all: pd.DataFrame,
    df_sub: pd.DataFrame,
    value_col: str | None,
    color_mode_key: str,
    y_title: str,
    unit_suffix: str,
    pie_value_label: str,
    unit_disp: str,
):
    # --- hue / kategorie / barvy
    if color_mode_key == COLOR_BY_MANAGEMENT:
        hue_col = "management_status"
        categories = (
            pd.Index(df_all[hue_col].astype(str).dropna().unique()).tolist()
            if hue_col in df_all.columns
            else []
        )
        color_map = _management_colors(df_all)
        color_map = {c: color_map.get(c, "#AAAAAA") for c in categories}
    else:
        hue_col = "species"
        categories = sorted(
            df_all.get("species", pd.Series([], dtype=str))
            .astype(str)
            .dropna()
            .unique()
            .tolist()
        )
        color_map = _species_colors(df_all)
        color_map = {c: color_map.get(c, "#AAAAAA") for c in categories}

    d = df_sub.copy()
    if hue_col not in d.columns:
        st.warning(t("warn_missing_column", column=hue_col))
        return
    d[hue_col] = d[hue_col].astype(str).str.strip()

    # --- PIE agregace + Σ do středu
    if value_col is None:
        pie_agg = d.groupby(hue_col, as_index=False).agg(value=(hue_col, "size"))
        total_raw = int(pie_agg["value"].sum())
        total_text = f"Σ =<br><b>{total_raw}</b><br>{unit_disp}"
        hover_value_token = "%{value:.0f}"
    else:
        if value_col not in d.columns:
            st.warning(t("warn_missing_column", column=value_col))
            return
        pie_agg = d.groupby(hue_col, as_index=False).agg(value=(value_col, "sum"))
        total_raw = float(pie_agg["value"].sum())
        total_text = f"Σ =<br><b>{total_raw:,.1f}</b><br>{unit_disp}".replace(",", " ")
        hover_value_token = "%{value:.1f}"

    if categories:
        pie_agg = pie_agg.set_index(hue_col).reindex(categories).fillna(0).reset_index()
    else:
        categories = pie_agg[hue_col].tolist()

    # --- DBH / HEIGHT biny
    dbh_bins, dbh_labels = _make_bins_labels(df_all, "dbh", 10, t("unit_cm"))
    if dbh_bins is None:
        dbh_bins, dbh_labels = np.array([0, 10]), [f"0–10 {t('unit_cm')}"]

    if "height" in df_all.columns:
        h_bins, h_labels = _make_bins_labels(df_all, "height", 5, t("unit_m"))
        if h_bins is None:
            h_bins, h_labels = np.array([0, 5]), [f"0–5 {t('unit_m')}"]
    else:
        h_bins, h_labels = np.array([0, 5]), [f"0–5 {t('unit_m')}"]

    # --- Long tabulky
    def long_binned(
        df_in: pd.DataFrame,
        base_col: str,
        bins: np.ndarray,
        labels: list[str],
        hue: str,
        value_col: str | None,
    ) -> pd.DataFrame:
        tt = df_in.copy()
        tt[hue] = tt[hue].astype(str)
        vals = pd.to_numeric(tt[base_col], errors="coerce")
        cats = pd.Categorical(
            pd.cut(vals, bins=bins, labels=labels, include_lowest=True, right=False, ordered=True),
            categories=labels,
            ordered=True,
        )
        tt = tt.assign(bin=cats).dropna(subset=["bin"])
        if tt.empty:
            return pd.DataFrame(columns=["bin", hue, "value"])
        if value_col is None:
            pv = tt.pivot_table(index="bin", columns=hue, aggfunc="size", fill_value=0)
        else:
            tt["weight"] = pd.to_numeric(tt[value_col], errors="coerce").fillna(0.0)
            pv = tt.pivot_table(index="bin", columns=hue, values="weight", aggfunc="sum", fill_value=0.0)
        long = pv.stack().rename("value").reset_index()
        long["bin"] = long["bin"].astype(str)
        return long

    dbh_long = long_binned(df_sub, "dbh", dbh_bins, dbh_labels, hue_col, value_col)
    height_long = (
        long_binned(df_sub, "height", h_bins, h_labels, hue_col, value_col)
        if "height" in df_sub.columns
        else pd.DataFrame(columns=["bin", hue_col, "value"])
    )

    # --- Y-osa: horní hranice
    def upper_from_long(long_df: pd.DataFrame, labels: list[str]) -> float:
        if long_df.empty:
            return 10
        totals = long_df.groupby("bin")["value"].sum().reindex(labels).fillna(0)
        maxv = float(totals.max())
        if maxv <= 0:
            return 10
        step = 10
        return math.ceil(maxv / step) * step

    dbh_y_upper = upper_from_long(
        long_binned(df, "dbh", dbh_bins, dbh_labels, hue_col, value_col),
        dbh_labels,
    )
    h_y_upper = upper_from_long(
        long_binned(df, "height", h_bins, h_labels, hue_col, value_col),
        h_labels,
    )

    # --- Subplots: pie | dbh | height
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(t("stand_composition"), t("in_dbh_class"), t("in_height_class")),
        horizontal_spacing=0.06,
    )

    if hasattr(st.session_state, "plot_title_font"):
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

    # ---------------- PIE ----------------
    pie_plot = pie_agg[pie_agg["value"] > 0].copy()
    no_data = pie_plot.empty

    if no_data:
        pie_labels = [t("no_data")]
        pie_values = [1]
        pie_colors = ["#EEEEEE"]
        textinfo = "none"
        texttemplate = None
        pie_text = None
        pie_hover = ""
    else:
        pie_labels = pie_plot[hue_col].astype(str).tolist()
        pie_values = pie_plot["value"].astype(float).tolist()
        pie_colors = [color_map.get(c, "#AAAAAA") for c in pie_plot[hue_col]]

        textinfo = "text"
        texttemplate = "%{percent:.1%}"

        base_hover = (
            f"{hue_col}: %{{label}}<br>"
            f"{pie_value_label}: {hover_value_token} {unit_disp}"
            "<extra></extra>"
        )

        pie_text = None
        pie_hover_list = [base_hover] * len(pie_values)

        if value_col == "canopy_cover_pct":
            total_cover = float(np.nansum(pie_values))
            uncovered = max(0.0, 100.0 - total_cover)

            if uncovered > 1e-6:
                pie_labels.append(t("uncovered"))
                pie_values.append(uncovered)
                pie_colors.append("#FFFFFF")
                pie_hover_list.append("<extra></extra>")

            total_for_percent = float(np.nansum(pie_values)) or 1.0
            pct_text = []
            for lab, val in zip(pie_labels, pie_values):
                if lab == t("uncovered"):
                    pct_text.append("")
                else:
                    pct_text.append(f"{(val / total_for_percent) * 100.0:.1f}%")

            textinfo = "text"
            texttemplate = None
            pie_text = pct_text
            pie_hover = pie_hover_list
        else:
            pie_hover = base_hover

    fig.add_trace(
        go.Pie(
            labels=pie_labels,
            values=pie_values,
            hole=0.50,
            marker=dict(colors=pie_colors, line=dict(color="#FFFFFF", width=1)),
            textinfo=textinfo,
            texttemplate=texttemplate,
            text=pie_text,
            textposition="inside",
            insidetextorientation="radial",
            textfont=dict(size=11),
            hovertemplate=pie_hover,
            showlegend=False,
            sort=False,
        ),
        row=1,
        col=1,
    )

    # --- středový Σ
    pie_trace = fig.data[-1]
    if hasattr(pie_trace, "domain") and pie_trace.domain:
        cx = (pie_trace.domain.x[0] + pie_trace.domain.x[1]) / 2
        cy = (pie_trace.domain.y[0] + pie_trace.domain.y[1]) / 2
    else:
        cx, cy = 0.17, 0.5

    fig.add_annotation(
        x=cx,
        y=cy,
        xref="paper",
        yref="paper",
        text=(total_text if not no_data else f"Σ = 0 {unit_disp}"),
        showarrow=False,
        font=dict(size=26, color="black"),
        xanchor="center",
        yanchor="middle",
    )

    # --- DBH
    for cat in categories:
        y_vals = (
            dbh_long[dbh_long[hue_col] == cat]
            .set_index("bin")
            .reindex(dbh_labels)["value"]
            .fillna(0)
            .tolist()
        )
        fig.add_trace(
            go.Bar(
                x=dbh_labels,
                y=y_vals,
                name=cat,
                marker_color=color_map.get(cat, "#AAAAAA"),
                legendgroup=cat,
                showlegend=True,
                hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # --- HEIGHT
    if not height_long.empty:
        for cat in categories:
            y_vals = (
                height_long[height_long[hue_col] == cat]
                .set_index("bin")
                .reindex(h_labels)["value"]
                .fillna(0)
                .tolist()
            )
            fig.add_trace(
                go.Bar(
                    x=h_labels,
                    y=y_vals,
                    name=cat,
                    marker_color=color_map.get(cat, "#AAAAAA"),
                    legendgroup=cat,
                    showlegend=False,
                    hovertemplate=f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y}}<extra></extra>",
                ),
                row=1,
                col=3,
            )

    fig.update_layout(
        barmode="stack",
        height=CHART_HEIGHT + 80,
        margin=dict(l=10, r=10, t=60, b=120),
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=dbh_labels, row=1, col=2)
    fig.update_xaxes(title_text=None, tickangle=45, categoryorder="array", categoryarray=h_labels, row=1, col=3)

    if value_col is None:
        fig.update_yaxes(title_text=y_title, row=1, col=2, tick0=0, dtick=25, range=[0, dbh_y_upper])
        fig.update_yaxes(title_text=None, row=1, col=3, tick0=0, dtick=25, range=[0, h_y_upper])
    else:
        fig.update_yaxes(title_text=y_title, row=1, col=2, range=[0, dbh_y_upper])
        fig.update_yaxes(title_text=None, row=1, col=3, range=[0, h_y_upper])

    st.plotly_chart(fig, width="stretch")


render_three_panel_with_shared_legend(
    df_all=df,
    df_sub=df_sel,
    value_col=value_col,
    color_mode_key=color_mode,
    y_title=y_title,
    unit_suffix=unit_suffix,
    pie_value_label=pie_value_label,
    unit_disp=unit_disp,
)
