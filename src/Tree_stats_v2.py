import streamlit as st
import pandas as pd
import numpy as np
import math
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# ---------- DATA ----------
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- HLAVIČKA ----------
st.markdown("### Plot summary:     **Per Hectare Values**")
st.markdown(
    f"**Area:** {plot_info['size_ha'].iloc[0]:.4f} ha"
    if "size_ha" in plot_info.columns
    else ""
)

# ---------- SPOLEČNÉ ----------
CHART_HEIGHT = 360
Before = st.session_state.Before
After = st.session_state.After
Removed = st.session_state.Removed

colorBySpp = st.session_state.Species
colorByMgmt = st.session_state.Management

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

area_m2 = area_ha * 10_000.0


def _make_masks(d: pd.DataFrame):
    keep_status = {"Target tree", "Untouched"}
    mask_after = d.get("management_status", pd.Series(False, index=d.index)).isin(
        keep_status
    )
    mask_removed = (
        ~mask_after
        if "management_status" in d.columns
        else pd.Series(False, index=d.index)
    )
    mask_before = pd.Series(True, index=d.index)  # vše
    return {Before: mask_before, After: mask_after, Removed: mask_removed}


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
    """Barvy ze sloupce managementColorHex pro všechny management_status,
    chybějící -> šedá."""
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


def _make_bins_labels(
    df_all: pd.DataFrame, value_col: str, bin_size: float, unit_label: str
):
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


def _y_upper_from_long(long_df: pd.DataFrame, labels: list[str]) -> float:
    if long_df.empty:
        return 10.0
    totals = long_df.groupby("bin")["value"].sum().reindex(labels).fillna(0.0)
    maxv = float(totals.max())
    if maxv <= 0:
        return 10.0
    magnitude = 10 ** int(np.floor(np.log10(maxv)))
    step = magnitude / 2
    upper = math.ceil(maxv / step) * step
    return upper


# ---------- OVLÁDÁNÍ ----------
c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 3, 0.25, 4, 0.25, 2, 0.5])

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
        options=["Tree count", "Volume (m³)", "Basal area (m²)", "Stocking"],
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
    """
    Vrátí (metric_key, y_title, unit_suffix, pie_value_label).

    metric_key:
        - "count"      -> počet stromů/ha
        - "volume"     -> objem m³/ha
        - "basal"      -> bazální plocha m²/ha
        - "stocking"   -> stocking v % plochy
    """
    if label == "Tree count":
        return "count", "Trees", "trees/ha", "Trees"
    if label == "Volume (m³)":
        return "volume", "Volume (m³)", "m³/ha", "Volume (m³)"
    if label == "Basal area (m²)":
        return "basal", "Basal area (m²)", "m²/ha", "Basal area (m²)"
    if label == "Stocking":
        return "stocking", "Stocking (%)", "%", "Stocking"
    # fallback
    return "count", "Trees", "trees/ha", "Trees"


metric_key, y_title, unit_suffix, pie_value_label = _metric_meta(sum_metric_label)

masks = _make_masks(df)
mask = masks.get(dist_mode, pd.Series(True, index=df.index))
df_sel = df[mask].copy()


def render_three_panel_with_shared_legend(
    df_all: pd.DataFrame,
    df_sub: pd.DataFrame,
    metric_key: str,
    color_mode: str,
    y_title: str,
    unit_suffix: str,
    pie_value_label: str,
    area_ha: float,
    area_m2: float,
):
    # --- Nastavení hue / kategorií / barev
    if color_mode == colorByMgmt:
        hue_col = "management_status"
        if hue_col in df_all.columns:
            categories = pd.Index(
                df_all[hue_col].astype(str).dropna().unique()
            ).tolist()
        else:
            categories = []
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
        st.warning(f"Chybí sloupec '{hue_col}' – nelze vykreslit grafy.")
        return
    d[hue_col] = d[hue_col].astype(str).str.strip()

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

    # --- Funkce pro výpočet "váhy" stromu podle metriky
    def per_tree_weight_for_metric(df_in: pd.DataFrame, metric_key: str) -> pd.Series:
        if metric_key == "count":
            return pd.Series(1.0 / area_ha, index=df_in.index)
        elif metric_key == "volume":
            return (
                pd.to_numeric(df_in.get("volume"), errors="coerce").fillna(0.0)
                / area_ha
            )
        elif metric_key == "basal":
            return (
                pd.to_numeric(df_in.get("basal_area_m2"), errors="coerce").fillna(0.0)
                / area_ha
            )
        elif metric_key == "stocking":
            if "horizontal_crown_projection" not in df_in.columns:
                return pd.Series(0.0, index=df_in.index)
            crown = pd.to_numeric(
                df_in["horizontal_crown_projection"], errors="coerce"
            ).fillna(0.0)
            # Přepočet na % z plochy porostu (0–100)
            return (crown / area_m2) * 100.0
        else:
            return pd.Series(0.0, index=df_in.index)

    # --- PIE agregace + celkový součet pro střed
    weights = per_tree_weight_for_metric(d, metric_key)

    if metric_key == "stocking":
        if "horizontal_crown_projection" not in df_all.columns:
            st.warning(
                "Chybí sloupec 'horizontal_crown_projection' – nelze spočítat Stocking."
            )
            return

        # stocking % podle kategorií
        pie_agg = (
            d.assign(weight=weights)
            .groupby(hue_col, as_index=False)
            .agg(value=("weight", "sum"))
        )

        total_stocking_percent = float(pie_agg["value"].sum())
        total_stocking_percent = max(0.0, min(100.0, total_stocking_percent))

        # přidej "Empty" výseč = zbytek plochy
        empty_part = max(0.0, 100.0 - total_stocking_percent)
        pie_agg = pie_agg[pie_agg["value"] > 0].copy()

        # legendární kategorie pro stocking zachovej (species / mgmt), "Empty" bez legendy
        show_empty = empty_part > 0.0

        if show_empty:
            pie_agg = pd.concat(
                [
                    pie_agg,
                    pd.DataFrame({hue_col: ["Empty"], "value": [empty_part]}),
                ],
                ignore_index=True,
            )

        total_raw = total_stocking_percent
        unit_disp = "%"
        total_text = (
            f"Stocking<br><b>{total_stocking_percent:.0f} %</b>"  # střed koláče
        )
        hover_value_token = "%{value:.1f}"

        # barvy – Empty světle šedá
        color_map_with_empty = {**color_map, "Empty": "#EEEEEE"}
        categories_for_pie = pie_agg[hue_col].tolist()
        pie_colors = [
            color_map_with_empty.get(c, "#AAAAAA") for c in categories_for_pie
        ]
        pie_labels = pie_agg[hue_col]
        pie_values = pie_agg["value"]

        hovertemplate = (
            f"{hue_col}: %{{label}}<br>"
            f"{pie_value_label}: {hover_value_token}"
            "<extra></extra>"
        )

        # pro další grafy nechceme "Empty" jako kategorii
        categories_for_bars = [c for c in categories if c in d[hue_col].unique()]

    else:
        # běžné metriky: count / volume / basal
        if metric_key == "count":
            unit_disp = "trees/ha"
        elif metric_key == "volume":
            unit_disp = "m³/ha"
        elif metric_key == "basal":
            unit_disp = "m²/ha"
        else:
            unit_disp = ""

        t = d.assign(weight=weights)
        pie_agg = t.groupby(hue_col, as_index=False).agg(value=("weight", "sum"))
        total_raw = float(pie_agg["value"].sum())
        pie_agg = pie_agg[pie_agg["value"] > 0].copy()

        if metric_key == "count":
            total_text = f"Σ =<br><b>{total_raw:,.0f}</b><br>{unit_disp}".replace(
                ",", " "
            )
            hover_value_token = "%{value:.0f}"
        else:
            total_text = f"Σ =<br><b>{total_raw:,.1f}</b><br>{unit_disp}".replace(
                ",", " "
            )
            hover_value_token = "%{value:.1f}"

        # respektuj pořadí kategorií
        if categories:
            pie_agg = (
                pie_agg.set_index(hue_col).reindex(categories).fillna(0.0).reset_index()
            )
        else:
            categories = pie_agg[hue_col].tolist()

        pie_labels = pie_agg[hue_col]
        pie_values = pie_agg["value"]
        pie_colors = [color_map.get(c, "#AAAAAA") for c in pie_labels]

        hovertemplate = (
            f"{hue_col}: %{{label}}<br>"
            f"{pie_value_label}: {hover_value_token} {unit_disp}"
            "<extra></extra>"
        )

        categories_for_bars = categories

    no_data = pie_agg.empty

    # --- Long tabulky pro DBH / Height (stacked)
    def long_binned(
        df_in: pd.DataFrame,
        base_col: str,
        bins: np.ndarray,
        labels: list[str],
        hue: str,
        metric_key: str,
    ) -> pd.DataFrame:
        t = df_in.copy()
        t[hue] = t[hue].astype(str)
        vals = pd.to_numeric(t[base_col], errors="coerce")
        cats = pd.Categorical(
            pd.cut(
                vals,
                bins=bins,
                labels=labels,
                include_lowest=True,
                right=False,
                ordered=True,
            ),
            categories=labels,
            ordered=True,
        )
        t = t.assign(bin=cats).dropna(subset=["bin"])
        if t.empty:
            return pd.DataFrame(columns=["bin", hue, "value"])
        t["weight"] = per_tree_weight_for_metric(t, metric_key)
        pv = t.pivot_table(
            index="bin", columns=hue, values="weight", aggfunc="sum", fill_value=0.0
        )
        long = pv.stack().rename("value").reset_index()
        long["bin"] = long["bin"].astype(str)
        return long

    dbh_long = long_binned(df_sub, "dbh", dbh_bins, dbh_labels, hue_col, metric_key)
    if "height" in df_sub.columns:
        height_long = long_binned(
            df_sub, "height", h_bins, h_labels, hue_col, metric_key
        )
    else:
        height_long = pd.DataFrame(columns=["bin", hue_col, "value"])

    # --- Y-osa: horní hranice
    if metric_key == "stocking":
        dbh_y_upper = 100.0
        h_y_upper = 100.0
    else:
        dbh_y_upper = _y_upper_from_long(dbh_long, dbh_labels)
        h_y_upper = _y_upper_from_long(height_long, h_labels)

    # --- Subplots: pie | dbh | height
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Stand Composition", "In DBH class", "In Height class"),
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

    # --- 1) PIE (hole=0.50)
    if no_data:
        pie_labels_plot = ["No data"]
        pie_values_plot = [1]
        pie_colors_plot = ["#EEEEEE"]
        hovertemplate_plot = ""
        textinfo = "none"
        texttemplate = None
    else:
        pie_labels_plot = pie_labels
        pie_values_plot = pie_values
        pie_colors_plot = pie_colors
        textinfo = "text"
        texttemplate = "%{percent:.1%}"
        hovertemplate_plot = hovertemplate

    fig.add_trace(
        go.Pie(
            labels=pie_labels_plot,
            values=pie_values_plot,
            hole=0.50,
            marker=dict(colors=pie_colors_plot),
            textinfo=textinfo,
            texttemplate=texttemplate,
            textposition="inside",
            insidetextorientation="radial",
            textfont=dict(size=11),
            hovertemplate=hovertemplate_plot,
            showlegend=False,
            sort=False,
        ),
        row=1,
        col=1,
    )

    # --- středový text (Σ nebo stocking)
    pie_trace = fig.data[-1]
    if hasattr(pie_trace, "domain") and pie_trace.domain:
        cx = (pie_trace.domain.x[0] + pie_trace.domain.x[1]) / 2
        cy = (pie_trace.domain.y[0] + pie_trace.domain.y[1]) / 2
    else:
        cx, cy = 0.17, 0.5  # fallback

    fig.add_annotation(
        x=cx,
        y=cy,
        xref="paper",
        yref="paper",
        text=(total_text if not no_data else "Σ = 0 " + unit_disp),
        showarrow=False,
        font=dict(size=24, color="black"),
        xanchor="center",
        yanchor="middle",
    )

    # --- 2) DBH (stack podle hue)
    for cat in categories_for_bars:
        y_vals = (
            dbh_long[dbh_long[hue_col] == cat]
            .set_index("bin")
            .reindex(dbh_labels)["value"]
            .fillna(0.0)
            .tolist()
        )
        if metric_key == "stocking":
            hover_tmpl = (
                f"%{{x}}<br>{hue_col}: {cat}<br>Stocking: %{{y:.1f}} %<extra></extra>"
            )
        else:
            if metric_key == "count":
                hover_tmpl = f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y:.0f}}<extra></extra>"
            else:
                hover_tmpl = f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y:.1f}}<extra></extra>"

        fig.add_trace(
            go.Bar(
                x=dbh_labels,
                y=y_vals,
                name=cat,
                marker_color=color_map.get(cat, "#AAAAAA"),
                legendgroup=cat,
                showlegend=True,
                hovertemplate=hover_tmpl,
            ),
            row=1,
            col=2,
        )

    # --- 3) HEIGHT (stack)
    if not height_long.empty:
        for cat in categories_for_bars:
            y_vals = (
                height_long[height_long[hue_col] == cat]
                .set_index("bin")
                .reindex(h_labels)["value"]
                .fillna(0.0)
                .tolist()
            )
            if metric_key == "stocking":
                hover_tmpl = f"%{{x}}<br>{hue_col}: {cat}<br>Stocking: %{{y:.1f}} %<extra></extra>"
            else:
                if metric_key == "count":
                    hover_tmpl = f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y:.0f}}<extra></extra>"
                else:
                    hover_tmpl = f"%{{x}}<br>{hue_col}: {cat}<br>{y_title}: %{{y:.1f}}<extra></extra>"

            fig.add_trace(
                go.Bar(
                    x=h_labels,
                    y=y_vals,
                    name=cat,
                    marker_color=color_map.get(cat, "#AAAAAA"),
                    legendgroup=cat,
                    showlegend=False,
                    hovertemplate=hover_tmpl,
                ),
                row=1,
                col=3,
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
    fig.update_xaxes(
        title_text=None,
        tickangle=45,
        categoryorder="array",
        categoryarray=dbh_labels,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text=None,
        tickangle=45,
        categoryorder="array",
        categoryarray=h_labels,
        row=1,
        col=3,
    )

    if metric_key == "stocking":
        fig.update_yaxes(
            title_text=y_title,
            row=1,
            col=2,
            range=[0, 100],
            tick0=0,
            dtick=20,
        )
        fig.update_yaxes(
            title_text=None,
            row=1,
            col=3,
            range=[0, 100],
            tick0=0,
            dtick=20,
        )
    else:
        if metric_key == "count":
            fig.update_yaxes(
                title_text=y_title,
                row=1,
                col=2,
                tick0=0,
                dtick=max(5, dbh_y_upper / 8),
                range=[0, dbh_y_upper],
            )
            fig.update_yaxes(
                title_text=None,
                row=1,
                col=3,
                tick0=0,
                dtick=max(5, h_y_upper / 8),
                range=[0, h_y_upper],
            )
        else:
            fig.update_yaxes(title_text=y_title, row=1, col=2, range=[0, dbh_y_upper])
            fig.update_yaxes(title_text=None, row=1, col=3, range=[0, h_y_upper])

    st.plotly_chart(fig, use_container_width=True)


# ---------- RENDER PLOTS ----------
render_three_panel_with_shared_legend(
    df_all=df,
    df_sub=df_sel,
    metric_key=metric_key,
    color_mode=color_mode,
    y_title=y_title,
    unit_suffix=unit_suffix,
    pie_value_label=pie_value_label,
    area_ha=area_ha,
    area_m2=area_m2,
)
