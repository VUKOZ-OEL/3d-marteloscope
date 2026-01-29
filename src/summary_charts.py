# src/summary_charts.py
from __future__ import annotations

from typing import Callable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---------- Stable IDs (same as UI) ----------
STATUS_BEFORE = "label_before"
STATUS_AFTER = "label_after"
STATUS_REMOVED = "label_removed"

COLOR_BY_SPECIES = "species"
COLOR_BY_MANAGEMENT = "management_label"

METRIC_TREE_COUNT = "metric_tree_count"
METRIC_VOLUME = "metric_volume_m3"
METRIC_BASAL_AREA = "metric_basal_area_m2"
METRIC_CANOPY_COVER = "metric_canopy_cover_pct"


# ---------- Helpers ----------
def _make_masks(d: pd.DataFrame) -> Dict[str, pd.Series]:
    keep = {"Target tree", "Untouched"}
    if "management_status" in d.columns:
        after = d["management_status"].astype(str).isin(keep)
        removed = ~after
    else:
        after = pd.Series(False, index=d.index)
        removed = pd.Series(False, index=d.index)
    before = pd.Series(True, index=d.index)
    return {STATUS_BEFORE: before, STATUS_AFTER: after, STATUS_REMOVED: removed}


def _safe_hex(x: object, default: str = "#AAAAAA") -> str:
    if not isinstance(x, str):
        return default
    s = x.strip()
    if len(s) == 7 and s.startswith("#"):
        return s
    return default


def _species_colors(d: pd.DataFrame) -> Dict[str, str]:
    if "species" not in d.columns or "speciesColorHex" not in d.columns:
        return {}
    tmp = d[["species", "speciesColorHex"]].copy()
    tmp["species"] = tmp["species"].astype(str)
    tmp["speciesColorHex"] = tmp["speciesColorHex"].map(lambda v: _safe_hex(v, "#AAAAAA"))
    return tmp.groupby("species")["speciesColorHex"].first().to_dict()


def _management_colors(d: pd.DataFrame) -> Dict[str, str]:
    if "management_status" not in d.columns or "managementColorHex" not in d.columns:
        return {}
    tmp = d[["management_status", "managementColorHex"]].copy()
    tmp["management_status"] = tmp["management_status"].astype(str)
    tmp["managementColorHex"] = tmp["managementColorHex"].map(lambda v: _safe_hex(v, "#AAAAAA"))
    return tmp.groupby("management_status")["managementColorHex"].first().to_dict()


def _metric_meta(t: Callable[[str], str], metric_id: str) -> Tuple[Optional[str], str, str]:
    """
    Returns:
      value_col | None (counts),
      y_title,
      unit_disp
    """
    if metric_id == METRIC_TREE_COUNT:
        return None, t("trees"), "trees"

    if metric_id == METRIC_VOLUME:
        return "volume", f"{t('value_volume')} (m³)", "m³"

    if metric_id == METRIC_BASAL_AREA:
        return "basal_area_m2", f"{t('basal_area')} (m²)", "m²"

    if metric_id == METRIC_CANOPY_COVER:
        return "canopy_cover_pct", t("metric_canopy_cover_pct"), "%"

    return None, t("trees"), "trees"


def _make_bins(values: pd.Series, bin_size: float) -> np.ndarray:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return np.array([0.0, bin_size], dtype=float)
    vmin = float(np.floor(vals.min() / bin_size) * bin_size)
    vmax = float(np.ceil(vals.max() / bin_size) * bin_size)
    if vmax <= vmin:
        vmax = vmin + bin_size
    return np.arange(vmin, vmax + bin_size, bin_size, dtype=float)


def _agg_pie(d: pd.DataFrame, hue_col: str, value_col: Optional[str]) -> pd.Series:
    if value_col is None:
        return d.groupby(hue_col).size()
    return pd.to_numeric(d[value_col], errors="coerce").fillna(0.0).groupby(d[hue_col]).sum()


def _agg_binned_stacked(
    d: pd.DataFrame,
    base_col: str,
    bins: np.ndarray,
    hue_col: str,
    value_col: Optional[str],
    categories: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      matrix [n_cat, n_bins] (stack heights),
      labels ['0–10', ...]
    """
    if base_col not in d.columns or hue_col not in d.columns:
        return np.zeros((len(categories), len(bins) - 1), dtype=float), []

    vals = pd.to_numeric(d[base_col], errors="coerce")
    bin_idx = pd.cut(vals, bins=bins, include_lowest=True, right=False)

    dd = d.copy()
    dd["_bin"] = bin_idx
    dd = dd.dropna(subset=["_bin"])
    if dd.empty:
        return np.zeros((len(categories), len(bins) - 1), dtype=float), []

    if value_col is None:
        pv = dd.pivot_table(index="_bin", columns=hue_col, aggfunc="size", fill_value=0)
    else:
        dd["_w"] = pd.to_numeric(dd[value_col], errors="coerce").fillna(0.0)
        pv = dd.pivot_table(index="_bin", columns=hue_col, values="_w", aggfunc="sum", fill_value=0.0)

    # align bins and categories
    bin_cats = list(bin_idx.cat.categories)
    pv = pv.reindex(bin_cats).fillna(0)
    pv = pv.reindex(columns=categories).fillna(0)

    labels = [f"{int(iv.left)}–{int(iv.right)}" for iv in bin_cats]
    mat = pv.to_numpy().T  # [n_cat, n_bins]
    return mat, labels


def build_three_panel_figure(
    *,
    plot_info: pd.DataFrame,
    df: pd.DataFrame,
    dist_mode: str,
    metric_id: str,
    color_mode: str,
    t: Callable[[str], str],
) -> Figure:
    """
    Matplotlib 3-panel figure for PDF/export:
      - Pie: stand composition
      - Stacked bars: DBH classes
      - Stacked bars: Height classes
    Legend is placed BELOW plots to avoid collisions with axes.
    Designed for landscape A4.
    """
    d = df.copy()

    # Prepare columns similar to Streamlit summary
    if "Volume_m3" in d.columns and "volume" not in d.columns:
        d["volume"] = pd.to_numeric(d["Volume_m3"], errors="coerce")

    if "dbh" in d.columns and "basal_area_m2" not in d.columns:
        dbh_cm = pd.to_numeric(d["dbh"], errors="coerce")
        d["basal_area_m2"] = np.pi * (dbh_cm / 200.0) ** 2

    # canopy cover %
    try:
        area_ha = float(plot_info["size_ha"].iloc[0])
        if not np.isfinite(area_ha) or area_ha <= 0:
            area_ha = 1.0
    except Exception:
        area_ha = 1.0

    if "surfaceAreaProjection" in d.columns and "canopy_cover_pct" not in d.columns:
        sap = pd.to_numeric(d["surfaceAreaProjection"], errors="coerce").fillna(0.0)
        d["canopy_cover_pct"] = (sap / (area_ha * 10_000.0)) * 100.0

    # masks
    masks = _make_masks(d)
    mask = masks.get(dist_mode, pd.Series(True, index=d.index))
    dsel = d[mask].copy()

    # hue & colors
    if color_mode == COLOR_BY_MANAGEMENT:
        hue_col = "management_status"
        hue_title = t("hover_management") if callable(t) else "Management"
        categories = sorted(d.get(hue_col, pd.Series([], dtype=str)).astype(str).dropna().unique().tolist())
        cmap = _management_colors(d)
    else:
        hue_col = "species"
        hue_title = t("hover_species") if callable(t) else "Species"
        categories = sorted(d.get(hue_col, pd.Series([], dtype=str)).astype(str).dropna().unique().tolist())
        cmap = _species_colors(d)

    value_col, y_title, unit_disp = _metric_meta(t, metric_id)

    # If missing hue
    if hue_col not in dsel.columns:
        fig = plt.figure(figsize=(13, 4), dpi=160)
        fig.text(0.5, 0.5, f"Missing column: {hue_col}", ha="center", va="center")
        return fig

    dsel[hue_col] = dsel[hue_col].astype(str).str.strip()

    # PIE
    pie = _agg_pie(dsel, hue_col, value_col).reindex(categories).fillna(0)

    if value_col == "canopy_cover_pct":
        covered = float(pie.sum())
        uncovered = max(0.0, 100.0 - covered)
        if uncovered > 1e-6:
            pie.loc[t("uncovered")] = uncovered
            pie_labels = list(pie.index.astype(str))
        else:
            pie_labels = list(pie.index.astype(str))
    else:
        pie_labels = list(pie.index.astype(str))

    pie_values = pie.values.astype(float)
    pie_colors = []
    for lab in pie_labels:
        if lab == t("uncovered"):
            pie_colors.append("#FFFFFF")
        else:
            pie_colors.append(_safe_hex(cmap.get(lab), "#AAAAAA"))

    # Bins + stacked matrices
    dbh_bins = _make_bins(d.get("dbh", pd.Series([], dtype=float)), 10.0)
    h_bins = _make_bins(d.get("height", pd.Series([], dtype=float)), 5.0)

    dbh_mat, dbh_labels = _agg_binned_stacked(dsel, "dbh", dbh_bins, hue_col, value_col, categories)
    h_mat, h_labels = _agg_binned_stacked(dsel, "height", h_bins, hue_col, value_col, categories)

    # ---- Figure layout (landscape-friendly) ----
    fig = plt.figure(figsize=(13.4, 4.4), dpi=160)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.85, 1.85], wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Pie plot (donut-ish)
    total = float(np.nansum(pie_values))
    if total <= 0:
        ax0.text(0.5, 0.5, t("no_data"), ha="center", va="center")
        ax0.set_axis_off()
    else:
        ax0.pie(
            pie_values,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.50, edgecolor="white"),
        )
        if value_col is None:
            center_txt = f"Σ\n{int(round(total))}\n{unit_disp}"
        else:
            center_txt = f"Σ\n{total:,.1f}\n{unit_disp}".replace(",", " ")
        ax0.text(0.0, 0.0, center_txt, ha="center", va="center", fontsize=12, fontweight="bold")
        ax0.set_title(t("stand_composition"), fontsize=11)

    # DBH stacked
    ax1.set_title(t("in_dbh_class"), fontsize=11)
    x = np.arange(len(dbh_labels))
    bottom = np.zeros(len(dbh_labels), dtype=float)
    for i, cat in enumerate(categories):
        if dbh_mat.size == 0:
            continue
        y = dbh_mat[i, :]
        if np.allclose(y, 0):
            continue
        ax1.bar(
            x, y, bottom=bottom,
            label=cat,
            color=_safe_hex(cmap.get(cat), "#AAAAAA"),
            linewidth=0,
        )
        bottom += y

    ax1.set_ylabel(y_title, fontsize=10)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.tick_params(axis="both", labelsize=8)

    if dbh_labels:
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{lab} {t('unit_cm')}" for lab in dbh_labels], rotation=35, ha="right", fontsize=8)

    # Height stacked
    ax2.set_title(t("in_height_class"), fontsize=11)
    x2 = np.arange(len(h_labels))
    bottom = np.zeros(len(h_labels), dtype=float)
    for i, cat in enumerate(categories):
        if h_mat.size == 0:
            continue
        y = h_mat[i, :]
        if np.allclose(y, 0):
            continue
        ax2.bar(
            x2, y, bottom=bottom,
            color=_safe_hex(cmap.get(cat), "#AAAAAA"),
            linewidth=0,
        )
        bottom += y

    ax2.grid(True, axis="y", alpha=0.25)
    ax2.tick_params(axis="both", labelsize=8)

    if h_labels:
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"{lab} {t('unit_m')}" for lab in h_labels], rotation=35, ha="right", fontsize=8)

    # Legend below (shared)
    if categories:
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(6, max(1, len(labels))),
                frameon=False,
                bbox_to_anchor=(0.5, 0.02),  # outside axes, below
                fontsize=8,
                title=hue_title,
                title_fontsize=9,
            )

    # Title
    fig.suptitle(f"{t(dist_mode)} — {t(metric_id)} — {t(color_mode)}", fontsize=12, fontweight="bold", y=0.98)

    # reserve room for legend and title
    fig.subplots_adjust(bottom=0.24, top=0.88)

    return fig
