# src/report_utils.py
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from matplotlib.figure import Figure as MplFigure

from src.summary_charts import (
    build_three_panel_figure,
    STATUS_BEFORE, STATUS_AFTER, STATUS_REMOVED,
    COLOR_BY_SPECIES, COLOR_BY_MANAGEMENT,
    METRIC_TREE_COUNT, METRIC_VOLUME, METRIC_BASAL_AREA, METRIC_CANOPY_COVER,
)


# ---------------------------
# Matplotlib -> PNG bytes
# ---------------------------
def mpl_fig_to_png_bytes(fig: MplFigure, dpi: int = 170) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x)
    return s if s.strip() else default


# ---------------------------
# Variants
# ---------------------------
@dataclass(frozen=True)
class SummaryVariant:
    dist_mode: str
    metric_id: str
    color_mode: str


def default_summary_variants() -> List[SummaryVariant]:
    return [
        SummaryVariant(d, m, c)
        for d in (STATUS_BEFORE, STATUS_AFTER, STATUS_REMOVED)
        for m in (METRIC_TREE_COUNT, METRIC_VOLUME, METRIC_BASAL_AREA, METRIC_CANOPY_COVER)
        for c in (COLOR_BY_SPECIES, COLOR_BY_MANAGEMENT)
    ]


def make_variant_title(t: Callable[[str], str], v: SummaryVariant) -> str:
    return f"{t(v.dist_mode)} — {t(v.metric_id)} — {t(v.color_mode)}"


def generate_all_summary_figs(
    *,
    plot_info: pd.DataFrame,
    trees: pd.DataFrame,
    t: Callable[[str], str],
    variants: Optional[List[SummaryVariant]] = None,
    # compatibility: ignored (was used in Plotly version)
    plot_title_font: Any = None,
) -> List[Tuple[str, MplFigure]]:
    variants = variants or default_summary_variants()

    out: List[Tuple[str, MplFigure]] = []
    for v in variants:
        title = make_variant_title(t, v)
        fig = build_three_panel_figure(
            plot_info=plot_info,
            df=trees,
            dist_mode=v.dist_mode,
            metric_id=v.metric_id,
            color_mode=v.color_mode,
            t=t,
        )
        out.append((title, fig))
    return out


# ---------------------------
# First-page summary table
# ---------------------------
def _ensure_metrics(df: pd.DataFrame, area_ha: float) -> pd.DataFrame:
    d = df.copy()
    if "Volume_m3" in d.columns and "volume" not in d.columns:
        d["volume"] = pd.to_numeric(d["Volume_m3"], errors="coerce")
    if "dbh" in d.columns and "basal_area_m2" not in d.columns:
        dbh_cm = pd.to_numeric(d["dbh"], errors="coerce")
        d["basal_area_m2"] = np.pi * (dbh_cm / 200.0) ** 2
    if "surfaceAreaProjection" in d.columns and "canopy_cover_pct" not in d.columns:
        sap = pd.to_numeric(d["surfaceAreaProjection"], errors="coerce").fillna(0.0)
        d["canopy_cover_pct"] = (sap / (area_ha * 10_000.0)) * 100.0
    return d


def _make_masks_for_report(d: pd.DataFrame) -> dict:
    keep = {"Target tree", "Untouched"}
    if "management_status" in d.columns:
        after = d["management_status"].astype(str).isin(keep)
        harvested = ~after
    else:
        after = pd.Series(False, index=d.index)
        harvested = pd.Series(False, index=d.index)
    before = pd.Series(True, index=d.index)
    return {"before": before, "after": after, "harvested": harvested}


def _sum_metric(d: pd.DataFrame, metric: str) -> float:
    if metric == "tree_count":
        return float(len(d))
    if metric == "volume":
        if "volume" not in d.columns:
            return float("nan")
        return float(pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).sum())
    if metric == "basal_area":
        if "basal_area_m2" not in d.columns:
            return float("nan")
        return float(pd.to_numeric(d["basal_area_m2"], errors="coerce").fillna(0.0).sum())
    return float("nan")


def _stand_canopy_cover_pct(d: pd.DataFrame, area_ha: float) -> float:
    # canopy cover computed from sum(surfaceAreaProjection)/plot_area
    if area_ha <= 0 or "surfaceAreaProjection" not in d.columns:
        return float("nan")
    sap = pd.to_numeric(d["surfaceAreaProjection"], errors="coerce").fillna(0.0).sum()
    return (sap / (area_ha * 10_000.0)) * 100.0


def compute_report_table(trees: pd.DataFrame, plot_info: pd.DataFrame) -> dict:
    try:
        area_ha = float(plot_info["size_ha"].iloc[0])
        if not np.isfinite(area_ha) or area_ha <= 0:
            area_ha = 1.0
    except Exception:
        area_ha = 1.0

    d = _ensure_metrics(trees, area_ha)
    masks = _make_masks_for_report(d)

    out = {"area_ha": area_ha}
    for mode in ("before", "after", "harvested"):
        dd = d[masks[mode]].copy()
        out[(mode, "tree_count")] = _sum_metric(dd, "tree_count")
        out[(mode, "volume")] = _sum_metric(dd, "volume")
        out[(mode, "basal_area")] = _sum_metric(dd, "basal_area")
        out[(mode, "canopy_cover")] = _stand_canopy_cover_pct(dd, area_ha)

    return out


# ---------------------------
# PDF rendering
# ---------------------------
def build_intervention_report_pdf(
    *,
    plot_info: pd.DataFrame,
    trees: pd.DataFrame,
    figs: Sequence[Tuple[str, MplFigure]],
    intervention_label: str,
    t: Callable[[str], str],
    language: str = "cs",
    created_dt: Optional[datetime] = None,
    png_dpi: int = 170,
) -> bytes:
    created_dt = created_dt or datetime.now()

    # Plot info accessor
    def pi(col: str, default: str = "") -> str:
        try:
            return _safe_str(plot_info[col].iloc[0], default=default)
        except Exception:
            return default

    # avoid "Area::" etc. (strip any trailing colon from translation keys)
    def _lbl(key: str) -> str:
        s = _safe_str(t(key), key)
        return s.rstrip().rstrip(":")

    def _kv(key: str, value: str) -> str:
        return f"{_lbl(key)}: {value}"

    # area (for per-ha column)
    try:
        area_ha = float(plot_info["size_ha"].iloc[0])
        if not np.isfinite(area_ha) or area_ha <= 0:
            area_ha = 1.0
    except Exception:
        area_ha = 1.0

    def per_ha(x: float) -> Optional[float]:
        if area_ha <= 0:
            return None
        return x / area_ha

    def fmt_num(x: float, nd: int = 1) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "-"
        if nd == 0:
            return f"{int(round(float(x)))}"
        return f"{float(x):,.{nd}f}".replace(",", " ")

    # --- Prepare title-page table stats ---
    stats = compute_report_table(trees, plot_info)

    # --- PDF canvas (LANDSCAPE) ---
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(A4))
    W, H = landscape(A4)

    M = 16 * mm
    y = H - 16 * mm

    def new_page():
        nonlocal y
        c.showPage()
        y = H - 16 * mm

    def draw_header():
        nonlocal y
        c.setFont("Helvetica-Bold", 18)
        c.drawString(M, y, _safe_str(t("report_title"), "Intervention report"))
        y -= 7 * mm

        c.setFont("Helvetica", 10)
        c.drawString(M, y, f"{_lbl('plot')}: {pi('name', '-')}")
        y -= 5 * mm

        c.drawString(M, y, f"{_lbl('created')}: {created_dt.strftime('%Y-%m-%d %H:%M')}")
        y -= 5 * mm

        # intervention label already prepared by dashboard (user vs saved name)
        c.drawString(M, y, f"{_lbl('intervention')}: {intervention_label}")
        y -= 8 * mm

        c.setLineWidth(0.7)
        c.line(M, y, W - M, y)
        y -= 8 * mm

    def section_title(txt: str):
        nonlocal y
        c.setFont("Helvetica-Bold", 13)
        c.drawString(M, y, txt)
        y -= 6 * mm

    def line(txt: str):
        nonlocal y
        c.setFont("Helvetica", 10)
        c.drawString(M, y, txt)
        y -= 5 * mm

    def ensure_space(min_y: float):
        nonlocal y
        if y < min_y:
            new_page()
            draw_header()

    def draw_table(x: float, top_y: float, col_w: List[float], row_h: float, rows: List[List[str]]):
        """
        Simple ReportLab table renderer (canvas-based).
        """
        n_rows = len(rows)
        table_h = n_rows * row_h
        table_w = float(sum(col_w))

        # header background
        c.setFillGray(0.93)
        c.rect(x, top_y - row_h, table_w, row_h, stroke=0, fill=1)
        c.setFillGray(0)

        # outer border
        c.setLineWidth(0.6)
        c.rect(x, top_y - table_h, table_w, table_h, stroke=1, fill=0)

        # vertical lines
        xx = x
        for w in col_w[:-1]:
            xx += w
            c.line(xx, top_y, xx, top_y - table_h)

        # horizontal lines
        yy = top_y
        for i in range(1, n_rows):
            yy -= row_h
            c.line(x, yy, x + table_w, yy)

        # text
        pad = 2.4 * mm
        yy_text = top_y - row_h + 0.28 * row_h
        for r_i, row in enumerate(rows):
            xx_text = x
            if r_i == 0:
                c.setFont("Helvetica-Bold", 9)
            else:
                c.setFont("Helvetica", 9)
            for j, cell in enumerate(row):
                c.drawString(xx_text + pad, yy_text, str(cell))
                xx_text += col_w[j]
            yy_text -= row_h

    # ---------- PAGE 1 (Title + overview + intervention summary table) ----------
    draw_header()

    section_title(_safe_str(t("site_overview"), "Site overview"))

    overview_lines = [
        _kv("forest_type", pi("forest_type", "-")),
        _kv("area", f"{pi('size_ha','-')} ha"),
        _kv("altitude", f"{pi('altitude','-')} m"),
        _kv("precipitation", f"{pi('precipitation','-')} mm/year"),
        _kv("average_temperature", f"{pi('temperature','-')} °C"),
        _kv("owner", pi("owner", "-")),
        _kv("location", pi("state", "-")),
        _kv("scan_date", pi("scan_date", "-")),
    ]
    for ln in overview_lines:
        line(ln)

    y -= 2 * mm
    c.line(M, y, W - M, y)
    y -= 8 * mm

    section_title(_safe_str(t("report_intervention_summary"), "Intervention summary"))

    # Table rows
    before_cnt = float(stats[("before", "tree_count")])
    rows = [
        [_safe_str(t("metric"), "Metric"),
         _safe_str(t("before"), "Before"),
         _safe_str(t("after"), "After"),
         _safe_str(t("harvested"), "Harvested"),
         _safe_str(t("per_ha"), "Per ha (before)")],
        [_safe_str(t(METRIC_TREE_COUNT), "Tree count"),
         fmt_num(stats[("before", "tree_count")], 0),
         fmt_num(stats[("after", "tree_count")], 0),
         fmt_num(stats[("harvested", "tree_count")], 0),
         fmt_num(per_ha(stats[("before", "tree_count")]) or float("nan"), 1)],
        [_safe_str(t(METRIC_VOLUME), "Volume (m³)"),
         fmt_num(stats[("before", "volume")], 1),
         fmt_num(stats[("after", "volume")], 1),
         fmt_num(stats[("harvested", "volume")], 1),
         fmt_num(per_ha(stats[("before", "volume")]) or float("nan"), 1)],
        [_safe_str(t(METRIC_BASAL_AREA), "Basal area (m²)"),
         fmt_num(stats[("before", "basal_area")], 1),
         fmt_num(stats[("after", "basal_area")], 1),
         fmt_num(stats[("harvested", "basal_area")], 1),
         fmt_num(per_ha(stats[("before", "basal_area")]) or float("nan"), 1)],
        [_safe_str(t(METRIC_CANOPY_COVER), "Canopy cover (%)"),
         f"{fmt_num(stats[('before','canopy_cover')], 1)} %",
         f"{fmt_num(stats[('after','canopy_cover')], 1)} %",
         f"{fmt_num(stats[('harvested','canopy_cover')], 1)} %",
         "-"],
    ]

    # Wide landscape widths
    col_w = [70 * mm, 38 * mm, 38 * mm, 38 * mm, 45 * mm]
    table_top = y
    draw_table(M, table_top, col_w, row_h=8 * mm, rows=rows)
    y -= (len(rows) * 8 * mm + 10 * mm)

    # Footer note on page 1
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(M, 10 * mm, _safe_str(t("report_footer_note"), ""))

    # ---------- CHART PAGES ----------
    new_page()
    draw_header()
    section_title(_safe_str(t("charts"), "Charts"))

    # Chart image box (landscape)
    max_w = W - 2 * M
    max_h = 155 * mm

    for title, fig in figs:
        # need space for title + image
        ensure_space(min_y=(max_h + 28 * mm))

        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, y, _safe_str(title, _safe_str(t("chart"), "Chart")))
        y -= 6 * mm

        png = mpl_fig_to_png_bytes(fig, dpi=png_dpi)
        img = ImageReader(io.BytesIO(png))
        iw, ih = img.getSize()

        scale = min(max_w / iw, max_h / ih)
        sw, sh = iw * scale, ih * scale

        # draw
        c.drawImage(img, M, y - sh, width=sw, height=sh, preserveAspectRatio=True, anchor="nw")
        y -= (sh + 10 * mm)

        # close mpl fig to avoid memory leaks
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    c.save()
    return buf.getvalue()
