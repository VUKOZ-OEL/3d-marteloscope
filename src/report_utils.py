# src/report_utils.py
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple

import pandas as pd

from reportlab.lib.pagesizes import A4
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


def mpl_fig_to_png_bytes(fig: MplFigure, dpi: int = 170) -> bytes:
    """
    Matplotlib Figure -> PNG bytes (no external engine).
    """
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=dpi, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


def _safe_str(x: Any, default: str = "") -> str:
    if x is None:
        return default
    s = str(x)
    return s if s.strip() else default


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

    def pi(col: str, default: str = "") -> str:
        try:
            return _safe_str(plot_info[col].iloc[0], default=default)
        except Exception:
            return default

    # area for per-ha (optional)
    try:
        area_ha = float(plot_info["size_ha"].iloc[0])
        if not (area_ha > 0):
            area_ha = None
    except Exception:
        area_ha = None

    def per_ha(x: float) -> Optional[float]:
        if area_ha is None:
            return None
        return x / area_ha

    # basic stats
    n_trees = int(len(trees)) if trees is not None else 0

    vol_sum = None
    if trees is not None:
        if "Volume_m3" in trees.columns:
            vol_sum = pd.to_numeric(trees["Volume_m3"], errors="coerce").fillna(0).sum()
        elif "volume" in trees.columns:
            vol_sum = pd.to_numeric(trees["volume"], errors="coerce").fillna(0).sum()

    # --- PDF canvas ---
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    M = 16 * mm
    y = H - 16 * mm

    def header():
        nonlocal y
        c.setFont("Helvetica-Bold", 16)
        c.drawString(M, y, t("report_title"))
        y -= 7 * mm

        c.setFont("Helvetica", 10)
        c.drawString(M, y, f"{t('plot')}: {pi('name', '-')}")
        y -= 5 * mm

        c.drawString(M, y, f"{t('created')}: {created_dt.strftime('%Y-%m-%d %H:%M')}")
        y -= 5 * mm

        c.drawString(M, y, f"{t('intervention')}: {intervention_label}")
        y -= 7 * mm

        c.setLineWidth(0.6)
        c.line(M, y, W - M, y)
        y -= 8 * mm

    def draw_section_title(txt: str):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(M, y, txt)
        y -= 6 * mm

    def draw_line(txt: str):
        nonlocal y
        c.setFont("Helvetica", 10)
        c.drawString(M, y, txt)
        y -= 5 * mm

    def new_page():
        nonlocal y
        c.showPage()
        y = H - 16 * mm

    header()

    # --- Overview block ---
    draw_section_title(t("site_overview"))
    overview_lines = [
        f"{t('forest_type')}: {pi('forest_type','-')}",
        f"{t('area')}: {pi('size_ha','-')} ha",
        f"{t('altitude')}: {pi('altitude','-')} m",
        f"{t('precipitation')}: {pi('precipitation','-')} mm/year",
        f"{t('average_temperature')}: {pi('temperature','-')} °C",
        f"{t('owner')}: {pi('owner','-')}",
        f"{t('location')}: {pi('state','-')}",
        f"{t('scan_date')}: {pi('scan_date','-')}",
    ]
    for ln in overview_lines:
        draw_line(ln)

    y -= 2 * mm
    c.line(M, y, W - M, y)
    y -= 8 * mm

    # --- Stand summary ---
    draw_section_title(t("stand_summary"))
    ln = f"{t('number_of_trees_label')}: {n_trees}"
    p = per_ha(float(n_trees))
    if p is not None:
        ln += f"  ({p:.1f} / ha)"
    draw_line(ln)

    if vol_sum is not None:
        ln = f"{t('wood_volume_label')}: {vol_sum:,.1f} m³".replace(",", " ")
        p = per_ha(float(vol_sum))
        if p is not None:
            ln += f"  ({p:.1f} m³/ha)"
        draw_line(ln)

    y -= 2 * mm
    c.line(M, y, W - M, y)
    y -= 8 * mm

    # --- Charts ---
    draw_section_title(t("charts"))
    # area reserved for image
    max_w = W - 2 * M
    max_h = 150 * mm  # nice big chart on page

    for title, fig in figs:
        # ensure enough room for title + image; else new page
        if y < (max_h + 30 * mm):
            new_page()
            header()
            draw_section_title(t("charts"))

        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, y, _safe_str(title, t("chart")))
        y -= 6 * mm

        png = mpl_fig_to_png_bytes(fig, dpi=png_dpi)
        img = ImageReader(io.BytesIO(png))
        iw, ih = img.getSize()

        scale = min(max_w / iw, max_h / ih)
        sw, sh = iw * scale, ih * scale

        c.drawImage(img, M, y - sh, width=sw, height=sh, preserveAspectRatio=True, anchor="nw")
        y -= (sh + 10 * mm)

        # close MPL figure to avoid memory growth
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

    c.save()
    return buf.getvalue()
