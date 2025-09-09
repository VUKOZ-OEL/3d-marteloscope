# iland_plots.py
# -*- coding: utf-8 -*-
"""
Python převod z R skriptu pro iLand výstupy:
- Načtení více SQLite souborů s tabulkami 'tree' a 'treeremoved'
- Agregace úhynů podle druhu (absolute / relative / relative_toForest)
- Náhled grafu v samostatném okně (plt.show)
- Uložení do JPG 300 dpi do složky ./output

Požadavky: pandas, matplotlib, pillow (pro JPG), numpy
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

Scale = Literal["absolute", "relative", "relative_toForest"]

# ---- definice barev pro druhy (podle R skriptu) ----
SPECIES_CODES: List[str] = [
    "abal","abam","abgr","abpr","acca","acma","acpl","acps","acru",
    "algl","alin","alru","alvi","bepe","cabe","casa","coav","fasy",
    "frex","lade","piab","pice","pini","pipo","pisi","pisy","poni",
    "potr","psme","qupe","qupu","quro","reader","rops","saca","soar",
    "soau","thpl","tico","tipl","tshe","tsme","ulgl"
]

def build_species_palette(codes: List[str]) -> Dict[str, Tuple[float,float,float, float]]:
    """
    Vytvoří mapu species -> barva (RGBA) pomocí colormap 'turbo' (nebo 'viridis' jako fallback).
    """
    try:
        cmap = plt.get_cmap("turbo")
    except ValueError:
        cmap = plt.get_cmap("viridis")
    # rozprostřít barvy přes jednotkový interval
    if len(codes) == 1:
        vals = np.array([0.5])
    else:
        vals = np.linspace(0.02, 0.98, len(codes))
    colors = {code: cmap(v) for code, v in zip(codes, vals)}
    return colors

PALETTE: Dict[str, Tuple[float,float,float,float]] = build_species_palette(SPECIES_CODES)

# ---- utilitky ----
def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

@dataclass
class ILandOutputs:
    living: Dict[str, pd.DataFrame]
    death: Dict[str, pd.DataFrame]

def read_outputs_tree_level(root_folder: str | os.PathLike) -> ILandOutputs:
    """
    Načte všechny soubory ve složce jako SQLite a přečte tabulky 'tree' a 'treeremoved'.
    Každé DB přiřadí replikaci ve tvaru 'rep_001' apod.
    """
    root = Path(root_folder)
    files = sorted([p for p in root.iterdir() if p.is_file()])
    living: Dict[str, pd.DataFrame] = {}
    death: Dict[str, pd.DataFrame] = {}

    pad = len(str(len(files)))
    for i, f in enumerate(files, start=1):
        rep = f"rep_{str(i).zfill(pad)}"
        with sqlite3.connect(str(f)) as con:
            try:
                t = pd.read_sql_query("SELECT * FROM tree", con)
            except Exception:
                t = pd.DataFrame()
            try:
                d = pd.read_sql_query("SELECT * FROM treeremoved", con)
            except Exception:
                d = pd.DataFrame()

        if not t.empty:
            t = t.copy()
            t["replication"] = rep
            living[rep] = t
        if not d.empty:
            d = d.copy()
            d["replication"] = rep
            death[rep] = d

    return ILandOutputs(living=living, death=death)

def combine_with_ids(mapping: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Z dict replikací (id -> DataFrame) udělá jeden DataFrame a přidá sloupec replication (již je)."""
    if not mapping:
        return pd.DataFrame(columns=["replication"])
    return pd.concat(mapping.values(), ignore_index=True)

# ==== Deaths by Species (stacked counts) ====
def deaths_by_species(df_death: pd.DataFrame,
                           df_living: pd.DataFrame | None,
                           scale: Scale) -> pd.DataFrame:
    """
    Vrátí DataFrame se sloupci: species, value (no_death ve zvolené škále).
    - absolute: průměr úhynů na replikaci (počet / no. replikací)
    - relative: podíl úhynů daného druhu na všech úhynech (0-1)
    - relative_toForest: úhyny daného druhu relativně k počtu živých stromů tohoto druhu
                         (agregované přes replikace; výsledek v 0-1, resp. může být <=1)
    """
    # ochrana na prázdné vstupy
    if df_death is None or df_death.empty:
        return pd.DataFrame(columns=["species", "value"])

    # počty úhynů dle druhu
    deaths = (df_death.groupby("species")["id"].size()
              .rename("no_death").reset_index())

    if scale == "absolute":
        nreps = df_death["replication"].nunique() if "replication" in df_death else 1
        deaths["value"] = deaths["no_death"] / max(nreps, 1)
        out = deaths[["species", "value"]].copy()
        return out.sort_values("species").reset_index(drop=True)

    elif scale == "relative":
        total = deaths["no_death"].sum()
        denom = total if total > 0 else 1.0
        deaths["value"] = deaths["no_death"] / denom
        out = deaths[["species", "value"]].copy()
        return out.sort_values("species").reset_index(drop=True)

    elif scale == "relative_toForest":
        if df_living is None or df_living.empty:
            # Bez živých stromů nelze normalizovat – vrať relativní k celku
            total = deaths["no_death"].sum()
            denom = total if total > 0 else 1.0
            deaths["value"] = deaths["no_death"] / denom
            return deaths[["species", "value"]].sort_values("species").reset_index(drop=True)

        living_counts = (df_living.groupby("species")["id"].size()
                         .rename("no_living").reset_index())
        tmp = pd.merge(deaths, living_counts, on="species", how="inner")
        # normalizace: úhyny / počet živých (po druzích)
        tmp["value"] = tmp["no_death"] / tmp["no_living"].replace({0: np.nan})
        tmp["value"] = tmp["value"].fillna(0.0)
        out = tmp[["species", "value"]].copy()
        return out.sort_values("species").reset_index(drop=True)

    else:
        raise ValueError("scale must be one of: 'absolute', 'relative', 'relative_toForest'")

def plot_bar_by_species(df_vals: pd.DataFrame,
                        palette: Dict[str, Tuple[float,float,float,float]] = PALETTE,
                        scale: Scale = "absolute",
                        title: str | None = None):
    """
    Vykreslí sloupcový graf hodnot (value) podle species s barevnou škálou pro každý druh.
    """
    if df_vals.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    # seřadit x podle species (fixní pořadí z R? -> použijeme abecední)
    df_vals = df_vals.sort_values("species").reset_index(drop=True)

    x = df_vals["species"].tolist()
    y = df_vals["value"].to_numpy()
    colors = [palette.get(s, (0.3,0.3,0.3,1.0)) for s in x]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Species")
    if scale == "absolute":
        ax.set_ylabel("Number of death trees")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    else:
        ax.set_ylabel("Number of death trees (%)")
        # osy v procentech 0-100
        if y.size:
            ymax = float(np.nanmax(y))
        else:
            ymax = 0.0
        # pro relative_toForest můžeme mít <1; posuň na pěkný krok 0.1
        if scale == "relative":
            ax.set_ylim(0, 1.01)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_yticklabels([f"{int(v*100)}" for v in np.linspace(0, 1, 6)])
        else:
            # relative_toForest
            top = max(0.1, np.ceil(ymax*10)/10.0)
            ax.set_ylim(0, top)
            ticks = np.linspace(0, top, num=int(top*10)+1 if top<=1.01 else 11)
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    if title:
        ax.set_title(title)

    # zlepšit čitelnost x-etiket
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return fig, ax

def show_tree_deaths_by_species(outputs: ILandOutputs,
                                scale: Scale = "relative_toForest",
                                preview: bool = True,
                                save: bool = False,
                                out_dir: str | os.PathLike = "output",
                                filename: str | None = None) -> Path | None:
    """
    Vytvoří graf úhynů podle druhu pro zvolenou škálu.
    - preview=True: otevře okno s grafem (plt.show())
    - save=True: uloží JPG 300 dpi do out_dir
    Vrací cestu k souboru (pokud se ukládalo), jinak None.
    """
    df_death = combine_with_ids(outputs.death)
    df_living = combine_with_ids(outputs.living)

    vals = deaths_by_species(df_death, df_living, scale=scale)
    title = {
        "absolute": "iLand: deaths by species (absolute)",
        "relative": "iLand: deaths by species (relative)",
        "relative_toForest": "iLand: deaths by species (relative to forest species presence)"
    }[scale]
    fig, ax = plot_bar_by_species(vals, palette=PALETTE, scale=scale, title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_deaths_by_species_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        # Uložení s 300 dpi, kvalitní JPEG (requires pillow)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview:
        plt.show()
    plt.close(fig)
    return out_path

# ==== Deaths by Year x Species (stacked counts) ====
def deaths_by_year_species(df_death: pd.DataFrame,
                           df_living: pd.DataFrame | None = None,
                           scale: Scale = "absolute") -> pd.DataFrame:
    """
    Počty úhynů podle (year, species) s R-like definicemi:
      - absolute: průměrný počet úhynů na replikaci (per year & species)
      - relative: podíl druhu v rámci roku (součet přes druhy v daném roce = 1)
      - relative_toForest: úhyny / počet živých jedinců druhu v daném roce
    Vrací wide tabulku: index=year, columns=species, values=value.
    """
    if df_death is None or df_death.empty:
        return pd.DataFrame()

    deaths = (df_death.groupby(["year", "species"])["id"]
              .size().rename("no_death").reset_index())

    if scale == "absolute":
        nreps = df_death["replication"].nunique() if "replication" in df_death else 1
        deaths["value"] = deaths["no_death"] / max(nreps, 1)

    elif scale == "relative":
         year_tot = (deaths.groupby("year")["no_death"].sum()
                .rename("year_total").reset_index())
         deaths = deaths.merge(year_tot, on="year", how="left")
         deaths["value"] = (deaths["no_death"] /
                            deaths["year_total"].replace(0, np.nan)).fillna(0.0)

    elif scale == "relative_toForest":
        if df_living is not None and not df_living.empty and "year" in df_living.columns:
            # DĚLÍME CELKOVÝM POČTEM ŽIVÝCH V DANÉM ROCE (všechny druhy)
            living_year = (df_living.groupby("year")["id"]
                           .size().rename("living_year_total").reset_index())
            tmp = deaths.merge(living_year, on="year", how="left")
            tmp["value"] = (tmp["no_death"] /
                            tmp["living_year_total"].replace(0, np.nan)).fillna(0.0)
            deaths = tmp
        else:
            # fallback: dělíme GLOBÁLNÍM počtem živých (všechny roky, všechny druhy)
            grand_living = float(df_living["id"].size) if (df_living is not None and not df_living.empty) else 0.0
            denom = grand_living if grand_living > 0 else 1.0
            deaths["value"] = deaths["no_death"] / denom
    else:
        raise ValueError("scale must be 'absolute', 'relative', or 'relative_toForest'")


    wide = (deaths.pivot(index="year", columns="species", values="value")
            .fillna(0.0).sort_index())
    # pořadí sloupců dle SPECIES_CODES (jako v R paletě)
    ordered = [s for s in SPECIES_CODES if s in wide.columns]
    extra = [s for s in wide.columns if s not in SPECIES_CODES]
    wide = wide.reindex(columns=ordered + extra)
    return wide

def plot_stacked_deaths_by_year_species(wide: pd.DataFrame,
                                        scale: Scale = "absolute",
                                        title: str | None = None):
    """
    Stacked bar: x=year, y=hustota/počet, fill=species.
    Styly os přiblížené R:
      - X: 0 .. max(year), kroky po 5
      - absolute: Y na stovky (ceil), kroky po 100
      - relative: Y 0..1.01, kroky 0.2, štítky v %
      - relative_toForest: Y 0..top (ceil na 0.1), kroky 0.1, štítky v %
    """
    if wide is None or wide.empty:
        fig, ax = plt.subplots(figsize=(11,6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    years = wide.index.to_list()
    species_cols = wide.columns.to_list()

    fig, ax = plt.subplots(figsize=(11,6))
    bottoms = np.zeros(len(years))
    for s in species_cols:
        vals = wide[s].to_numpy()
        ax.bar(years, vals, bottom=bottoms, label=s,
               color=PALETTE.get(s, None), edgecolor="black", linewidth=0.2)
        bottoms = bottoms + vals

    ax.set_xlabel("Calendar year")

    if scale == "absolute":
        ax.set_ylabel("Number of death trees")
        per_year_total = wide.sum(axis=1).to_numpy()
        ymax = float(np.nanmax(per_year_total)) if per_year_total.size else 0.0
        top = int(np.ceil(ymax/100.0)*100.0) if ymax > 0 else 100
        ax.set_ylim(0, top)
        ax.set_yticks(list(range(0, top+1, 100)))

    elif scale == "relative":
        ax.set_ylabel("Number of death trees (%)")
        ax.set_ylim(0, 1.01)
        ticks = np.linspace(0, 1, 6)  # 0,0.2,...,1
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    else:  # relative_toForest
        ax.set_ylabel("Number of death trees (%)")
        per_year_total = wide.sum(axis=1).to_numpy()
        ymax = float(np.nanmax(per_year_total)) if per_year_total.size else 0.0
        top = max(0.1, np.ceil(ymax*10)/10.0)
        ax.set_ylim(0, top)
        ticks = np.round(np.linspace(0, top, int(top*10)+1), 10)  # po 0.1
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    if title:
        ax.set_title(title)

    # X osa jako v R: od 0 do max(year), krok 5
    max_year = int(max(years))
    ax.set_xlim(-0.5, max_year + 0.5)
    ax.set_xticks(list(range(0, max_year+1, 5)))

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax

def show_tree_deaths_by_year_species(outputs: ILandOutputs,
                                     scale: Scale = "absolute",
                                     preview: bool = True,
                                     save: bool = False,
                                     out_dir: str | os.PathLike = "output",
                                     filename: str | None = None):
    """Wrapper k R: show.tree.deaths.byYearSpecies.noIndivs"""
    df_death = combine_with_ids(outputs.death)
    df_living = combine_with_ids(outputs.living)
    wide = deaths_by_year_species(df_death, df_living=df_living, scale=scale)
    title = f"iLand: deaths by year × species ({scale})"
    fig, ax = plot_stacked_deaths_by_year_species(wide, scale=scale, title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_deaths_by_year_species_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview:
        plt.show()
    plt.close(fig)
    return out_path

# ==== Deaths by Species — VOLUME (stacked by species) ====
def deaths_volume_by_species(df_death: pd.DataFrame,
                             df_living: pd.DataFrame | None,
                             scale: Literal["absolute","relative","relative_toForest"] = "absolute"
                             ) -> pd.DataFrame:
    """
    Agregace death volume (volume_m3) podle DRUHU napříč roky.
      - absolute: průměrný death volume na replikaci
      - relative: podíl druhu na CELKOVÉM death volume (všechna léta, všechny druhy)
      - relative_toForest: death volume druhu / living volume druhu (napříč roky)
    Vrací DataFrame: ['species','value'].
    """
    if df_death is None or df_death.empty:
        return pd.DataFrame(columns=["species","value"])

    vol = (df_death.groupby("species")["volume_m3"]
           .sum().rename("death_vol").reset_index())

    if scale == "absolute":
        nreps = df_death["replication"].nunique() if "replication" in df_death else 1
        vol["value"] = vol["death_vol"] / max(nreps, 1)

    elif scale == "relative":
        total = float(vol["death_vol"].sum()) or 1.0
        vol["value"] = vol["death_vol"] / total

    elif scale == "relative_toForest":
        if df_living is None or df_living.empty:
            total = float(vol["death_vol"].sum()) or 1.0
            vol["value"] = vol["death_vol"] / total
        else:
            live = (df_living.groupby("species")["volume_m3"]
                    .sum().rename("living_vol").reset_index())
            tmp = vol.merge(live, on="species", how="inner")
            tmp["value"] = (tmp["death_vol"] /
                            tmp["living_vol"].replace(0, np.nan)).fillna(0.0)
            vol = tmp
    else:
        raise ValueError("scale must be 'absolute','relative','relative_toForest'")

    out = vol[["species","value"]].copy()
    return out.sort_values("species").reset_index(drop=True)

def plot_bar_deaths_volume_by_species(df_vals: pd.DataFrame,
                                      palette: Dict[str, Tuple[float,float,float,float]] = PALETTE,
                                      scale: Literal["absolute","relative","relative_toForest"] = "absolute",
                                      title: str | None = None):
    """Sloupcový graf death volume podle druhu (styling dle R)."""
    if df_vals.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    df_vals = df_vals.sort_values("species").reset_index(drop=True)
    x = df_vals["species"].tolist()
    y = df_vals["value"].to_numpy()
    colors = [palette.get(s, (0.3,0.3,0.3,1.0)) for s in x]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x, y, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Species")

    if scale == "absolute":
        ax.set_ylabel("Death volume")
        ymax = float(np.nanmax(y)) if y.size else 0.0
        top = int(np.ceil(ymax/10.0)*10.0) if ymax > 0 else 10   # ceiling(max/10)*10
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top+1, 25))                   # breaks seq(0, 1000, 25) v R šlo obecně
    elif scale == "relative":
        ax.set_ylabel("Death volume (%)")
        ax.set_ylim(0, 1.01)
        ticks = np.linspace(0, 1, 6)  # 0,0.2,...,1
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])
    else:
        ax.set_ylabel("Death volume (%)")
        ymax = float(np.nanmax(y)) if y.size else 0.0
        top = max(0.1, np.ceil(ymax*10)/10.0)                    # ceiling(max*10)/10
        ax.set_ylim(0, top)
        ticks = np.linspace(0, top, num=int(top*10)+1 if top<=1.01 else 11)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    if title:
        ax.set_title(title)
    ax.tick_params(axis='x', rotation=90)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    return fig, ax

def show_deaths_volume_by_species(outputs: ILandOutputs,
                                  scale: Literal["absolute","relative","relative_toForest"] = "absolute",
                                  preview: bool = True,
                                  save: bool = False,
                                  out_dir: str | os.PathLike = "output",
                                  filename: str | None = None) -> Path | None:
    df_death = combine_with_ids(outputs.death)
    df_living = combine_with_ids(outputs.living)
    vals = deaths_volume_by_species(df_death, df_living, scale=scale)
    title = {
        "absolute": "iLand: death volume by species (absolute)",
        "relative": "iLand: death volume by species (relative)",
        "relative_toForest": "iLand: death volume by species (relative to forest volume)"
    }[scale]
    fig, ax = plot_bar_deaths_volume_by_species(vals, palette=PALETTE, scale=scale, title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_death_volume_by_species_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview: plt.show()
    plt.close(fig)
    return out_path

# ==== Deaths by Year × Species — VOLUME (stacked by year) ====
def deaths_volume_by_year_species(df_death: pd.DataFrame,
                                  df_living: pd.DataFrame | None,
                                  scale: Literal["absolute","relative","relative_toForest"] = "absolute"
                                  ) -> pd.DataFrame:
    """
    Agregace death volume (volume_m3) podle (year, species).
      - absolute: průměrný death volume na replikaci v daném roce & druhu
      - relative: podíl death volume druhu v rámci ROKU (součet přes druhy v roce = 1)
      - relative_toForest: death volume / CELKOVÉ living volume v daném roce (všechny druhy)
    Vrací wide: index=year, columns=species, values=value.
    """
    if df_death is None or df_death.empty:
        return pd.DataFrame()

    vol = (df_death.groupby(["year","species"])["volume_m3"]
           .sum().rename("death_vol").reset_index())

    if scale == "absolute":
        nreps = df_death["replication"].nunique() if "replication" in df_death else 1
        vol["value"] = vol["death_vol"] / max(nreps, 1)

    elif scale == "relative":
        per_year = (vol.groupby("year")["death_vol"].sum()
                    .rename("year_total").reset_index())
        vol = vol.merge(per_year, on="year", how="left")
        vol["value"] = (vol["death_vol"] /
                        vol["year_total"].replace(0, np.nan)).fillna(0.0)

    elif scale == "relative_toForest":
        if df_living is not None and not df_living.empty:
            living_year = (df_living.groupby("year")["volume_m3"]
                           .sum().rename("living_year_vol").reset_index())
            tmp = vol.merge(living_year, on="year", how="left")
            tmp["value"] = (tmp["death_vol"] /
                            tmp["living_year_vol"].replace(0, np.nan)).fillna(0.0)
            vol = tmp
        else:
            # fallback: dělíme globálním living volume (všechna léta, všechny druhy)
            grand_living = float(df_living["volume_m3"].sum()) if (df_living is not None and not df_living.empty) else 0.0
            denom = grand_living if grand_living > 0 else 1.0
            vol["value"] = vol["death_vol"] / denom
    else:
        raise ValueError("scale must be 'absolute','relative','relative_toForest'")

    wide = (vol.pivot(index="year", columns="species", values="value")
            .fillna(0.0).sort_index())
    ordered = [s for s in SPECIES_CODES if s in wide.columns]
    extra = [s for s in wide.columns if s not in SPECIES_CODES]
    return wide.reindex(columns=ordered + extra)

def plot_stacked_deaths_volume_by_year_species(wide: pd.DataFrame,
                                               scale: Literal["absolute","relative","relative_toForest"] = "absolute",
                                               title: str | None = None):
    """Stacked bar death volume přes roky; osa X jako v R: -0.5 .. max+0.5, kroky po 5."""
    if wide is None or wide.empty:
        fig, ax = plt.subplots(figsize=(11,6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    years = wide.index.to_list()
    species_cols = wide.columns.to_list()

    fig, ax = plt.subplots(figsize=(11,6))
    bottoms = np.zeros(len(years))
    for s in species_cols:
        vals = wide[s].to_numpy()
        ax.bar(years, vals, bottom=bottoms, label=s,
               color=PALETTE.get(s, None), edgecolor="black", linewidth=0.2)
        bottoms = bottoms + vals

    ax.set_xlabel("Calendar year")

    if scale == "absolute":
        ax.set_ylabel("Death volume")
        per_year_total = wide.sum(axis=1).to_numpy()
        ymax = float(np.nanmax(per_year_total)) if per_year_total.size else 0.0
        # R: ceiling(max/10)*10 ; na ose R měl i jemné breaks po 2.5 s jedním desetinným místem
        top = np.ceil(ymax/10.0)*10.0 if ymax > 0 else 10.0
        ax.set_ylim(0, top)
        ticks = np.arange(0, top+2.5, 2.5)  # 0, 2.5, 5.0, ...
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:.1f}" for t in ticks])
    elif scale == "relative":
        ax.set_ylabel("Death volume (%)")
        ax.set_ylim(0, 1.01)
        ticks = np.linspace(0, 1, 6)  # 0,0.2,...,1
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])
    else:
        ax.set_ylabel("Death volume (%)")
        per_year_total = wide.sum(axis=1).to_numpy()
        ymax = float(np.nanmax(per_year_total)) if per_year_total.size else 0.0
        top = max(0.01, np.ceil(ymax*100)/100.0)   # ceiling(max*100)/100
        ax.set_ylim(0, top)
        ticks = np.arange(0, top+0.01, 0.01)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    if title:
        ax.set_title(title)

    max_year = int(max(years))
    ax.set_xlim(-0.5, max_year + 0.5)            # -0.5 .. max+0.5
    ax.set_xticks(list(range(0, max_year+1, 5))) # kroky po 5

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax

def show_deaths_volume_by_year_species(outputs: ILandOutputs,
                                       scale: Literal["absolute","relative","relative_toForest"] = "absolute",
                                       preview: bool = True,
                                       save: bool = False,
                                       out_dir: str | os.PathLike = "output",
                                       filename: str | None = None) -> Path | None:
    df_death = combine_with_ids(outputs.death)
    df_living = combine_with_ids(outputs.living)
    wide = deaths_volume_by_year_species(df_death, df_living, scale=scale)
    title = f"iLand: death volume by year × species ({scale})"
    fig, ax = plot_stacked_deaths_volume_by_year_species(wide, scale=scale, title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_death_volume_by_year_species_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview: plt.show()
    plt.close(fig)
    return out_path

# ==== Trees Composition by Species Presence (stacked counts) ====
def trees_by_species_presence(df_living: pd.DataFrame,
                              scale: Literal["absolute","relative"] = "absolute") -> pd.DataFrame:
    """
    Složení porostu podle přítomnosti stromů (počty jedinců).
      - absolute: průměrný počet stromů na replikaci (per year × species)
      - relative: podíl druhu na celkovém počtu stromů v daném roce (součet přes druhy v roce = 1)
    Vrací wide tabulku: index=year, columns=species, values=value.
    """
    if df_living is None or df_living.empty:
        return pd.DataFrame()

    # počty stromů podle year × species
    trees = (df_living.groupby(["year", "species"])["id"]
             .size().rename("no_trees").reset_index())

    if scale == "absolute":
        nreps = df_living["replication"].nunique() if "replication" in df_living else 1
        trees["value"] = trees["no_trees"] / max(nreps, 1)

    elif scale == "relative":
        # podíl druhu v rámci ROKU
        per_year = (df_living.groupby("year")["id"]
                    .size().rename("year_total").reset_index())
        trees = trees.merge(per_year, on="year", how="left")
        trees["value"] = (trees["no_trees"] /
                          trees["year_total"].replace(0, np.nan)).fillna(0.0)
    else:
        raise ValueError("scale must be 'absolute' or 'relative'")

    wide = (trees.pivot(index="year", columns="species", values="value")
            .fillna(0.0).sort_index())

    # pořadí druhů podle SPECIES_CODES
    ordered = [s for s in SPECIES_CODES if s in wide.columns]
    extra = [s for s in wide.columns if s not in SPECIES_CODES]
    return wide.reindex(columns=ordered + extra)

def plot_trees_by_species_presence(wide: pd.DataFrame,
                                   scale: Literal["absolute","relative"] = "absolute",
                                   title: str | None = None):
    """Stacked bar: podíl nebo absolutní počty stromů (styling dle R)."""
    if wide is None or wide.empty:
        fig, ax = plt.subplots(figsize=(11,6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    years = wide.index.to_list()
    species_cols = wide.columns.to_list()

    fig, ax = plt.subplots(figsize=(11,6))
    bottoms = np.zeros(len(years))
    for s in species_cols:
        vals = wide[s].to_numpy()
        ax.bar(years, vals, bottom=bottoms,
               label=s, color=PALETTE.get(s, None),
               edgecolor="black", linewidth=0.2)
        bottoms += vals

    ax.set_xlabel("Calendar year")

    if scale == "absolute":
        ax.set_ylabel("Number of trees")
        per_year_total = wide.sum(axis=1).to_numpy()
        ymax = float(np.nanmax(per_year_total)) if per_year_total.size else 0.0
        top = int(np.ceil(ymax/100.0)*100.0) if ymax > 0 else 100
        ax.set_ylim(0, top)
        ax.set_yticks(list(range(0, top+1, 100)))

    elif scale == "relative":
        ax.set_ylabel("Number of trees (%)")
        ax.set_ylim(0, 1.01)
        ticks = np.linspace(0, 1, 6)  # 0,0.2,...,1
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int(v*100)}" for v in ticks])

    if title:
        ax.set_title(title)

    # osa X: od -0.5 do max_year+0.5 (R styl), kroky po 5 letech
    max_year = int(max(years))
    ax.set_xlim(-0.5, max_year + 0.5)
    ax.set_xticks(list(range(0, max_year+1, 5)))

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=4, fontsize=8, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax

def show_trees_by_species_presence(outputs: ILandOutputs,
                                   scale: Literal["absolute","relative"] = "absolute",
                                   preview: bool = True,
                                   save: bool = False,
                                   out_dir: str | os.PathLike = "output",
                                   filename: str | None = None) -> Path | None:
    """Wrapper pro R: show.trees.bySpecies.noIndivs"""
    df_living = combine_with_ids(outputs.living)
    wide = trees_by_species_presence(df_living, scale=scale)
    title = {
        "absolute": "iLand: trees by species presence (absolute)",
        "relative": "iLand: trees by species presence (relative)"
    }[scale]
    fig, ax = plot_trees_by_species_presence(wide, scale=scale, title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_trees_by_species_presence_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview:
        plt.show()
    plt.close(fig)
    return out_path

# ==== Trees by species volume over years (stacked bars) ====
def trees_by_species_volume(df_living: pd.DataFrame,
                            scale: Scale = "absolute") -> pd.DataFrame:
    """
    Aggreguje objem (volume_m3) podle species a year.
    Vrací pivot tabulku (index=year, columns=species, values=tree_vol),
    kde hodnoty jsou:
      - absolute: průměrný objem na replikaci (sum/počet replikací)
      - relative: podíl objemu daného druhu na celkovém objemu v daném roce (0-1)
    """
    if df_living is None or df_living.empty:
        return pd.DataFrame()

    # sum volume per species-year across all reps
    agg = (df_living.groupby(["species", "year"])["volume_m3"]
           .sum().rename("tree_vol").reset_index())

    if scale == "absolute":
        nreps = df_living["replication"].nunique() if "replication" in df_living else 1
        agg["tree_vol"] = agg["tree_vol"] / max(nreps, 1)

    elif scale == "relative":
        # total volume per year
        tot = (df_living.groupby("year")["volume_m3"]
               .sum().rename("total_year").reset_index())
        agg = agg.merge(tot, on="year", how="left")
        agg["tree_vol"] = agg["tree_vol"] / agg["total_year"].replace(0, np.nan)
        agg["tree_vol"] = agg["tree_vol"].fillna(0.0)

    else:
        raise ValueError("scale must be 'absolute' or 'relative' for volume plot")

    # pivot to wide for stacked bars
    wide = agg.pivot(index="year", columns="species", values="tree_vol").fillna(0.0)
    # sort by year and ensure species columns have consistent order
    wide = wide.sort_index()
    all_species = [s for s in SPECIES_CODES if s in wide.columns] + \
                  [s for s in wide.columns if s not in SPECIES_CODES]
    wide = wide.reindex(columns=all_species)
    return wide

def plot_stacked_volume_by_species_over_years(wide: pd.DataFrame,
                                              scale: Scale = "absolute",
                                              title: str | None = None):
    """
    Vykreslí stacked bar graf: x=year, y=sum volume, fill=species.
    """
    if wide is None or wide.empty:
        fig, ax = plt.subplots(figsize=(9,6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return fig, ax

    years = wide.index.to_list()
    species_cols = wide.columns.to_list()

    fig, ax = plt.subplots(figsize=(11,6))
    bottoms = np.zeros(len(years))
    for s in species_cols:
        vals = wide[s].to_numpy()
        ax.bar(years, vals, bottom=bottoms, label=s, color=PALETTE.get(s, None), edgecolor="black", linewidth=0.2)
        bottoms = bottoms + vals

    ax.set_xlabel("Calendar year")
    if scale == "absolute":
        ax.set_ylabel("Living volume (m³)")
        # nice y ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    else:
        ax.set_ylabel("Living volume (%)")
        ax.set_ylim(0, 1.01)
        ax.set_yticks(np.linspace(0,1,6))
        ax.set_yticklabels([f"{int(v*100)}" for v in np.linspace(0,1,6)])

    if title:
        ax.set_title(title)
    # x ticks every 5 years if the range is large
    if len(years) > 0:
        ymin, ymax = min(years), max(years)
        step = 1 if (ymax - ymin) <= 15 else 5
        ax.set_xticks(list(range(int(ymin), int(ymax)+1, step)))

    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig, ax

def show_trees_by_species_vol(outputs: ILandOutputs,
                              scale: str = "absolute",
                              preview: bool = True,
                              save: bool = False,
                              out_dir: str | os.PathLike = "output",
                              filename: str | None = None) -> Path | None:
    """
    Převod R funkce show.trees.bySpecies.vol:
      - scale='absolute' -> průměr objemu na replikaci
      - scale='relative' -> podíl objemu druhu v rámci roku
    """
    df_living = combine_with_ids(outputs.living)
    wide = trees_by_species_volume(df_living, scale="absolute" if scale=="absolute" else "relative")
    title = "iLand: living volume by species over years ({})".format(scale)
    fig, ax = plot_stacked_volume_by_species_over_years(wide, scale="absolute" if scale=="absolute" else "relative", title=title)

    out_path: Path | None = None
    if save:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"iland_volume_by_species_{scale}_{safe_name(pd.Timestamp.today().date().isoformat())}.jpg"
        out_path = out_path / filename
        fig.savefig(out_path, dpi=300, bbox_inches="tight", format="jpg")
    if preview:
        plt.show()
    plt.close(fig)
    return out_path