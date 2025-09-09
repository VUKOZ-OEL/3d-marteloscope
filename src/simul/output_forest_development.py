# --- imports ---
import sqlite3
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict
from typing import Optional

from outputs_graphs import read_outputs_tree_level

# --- barevná paleta (vč. 'death') ---
SPECIES = [
    "abal","abam","abgr","abpr","acca","acma","acpl","acps","acru",
    "algl","alin","alru","alvi","bepe","cabe","casa","coav","fasy",
    "frex","lade","piab","pice","pini","pipo","pisi","pisy","poni",
    "potr","psme","qupe","qupu","quro","reader","rops","saca","soar",
    "soau","thpl","tico","tipl","tshe","tsme","ulgl"
]
# použijeme matplotlib colormap 'turbo' (odpovídá viridis::turbo)
cmap = plt.get_cmap("turbo", len(SPECIES))
species_colors = {sp: cmap(i) for i, sp in enumerate(SPECIES)}
species_colors["death"] = "#CCCCCC"

# ======================================
# 2) PREP: vybrat stromy k "cutnutí"
# ======================================
def prepare_dataset(outputs, n_reps: Optional[int] = None, mark_death: bool = True) -> pd.DataFrame:
    """
    Z iLand výstupů připraví finální DataFrame pro kreslení:
      - sloučí tabulky treeremoved (death) a tree (living) přes replikace
      - pro každý rok spočte cílový objem k odumření per species (mean volume_m3)
      - seřadí kandidáty (id) dle pravděpodobnosti úhynu a kumulovaného objemu, vybere do cíle
      - z living průměruje DBH by (id,year,species,x,y)
      - pokud mark_death=True: od cut_year dál přepíše species -> "death"
        jinak řádky s year >= cut_year odfiltruje
    Vrací: pandas.DataFrame se sloupci id, year, species, x, y, dbh (a bez cut_year).
    """

    # --- rozlišení vstupu: ILandOutputs vs dict ---
    if hasattr(outputs, "death") and hasattr(outputs, "living"):
        death_dict  = outputs.death
        living_dict = outputs.living
    elif isinstance(outputs, dict) and "output_death" in outputs and "output_living" in outputs:
        death_dict  = outputs["output_death"]
        living_dict = outputs["output_living"]
    else:
        raise TypeError("prepare_dataset: očekávám ILandOutputs (.death/.living) nebo dict s 'output_death'/'output_living'.")

    # --- pomocná: spojení replikací do jednoho DF ---
    def combine_with_ids(dct: dict) -> pd.DataFrame:
        if not dct:
            return pd.DataFrame()
        parts = []
        for rep, df in dct.items():
            if df is None or df.empty:
                continue
            if "replication" not in df.columns:
                df = df.copy()
                df["replication"] = rep
            parts.append(df)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    death_trees  = combine_with_ids(death_dict)   # očekává sloupce: id, species, year, volume_m3, ...
    living_trees = combine_with_ids(living_dict)  # očekává sloupce: id, species, year, x, y, dbh, ...

    # kolik replikací pro výpočet pravděpodobnosti (pokud není dáno)
    if n_reps is None:
        n_reps = len(death_dict) if isinstance(death_dict, dict) else 100

    # --- pokud nejsou žádné "death" záznamy, vrať jen průměrovaný output ---
    if death_trees.empty:
        output = (
            living_trees.groupby(["id", "year", "species", "x", "y"], as_index=False)["dbh"]
            .mean()
            .sort_values(["year", "id"])
            .reset_index(drop=True)
        )
        return output

    # -------------------------
    # 1) výběr trees_to_cut
    # -------------------------
    trees_to_cut = []

    for yr in sorted(death_trees["year"].dropna().unique()):
        sub = death_trees.loc[death_trees["year"] == yr]

        # target objem per species (mean volume_m3 v tomto roce)
        deth_vol = sub.groupby("species", as_index=False)["volume_m3"].mean()

        # pravděpodobnost (id,species): počet výskytů / n_reps
        prob = (
            sub.groupby(["id", "species"], as_index=False)["year"]
               .count()
               .rename(columns={"year": "prob"})
        )
        prob["prob"] = prob["prob"] / float(n_reps)

        # průměrný objem per id v tomto roce
        vol_by_id = sub.groupby("id", as_index=False)["volume_m3"].mean()

        # spojit info do jedné tabulky
        prob = prob.merge(vol_by_id, on="id", how="left").rename(columns={"species": "spp"})

        # pro každý druh naplň cíl
        for sp in deth_vol["species"].unique():
            target = float(deth_vol.loc[deth_vol["species"] == sp, "volume_m3"].values[0])
            cand = prob.loc[prob["spp"] == sp].copy()
            if cand.empty or target <= 0:
                continue

            cand = cand.sort_values(["prob", "volume_m3"], ascending=[False, False]).reset_index(drop=True)
            cand["cum"] = cand["volume_m3"].cumsum()

            # první index, kde kumulace dosáhne cíle
            hit = np.argmax(cand["cum"].values >= target)
            chosen = cand["id"].values if cand["cum"].values[hit] < target else cand["id"].values[: hit + 1]

            trees_to_cut.extend([(int(yr), int(tid)) for tid in chosen])

    trees_to_cut = pd.DataFrame(trees_to_cut, columns=["year", "tree_id"])

    # -------------------------
    # 2) první rok cutnutí
    # -------------------------
    if trees_to_cut.empty:
        first_cut = pd.DataFrame(columns=["id", "cut_year"])
    else:
        first_cut = (
            trees_to_cut.groupby("tree_id", as_index=False)["year"]
                       .min()
                       .rename(columns={"tree_id": "id", "year": "cut_year"})
        )

    # -------------------------
    # 3) průměrný output z living
    # -------------------------
    output = (
        living_trees.groupby(["id", "year", "species", "x", "y"], as_index=False)["dbh"]
        .mean()
    )

    # -------------------------
    # 4) aplikace cut info
    # -------------------------
    out = output.merge(first_cut, on="id", how="left")

    if mark_death:
        mask = out["cut_year"].notna() & (out["year"] >= out["cut_year"])
        out.loc[mask, "species"] = "death"
        out = out.drop(columns=["cut_year"])
    else:
        out = out[(out["cut_year"].isna()) | (out["year"] < out["cut_year"])].drop(columns=["cut_year"])

    return out.sort_values(["year", "id"]).reset_index(drop=True)

def prepare_dataset_old(db: dict, n_reps: int = None, mark_death: bool = True) -> pd.DataFrame:
    """
    Připraví finální dataset stejně jako R:
      - death_trees: spojí treeremoved
      - living_trees: spojí tree
      - pro každý rok i:
          * target objem k odumření per species = mean(volume_m3) z treeremoved (stejně jako v R)
          * spočítá 'pravděpodobnost' pro id × species = (#výskytů v treeremoved) / n_reps
          * spojí průměrný volume_m3 per id z treeremoved
          * pro každý species seřadí kácení podle pravděpodobnosti, kumuluje volume_m3,
            a vybere stromy, dokud kumulace nedosáhne targetu → trees_to_cut (year, tree_id)
      - z living_trees udělá 'output' = mean(dbh) by id,year,species,x,y
      - pokud mark_death:
            přepíše species = 'death' od prvního cut_year výše
        jinak:
            odfiltruje řádky year >= cut_year

    Vrací output_final (pandas DataFrame).
    """
    # 2.1 spojení
    death_trees = pd.concat(db["output_death"].values(), ignore_index=True)
    living_trees = pd.concat(db["output_living"].values(), ignore_index=True)

    # kolik replikací? (pro pravděp.)
    if n_reps is None:
        n_reps = len(db["output_death"])

    # 2.2 připrav trees_to_cut
    trees_to_cut = []

    # roky, kde je něco v treeremoved
    for yr in sorted(death_trees["year"].dropna().unique()):
        sub_death = death_trees.loc[death_trees["year"] == yr].copy()

        # target objem per species = mean(volume_m3) (stejně jako v R)
        deth_vol = sub_death.groupby("species", as_index=False)["volume_m3"].mean()

        # pravděpodobnost: výskyt v treeremoved za rok pro (id, species)
        prob = (
            sub_death.groupby(["id", "species"], as_index=False)["year"]
            .count()
            .rename(columns={"year": "prob"})
        )
        prob["prob"] = prob["prob"] / float(n_reps)

        # průměrný objem per id (napříč replikacemi) v tomto roce
        vol_by_id = sub_death.groupby("id", as_index=False)["volume_m3"].mean()

        # spojíme, ať máme pro každý id: (species, prob, volume_m3)
        prob = prob.merge(vol_by_id, on="id", how="left").rename(columns={"species": "spp"})

        # pro každý druh vyber kandidáty do cílového objemu
        for sp in deth_vol["species"].unique():
            target = float(deth_vol.loc[deth_vol["species"] == sp, "volume_m3"].values[0])
            sub = prob.loc[prob["spp"] == sp].copy()
            if sub.empty or target <= 0:
                continue

            sub = sub.sort_values(["prob", "volume_m3"], ascending=[False, False]).reset_index(drop=True)
            sub["cum"] = sub["volume_m3"].cumsum()

            # index prvního, kde kumulace >= target
            idx = np.argmax(sub["cum"].values >= target)
            if sub["cum"].values[idx] < target:
                # nedosáhne cíle ani se všemi stromky → bereme všechny
                chosen = sub["id"].values
            else:
                chosen = sub["id"].values[: idx + 1]

            # ulož (year, id)
            trees_to_cut.extend([(int(yr), int(tid)) for tid in chosen])

    trees_to_cut = pd.DataFrame(trees_to_cut, columns=["year", "tree_id"])
    if trees_to_cut.empty:
        # žádné kácení – jen připrav output a vrať
        output = (
            living_trees.groupby(["id", "year", "species", "x", "y"], as_index=False)["dbh"]
            .mean()
        )
        return output.sort_values(["year", "id"]).reset_index(drop=True)

    # 2.3 první rok, kdy má být strom vyřazen
    first_cut = (
        trees_to_cut.groupby("tree_id", as_index=False)["year"]
        .min()
        .rename(columns={"tree_id": "id", "year": "cut_year"})
    )

    # 2.4 output = průměr DBH (jako v R)
    output = (
        living_trees.groupby(["id", "year", "species", "x", "y"], as_index=False)["dbh"]
        .mean()
    )

    # 2.5 spojíme cut info
    out = output.merge(first_cut, on="id", how="left")

    if mark_death:
        # změna species na "death" od cut_year dál
        mask = out["cut_year"].notna() & (out["year"] >= out["cut_year"])
        out.loc[mask, "species"] = "death"
        out = out.drop(columns=["cut_year"])
    else:
        # R-varianta: řádky s year >= cut_year zahodit
        out = out[(out["cut_year"].isna()) | (out["year"] < out["cut_year"])].copy()
        out = out.drop(columns=["cut_year"])

    out = out.sort_values(["year", "id"]).reset_index(drop=True)
    return out

# ======================================
# 3) RUN: nekumulativní animace
# ======================================
def run_forest_development(df: pd.DataFrame,
                           xlim=(0, 100),
                           ylim=(0, 100),
                           point_alpha=0.9,
                           fps=4,
                           save_path=None):
    """
    Nekumulativní animace: pro každý rok vykreslí jen body z toho roku.
    - df je výstup z prepare_dataset()
    - pokud save_path je None: jen zobrazí; jinak uloží GIF/MP4 podle přípony ('.gif' nebo '.mp4')
    """
    # seřadit a nachystat
    df = df.sort_values(["year", "id"]).reset_index(drop=True)
    years = sorted(df["year"].unique())

    # mapování barev
    def color_for_species(series):
        return [species_colors.get(s, "#000000") for s in series]

    # init figure
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    scat = ax.scatter([], [], s=[], c=[])
    title = ax.set_title("Rok: ")

    # škála velikosti (převod z DBH cm na plochu markeru)
    dbh_min, dbh_max = float(df["dbh"].min()), float(df["dbh"].max())
    s_min, s_max = 20, 250  # zhruba jako ggplot range = c(2, 9), ale v pixelech

    def size_scale(dbh_values):
        # lineární škála
        if dbh_max == dbh_min:
            return np.full_like(dbh_values, (s_min + s_max) / 2.0, dtype=float)
        return s_min + (dbh_values - dbh_min) * (s_max - s_min) / (dbh_max - dbh_min)

    # frame updater
    def update(frame_idx):
        yr = years[frame_idx]
        sub = df[df["year"] == yr]
        sizes = size_scale(sub["dbh"].values)
        colors = color_for_species(sub["species"].values)
        scat.set_offsets(sub[["x", "y"]].values)
        scat.set_sizes(sizes)
        scat.set_color(colors)
        scat.set_alpha(point_alpha)
        title.set_text(f"Rok: {yr}")
        return scat, title

    anim = FuncAnimation(fig, update, frames=len(years), interval=1000//fps, blit=False)

    if save_path is None:
        plt.show()
    else:
        # podle přípony uložíme GIF/MP4
        ext = Path(save_path).suffix.lower()
        if ext == ".gif":
            try:
                import imageio
            except ImportError:
                raise ImportError("Pro uložení GIF je potřeba 'imageio': pip install imageio")
            # uložení přes imageio (bez dočasných PNG)
            imgs = []
            for i in range(len(years)):
                update(i)
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                imgs.append(img)
            imageio.mimsave(save_path, imgs, fps=fps, loop=0)
        elif ext == ".mp4":
            try:
                from matplotlib.animation import FFMpegWriter
            except Exception as e:
                raise RuntimeError("Pro MP4 potřebuješ nainstalovaný ffmpeg.") from e
            writer = FFMpegWriter(fps=fps)
            anim.save(save_path, writer=writer)
        else:
            raise ValueError("Podporované přípony: .gif nebo .mp4")

    return anim

# =========================
# 4) POUŽITÍ
# =========================
# database = read_outputs_tree_level("database_examples_3")
# df_final = prepare_dataset(database, mark_death=True)  # nebo mark_death=False pro mazání řádků
# run_forest_development(df_final, xlim=(0,100), ylim=(0,100))                 # zobrazí
# run_forest_development(df_final, save_path="forest.gif")                      # uloží GIF
# run_forest_development(df_final, save_path="forest.mp4")                      # uloží MP4

