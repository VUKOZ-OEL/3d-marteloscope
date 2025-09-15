from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import sqlite3
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go




mark_death = False

@dataclass
class ILandOutputs:
    living: Dict[str, pd.DataFrame]
    death: Dict[str, pd.DataFrame]

mean_output_cols = [
    "dbh",
    "height",
    "basalArea",
    "volume_m3",
    "leafArea_m2",
    "foliageMass",
    "stemMass",
    "branchMass",
    "fineRootMass",
    "coarseRootMass"
]
# =========================================================
# ČTENÍ DATABÁZÍ
# =========================================================
def _read_sqlite_table(path: Path, table_name: str) -> pd.DataFrame:
    con = sqlite3.connect(str(path))
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
    finally:
        con.close()
    return df

def read_outputs_tree_level(root_folder: str) -> ILandOutputs:
    root = Path(root_folder)
    if not root.exists():
        return ILandOutputs(living={}, death={})
    files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".db",".sqlite"}])
    living, death = {}, {}
    pad = len(str(max(1, len(files))))
    for i, f in enumerate(files, 1):
        rep = f"rep_{str(i).zfill(pad)}"
        try:
            df_tree = _read_sqlite_table(f, "tree")
            if "replication" not in df_tree.columns:
                df_tree["replication"] = rep
            living[rep] = df_tree
        except Exception:
            continue
        try:
            df_removed = _read_sqlite_table(f, "treeremoved")
            if "replication" not in df_removed.columns:
                df_removed["replication"] = rep
            death[rep] = df_removed
        except Exception:
            pass
    return ILandOutputs(living=living, death=death)

def combine_with_ids(mapping: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(mapping.values(), ignore_index=True) if mapping else pd.DataFrame()

# ---------- Příprava per-tree datasetu (bez závislosti na output_forest_development) ----------
def prepare_tree_dataset(outputs: ILandOutputs, mark_death: bool = True) -> pd.DataFrame:
    """
    Sloučí replikace a aplikuje logiku výběru/označení úhynu:
      - Cíl odumření per species = mean(volume_m3) v daném roce z treeremoved
      - Kandidáty (id,species) seřadí dle pravděpodobnosti (výskyt/n_reps) a volume_m3, kumuluje do cíle
      - Z living dělá průměr DBH by id,year,species,x,y
      - Pokud mark_death=True, od cut_year výš přepíše species -> 'death', jinak řádky >= cut_year odfiltruje
    """
    death_trees  = combine_with_ids(outputs.death)
    living_trees = combine_with_ids(outputs.living)
    living_trees.to_feather("C:/Users/krucek/Documents/iLand/test/rep_out/living.feather")
    n_reps = len(outputs.death) if isinstance(outputs.death, dict) else 100

    # Pokud nejsou 'death' záznamy → vrať průměrný output
    if death_trees.empty:
        out = (living_trees.groupby(["id","year","species","x","y"], as_index=False)[mean_output_cols]
               .mean().sort_values(["year","id"]).reset_index(drop=True))
        return out

    trees_to_cut: List[Tuple[int,int]] = []
    for yr in sorted(death_trees["year"].dropna().unique()):
        sub = death_trees.loc[death_trees["year"] == yr]
        deth_vol = sub.groupby("species", as_index=False)["volume_m3"].mean()
        prob = (sub.groupby(["id","species"], as_index=False)["year"].count()
                  .rename(columns={"year":"prob"}))
        prob["prob"] = prob["prob"] / float(n_reps)
        vol_by_id = sub.groupby("id", as_index=False)["volume_m3"].mean()
        prob = prob.merge(vol_by_id, on="id", how="left").rename(columns={"species":"spp"})

        for sp in deth_vol["species"].unique():
            target = float(deth_vol.loc[deth_vol["species"] == sp, "volume_m3"].values[0])
            cand = prob.loc[prob["spp"] == sp].copy()
            if cand.empty or target <= 0: continue
            cand = cand.sort_values(["prob","volume_m3"], ascending=[False, False]).reset_index(drop=True)
            cand["cum"] = cand["volume_m3"].cumsum()
            hit = np.argmax(cand["cum"].values >= target)
            chosen = cand["id"].values if cand["cum"].values[hit] < target else cand["id"].values[:hit+1]
            trees_to_cut.extend([(int(yr), int(tid)) for tid in chosen])

    trees_to_cut = pd.DataFrame(trees_to_cut, columns=["year","tree_id"])
    if trees_to_cut.empty:
        out = (living_trees.groupby(["id","year","species","x","y"], as_index=False)[mean_output_cols]
               .mean().sort_values(["year","id"]).reset_index(drop=True))
        return out

    first_cut = (trees_to_cut.groupby("tree_id", as_index=False)["year"]
                 .min().rename(columns={"tree_id":"id", "year":"cut_year"}))

    output = (living_trees.groupby(["id","year","species","x","y"], as_index=False)[mean_output_cols]
              .mean())
    out = output.merge(first_cut, on="id", how="left")

    if mark_death:
        mask = out["cut_year"].notna() & (out["year"] >= out["cut_year"])
        out.loc[mask, "species"] = "death"
        out = out.drop(columns=["cut_year"])
    else:
        out = out[(out["cut_year"].isna()) | (out["year"] < out["cut_year"])].drop(columns=["cut_year"])

    return out.sort_values(["year","id"]).reset_index(drop=True)
# ---------- Load simulations  --------------
def load_simulation(root: str):

    outputs = read_outputs_tree_level(root)
    #df_death = pd.concat(outputs.death.values(), ignore_index=True) if outputs and outputs.death else pd.DataFrame()
    #df_living = pd.concat(outputs.living.values(), ignore_index=True) if outputs and outputs.living else pd.DataFrame()

    tree_df = prepare_tree_dataset(outputs, mark_death)
    return tree_df

# ------------------------------------------------------------------------------------------------------
# For coloring plots

def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def latin_to_code(latin: str) -> str:
    """
    Kód = první 2 písmena z každého slova (bez diakritiky, jen [a-z]).
    'Carpinus betulus' -> 'cabe'
    'Populus × canadensis' -> 'poca'
    """
    s = _strip_accents(latin).lower()
    parts = s.split()
    bits: List[str] = []
    for tok in parts:
        tok = "".join(ch for ch in tok if "a" <= ch <= "z")
        if tok:
            bits.append(tok[:2])
    return "".join(bits)

def latin_binomial_label(latin_raw: str) -> str:
    """
    Vytvoří botanicky stylizovaný label: první alfabetický token -> 'Genus' (První písmeno velké),
    všechny další alfabetické tokeny -> malými písmeny.
    Nealfabetické tokeny (např. '×') zachová.
    Příklady:
      'carpinus betulus'         -> 'Carpinus betulus'
      'POPULUS × CANADENSIS'     -> 'Populus × canadensis'
      'Quercus robur fastigiata' -> 'Quercus robur fastigiata'
    """
    parts = latin_raw.strip().split()
    out: List[str] = []
    first_alpha_done = False
    for tok in parts:
        if any(c.isalpha() for c in tok):
            if not first_alpha_done:
                out.append(tok[0].upper() + tok[1:].lower())
                first_alpha_done = True
            else:
                out.append(tok.lower())
        else:
            out.append(tok)
    return " ".join(out)

def build_species_maps(color_pallete: Dict[str, str]) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,str]]:
    """
    Z palety {latin -> #HEX} vyrobí:
      code2latin: { 'cabe': 'carpinus betulus', ... }  (lowercase pro konzistenci dat)
      code2color: { 'cabe': '#E15759', ... }
      code2label: { 'cabe': 'Carpinus betulus', ... }  (pro legendy/štítky)
    """
    code2latin: Dict[str, str] = {}
    code2color: Dict[str, str] = {}
    code2label: Dict[str, str] = {}

    for latin_raw, hexcol in color_pallete.items():
        latin_stripped = latin_raw.strip()
        code = latin_to_code(latin_stripped)  # generuje z původního (ne lowernutého) řetězce
        if not code:
            continue
        if code not in code2latin:
            code2latin[code] = latin_stripped.lower()
            code2color[code] = hexcol
            code2label[code] = latin_binomial_label(latin_stripped)

    return code2latin, code2color, code2label