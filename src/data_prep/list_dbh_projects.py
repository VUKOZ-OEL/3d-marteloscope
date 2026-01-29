import os
import pandas as pd
import streamlit as st
import src.io_utils as iou


def show_dbh_table_from_projects(project_json_paths: list[str], *, label_col: str = "label", dbh_col: str = "dbh"):
    """
    Načte DBH z více JSON projektů a zobrazí je v tabulce.
    - sloupce: jméno projektu (název souboru bez přípony)
    - řádky: label stromu
    - hodnoty: dbh

    Parametry:
      project_json_paths: list cest k .json projektům
      label_col: název sloupce s labelem stromu v načtených datech (default "label")
      dbh_col: název sloupce s DBH v načtených datech (default "dbh")
    """
    series_list = []

    for path in project_json_paths:
        trees = iou.load_project_json(path)  # očekávám DataFrame

        if trees is None or len(trees) == 0:
            continue

        if label_col not in trees.columns:
            raise KeyError(f"V projektu '{path}' chybí sloupec '{label_col}'. Dostupné: {list(trees.columns)}")
        if dbh_col not in trees.columns:
            raise KeyError(f"V projektu '{path}' chybí sloupec '{dbh_col}'. Dostupné: {list(trees.columns)}")

        project_name = os.path.splitext(os.path.basename(path))[0]

        s = (
            trees[[label_col, dbh_col]]
            .copy()
            .dropna(subset=[label_col])
            .groupby(label_col, as_index=True)[dbh_col]
            .first()
        )
        s.name = project_name
        series_list.append(s)

    if not series_list:
        st.warning("Nenačetl jsem žádná data (zkontroluj cesty a obsah JSON).")
        return

    out = pd.concat(series_list, axis=1).sort_index()
    out.index.name = "Tree label"

    st.dataframe(out, use_container_width=True)
    return out


project_json_paths = [
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\_PokojnaHora_mod.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\pokojna_test.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\pokojna_test_v2.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\pokojna_test_v3.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\pokojna_test_v4.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\PokojnaHora.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\PokojnaHora_tests.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\PokojnaHora_tests_merged.json",
    r"c:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\SLP_Pokojna\PokojnaHora_3df\SLP_Pokojna_hora.json",
]
show_dbh_table_from_projects(project_json_paths)
