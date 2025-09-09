# streamlit_forest_page.py
# -------------------------------------------------------------
# Streamlit stránka pro vizualizaci simulací růstu lesa (iLand)
# - Načítá všechny SQLite databáze ve zvolené složce (replikace)
# - Využívá read_outputs_tree_level() a prepare_dataset() z přiložených modulů
# - Používá barevnou paletu/species názvy konzistentní s Dashboardem (kódy + turbo paleta)
# - Nahoře je slider pro rok simulace (1–50)
# - Zobrazuje grafy z outputs_graphs + navíc prostorovou mapu jedinců pro vybraný rok
# -------------------------------------------------------------
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Lokální moduly (musí být ve stejné složce nebo v PYTHONPATH)
from simul.outputs_graphs import (
    read_outputs_tree_level, combine_with_ids, 
    trees_by_species_presence, plot_trees_by_species_presence,
    deaths_by_year_species, plot_stacked_deaths_by_year_species,
    trees_by_species_volume, plot_stacked_volume_by_species_over_years,
    PALETTE, SPECIES_CODES
)
from simul.output_forest_development import prepare_dataset, species_colors

st.set_page_config(page_title="Forest Growth (iLand) – Tree-level", layout="wide")
st.title("Forest Growth – Tree-level dashboard")

# -------------------------------
# SIDEBAR – vstupy & konfigurace
# -------------------------------
st.sidebar.header("Data")
root_folder = st.sidebar.text_input(
    "Složka s replikacemi (SQLite soubory)",
    value="database_examples_3",
    help="Uveďte cestu ke složce, která obsahuje SQLite soubory s tabulkami 'tree' a 'treeremoved'."
)

# Volitelný JSON s mapou názvů dřevin (kód -> plný název)
species_json_path = st.sidebar.text_input(
    "Volitelně: JSON se slovníkem názvů dřevin (např. {'piab':'Smrk ztepilý', ...})",
    value="",
    help="Pokud necháte prázdné, zobrazí se kódy dřevin."
)

# Načtení dat
data_load_ok = False
outputs = None
if root_folder and Path(root_folder).exists():
    try:
        outputs = read_outputs_tree_level(root_folder)
        data_load_ok = True
    except Exception as e:
        st.error(f"Chyba při načítání dat ze složky '{root_folder}': {e}")
else:
    st.info("Zadejte existující složku s replikacemi (SQLite).")

# -------------------------------
# MAPA NÁZVŮ & BAREV DŘEVIN
# -------------------------------
def rgba_to_hex(rgba):
    r, g, b, a = rgba
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

species_hex = {sp: rgba_to_hex(PALETTE.get(sp, (0.5,0.5,0.5,1.0))) for sp in SPECIES_CODES}

# species labels – načti z JSON, pokud je k dispozici
species_labels: Dict[str, str] = {sp: sp for sp in SPECIES_CODES}
if species_json_path and Path(species_json_path).exists():
    try:
        with open(species_json_path, "r", encoding="utf-8") as f:
            user_labels = json.load(f)
        # pouze doplníme známé klíče
        for k, v in user_labels.items():
            species_labels[k] = str(v)
    except Exception as e:
        st.warning(f"JSON s názvy dřevin se nepodařilo načíst: {e}")

# Globální slider (1–50) pro výběr roku – nahoře na stránce
year = st.slider("Rok simulace (1–50)", min_value=1, max_value=50, value=1, step=1)

# -------------------------------
# Výpočty a grafy
# -------------------------------
if data_load_ok and outputs is not None:
    # Kombinované rámce
    df_living = combine_with_ids(outputs.living)
    df_death  = combine_with_ids(outputs.death)

    # Připrav dataset na úrovni jedinců (viz příklad v modulech)
    df_tree = prepare_dataset(outputs, mark_death=True)
    # Ujisti se o datových typech
    for col in ["year","id","x","y","dbh"]:
        if col in df_tree.columns:
            df_tree[col] = pd.to_numeric(df_tree[col], errors="coerce")

    # 1) Grafy z outputs_graphs (stacked bary) – v samostatných záložkách
    tabs = st.tabs(["Trees by species (presence)", "Deaths by year × species", "Living volume by species over years", "Prostorová mapa (extra)"])

    # --- Tab 1: Trees by species presence ---
    with tabs[0]:
        left, mid = st.columns([3, 7])
        with left:
            scale1 = st.radio("Škála", ["absolute", "relative"], index=0, horizontal=True, key="scale_presence")
        wide1 = trees_by_species_presence(df_living, scale=scale1) if not df_living.empty else pd.DataFrame()
        if wide1.empty:
            st.warning("Chybí data v tabulce 'tree'.")
        else:
            # Překlad kódů v legendě (volitelně)
            wide1 = wide1.rename(columns=species_labels)
            fig1, ax1 = plot_trees_by_species_presence(wide1, scale=scale1, title="Trees by species presence")
            st.pyplot(fig1, use_container_width=True)

    # --- Tab 2: Deaths by year × species ---
    with tabs[1]:
        left, mid = st.columns([3, 7])
        with left:
            scale2 = st.radio("Škála", ["absolute", "relative", "relative_toForest"], index=0, horizontal=True, key="scale_deaths")
        wide2 = deaths_by_year_species(df_death, df_living=df_living, scale=scale2) if not df_death.empty else pd.DataFrame()
        if wide2.empty:
            st.info("Chybí data v tabulce 'treeremoved'.")
        else:
            wide2 = wide2.rename(columns=species_labels)
            fig2, ax2 = plot_stacked_deaths_by_year_species(wide2, scale=scale2, title="Deaths by year × species")
            st.pyplot(fig2, use_container_width=True)

    # --- Tab 3: Living volume by species over years ---
    with tabs[2]:
        left, mid = st.columns([3, 7])
        with left:
            scale3 = st.radio("Škála", ["absolute", "relative"], index=0, horizontal=True, key="scale_volume")
        wide3 = trees_by_species_volume(df_living, scale=scale3) if not df_living.empty else pd.DataFrame()
        if wide3.empty:
            st.info("Chybí sloupec 'volume_m3' v 'tree' nebo prázdná data.")
        else:
            wide3 = wide3.rename(columns=species_labels)
            fig3, ax3 = plot_stacked_volume_by_species_over_years(wide3, scale=scale3, title="Living volume by species over years")
            st.pyplot(fig3, use_container_width=True)

    # --- Tab 4 (EXTRA): Prostorová mapa jedinců pro vybraný rok ---
    with tabs[3]:
        st.markdown("#### Prostorová mapa jedinců (x–y), velikost ~ DBH, barva = druh")
        # filtr na vybraný rok
        df_year = df_tree[(df_tree["year"] == year)].copy()
        if df_year.empty:
            st.info("Pro zvolený rok nejsou dostupná data.")
        else:
            # přemapovat názvy a barvy
            df_year["species_label"] = df_year["species"].map(species_labels).fillna(df_year["species"])
            df_year["color_hex"] = df_year["species"].map(species_hex).fillna("#999999")
            # Plotly scatter
            fig_sc = px.scatter(
                df_year, x="x", y="y",
                color="species_label",
                size="dbh",
                size_max=16,
                hover_data={"id": True, "dbh": True, "x": True, "y": True, "species_label": True},
                title=f"Mapa stromů – rok {int(year)}",
            )
            # aplikace vlastních barev
            # (Plotly potřebuje dict 'category_orders' + 'color_discrete_map')
            species_order = sorted(df_year["species_label"].unique().tolist())
            color_map = {lbl: df_year.loc[df_year["species_label"]==lbl, "color_hex"].iloc[0] for lbl in species_order}
            fig_sc.update_traces(marker=dict(opacity=0.9, line=dict(width=0)))
            fig_sc.update_layout(
                legend_title_text="Dřevina",
                xaxis_title="x [m]", yaxis_title="y [m]",
                height=700,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=60, b=10)
            )
            fig_sc.update_xaxes(constrain="domain", scaleanchor="y", scaleratio=1)
            st.plotly_chart(fig_sc, use_container_width=True)

            # Doplňkový přehled: rozdělení DBH podle druhu v daném roce
            with st.expander("DBH distribuce podle druhu (vybraný rok)"):
                bins = st.slider("Šířka binu [cm]", 2, 20, 4, step=1, key="dbh_bin_width")
                # připrav histogramy pro každý druh v jednom figure (facety by species)
                df_year["dbh_bin"] = (np.floor(df_year["dbh"]/bins)*bins).astype(int)
                hist = (df_year.groupby(["species_label","dbh_bin"])["id"]
                                .size().rename("count").reset_index())
                species_in_year = sorted(hist["species_label"].unique().tolist())
                fig_h = go.Figure()
                for sp in species_in_year:
                    hh = hist[hist["species_label"]==sp]
                    fig_h.add_bar(x=hh["dbh_bin"].astype(str), y=hh["count"], name=sp)
                fig_h.update_layout(barmode="stack", height=420, margin=dict(l=10,r=10,t=40,b=10))
                fig_h.update_xaxes(title_text="DBH třídy [cm]")
                fig_h.update_yaxes(title_text="Počet stromů")
                st.plotly_chart(fig_h, use_container_width=True)
else:
    st.stop()
