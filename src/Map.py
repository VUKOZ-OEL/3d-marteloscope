import streamlit as st
from src.io import *
import pandas as pd
import folium
from streamlit_folium import st_folium

st.markdown(
    """
    <style>
        .main {
            padding: 10px;
        }
        .block-container {
            padding: 30px;
        }
        .map-container {
            height: 50%;  /* Dynamická výška mapy */
            width: 50% !important;  /* Celá dostupná šířka */
        }
    </style>
    """,
    unsafe_allow_html=True,
)


if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test.json"
    st.session_state.trees = load_project_json(file_path)


if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()  # Ukládá indexy vybraných bodů

df = st.session_state.trees

# Inicializace sloupce 'color' (každý bod má barvu)
if "color" not in df.columns:
    df["color"] = ["red" for _ in range(len(df))]  # Výchozí barva červená

# Callback pro výběr bodu
def select_point(index):
    if index in st.session_state.selected_points:
        st.session_state.selected_points.remove(index)
        df.at[index, "color"] = "red"  # Vrátit na červenou
    else:
        st.session_state.selected_points.add(index)
        df.at[index, "color"] = "green"  # Označit zeleně
    st.session_state.trees = df  # Uložit změny v session_state


# Rozdělení stránky do dvou sloupců
col1, col2 = st.columns([1, 4], gap="small")  # Levý sloupec 1 díl, pravý 7 díly

# Levý sloupec – ovládací prvky
with col1:
    st.header("Selection")

    if st.button("Target tree"):
        select_point()
        st.rerun()

    if st.button("Remove tree"):
        #select_point(selected_index)
        st.rerun()

    if st.button("Unselect tree"):
        #select_point(selected_index)
        st.rerun()

# Pravý sloupec – čtvercová mapa přes celou výšku
with col2:
    st.markdown(
        """
        <style>
        .map-container {
            height: 100vh !important;  /* Nastavení výšky mapy na celou obrazovku */
            width: 100%;  /* Celá šířka sloupce */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Inicializace mapy **bez podkladu**
    m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6, tiles=None)

    # Přidání bodů do mapy
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color=row["color"],  
            fill=True,
            fill_color=row["color"],
            fill_opacity=1,
            popup=f"Tree ID: {row['id']}",
        ).add_to(m)

    # Vykreslení mapy v kontejneru s celou výškou
    st_folium(m, use_container_width=True)

    #st.markdown('</div>', unsafe_allow_html=True)