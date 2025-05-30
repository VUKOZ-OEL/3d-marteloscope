import streamlit as st
import pandas as pd
import json
from shapely.geometry import Point, Polygon
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw
from src.io import load_project_json

# --- Načtení a inicializace ---
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    )
    st.session_state.trees = load_project_json(file_path)

df = st.session_state.trees

# Přidáme sloupec management_status, pokud neexistuje
if "management_status" not in df.columns:
    df["management_status"] = [""] * len(df)

# Inicializace výběru
if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

# --- Výpočet středu a zoomu mapy (převráceně lat/lng) ---
min_x, max_x = df["x"].min(), df["x"].max()
min_y, max_y = df["y"].min(), df["y"].max()
center_lng = (min_x + max_x) / 2
center_lat = (min_y + max_y) / 2
max_range = max(max_x - min_x, max_y - min_y)
zoom_level = 15 if max_range < 50 else 13

# --- Rozložení: levý panel tlačítka, pravý panel mapa ---
col_buttons, col_map = st.columns([1, 5], gap="small")

with col_buttons:
    st.markdown("## Akce")
    if st.session_state.selected_points:
        if st.button("Target tree"):
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "target"
            st.experimental_rerun()
        if st.button("Remove"):
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "remove"
            st.experimental_rerun()
        if st.button("Unselect"):
            st.session_state.selected_points.clear()
            st.experimental_rerun()
    st.markdown("---")
    st.button("Exportovat změny")

with col_map:
    # Vytvoříme folium mapu
    m = folium.Map(location=[center_lat, center_lng],
                   zoom_start=zoom_level,
                   tiles="cartodbpositron")
    # Přidáme Draw plugin s podporou rectangle
    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={"edit": False}
    ).add_to(m)

    # Přidáme body jako CircleMarker
    for idx, row in df.iterrows():
        # barva podle statusu + zvýraznění vybraných
        if idx in st.session_state.selected_points:
            fill = "yellow"; color = "black"
        elif row["management_status"] == "target":
            fill = "green"; color = "green"
        elif row["management_status"] == "remove":
            fill = "red"; color = "red"
        else:
            fill = "gray"; color = "gray"
        folium.CircleMarker(
            location=[row["y"], row["x"]],
            radius=5,
            fill=True,
            fill_color=fill,
            color=color,
            fill_opacity=0.8
        ).add_to(m)

    # vykreslíme mapu a zachytíme kliknutí i kreslení
    map_data = st_folium(m, width="100%", height=800)

    # 1) Kliknutí na mapu (vyber nejbližší bod)
    if map_data["last_clicked"]:
        lat, lng = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        # spočítáme nejbližší strom
        d2 = (df["y"] - lat)**2 + (df["x"] - lng)**2
        idx = int(d2.idxmin())
        # přepneme výběr
        if idx in st.session_state.selected_points:
            st.session_state.selected_points.remove(idx)
        else:
            st.session_state.selected_points.add(idx)
        st.experimental_rerun()

    # 2) Kreslení obdélníku (Draw): polygony jsou v all_drawings
    drawings = map_data["all_drawings"] or []
    if drawings:
        # vezmeme poslední nakreslený polygon (rectangle)
        feature = drawings[-1]
        coords = feature["geometry"]["coordinates"][0]
        poly = Polygon(coords)
        # vyber body uvnitř
        sel = set()
        for idx, row in df.iterrows():
            if poly.contains(Point(row["x"], row["y"])):
                sel.add(idx)
        st.session_state.selected_points = sel
        st.experimental_rerun()
