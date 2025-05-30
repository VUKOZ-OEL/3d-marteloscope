import streamlit as st
import pandas as pd
import json
from streamlit.components.v1 import html
from src.io import load_project_json

# --- Načtení a inicializace ---
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    st.session_state.trees = load_project_json(file_path)

df = st.session_state.trees

# Přidáme sloupec management_status, pokud neexistuje
if "management_status" not in df.columns:
    df["management_status"] = [""] * len(df)

# Inicializace výběru
if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

# --- Výpočet středu a zoomu mapy ---
min_x, max_x = df["x"].min(), df["x"].max()
min_y, max_y = df["y"].min(), df["y"].max()
center_x = (min_x + max_x) / 2
center_y = (min_y + max_y) / 2
max_range = max(max_x - min_x, max_y - min_y)
zoom_level = 8 if max_range < 10 else (5 if max_range < 50 else 3)

# --- Rozložení: levý panel na ovládací tlačítka, pravý panel na mapu ---
col_buttons, col_map = st.columns([1, 5], gap="small")

with col_buttons:
    st.markdown("## Akce")
    target_tree_btn = st.button("Target tree")
    remove_btn      = st.button("Remove")
    unselect_btn    = st.button("Unselect")
    st.markdown("---")
    export_btn      = st.button("Exportovat změny")

    if st.session_state.selected_points:
        if target_tree_btn:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "target"
            st.session_state.selected_points.clear()
            st.experimental_rerun()

        if remove_btn:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "remove"
            st.session_state.selected_points.clear()
            st.experimental_rerun()

        if unselect_btn:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = ""
            st.session_state.selected_points.clear()
            st.experimental_rerun()

with col_map:
    # Připravíme GeoJSON
    features = []
    for idx, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["x"], row["y"]]},
            "properties": {
                "index": idx,
                "management_status": row["management_status"]
            }
        })
    geojson = {"type": "FeatureCollection", "features": features}

    # Iframe má 800px výšku, vnitřní div stejně 800px
    html(
        f"""
        <div id="deck-map" style="width:100%; height:800px; max-width:100vw;"></div>
        <script src="https://unpkg.com/deck.gl@8.7.0/dist.min.js"></script>
        <script>
        const geojson = {json.dumps(geojson)};
        window.addEventListener("message", event => {{
            try {{
                const data = JSON.parse(event.data);
                if (data.type === "select") {{
                    const url = new URL(window.location);
                    url.searchParams.set("selected", data.index);
                    window.location.href = url.toString();
                }}
            }} catch (e) {{ console.error(e); }}
        }});

        new deck.DeckGL({{
            container: 'deck-map',
            views: [new deck.OrthographicView()],
            initialViewState: {{ target: [{center_x}, {center_y}, 0], zoom: {zoom_level} }},
            controller: true,
            layers: [ new deck.GeoJsonLayer({{
                id: 'geojson-layer',
                data: geojson,
                pickable: true,
                filled: true,
                stroked: false,
                getRadius: f => 1,
                pointRadiusScale: 1,
                getFillColor: f => {{
                    switch(f.properties.management_status) {{
                        case "target": return [0, 128, 0];
                        case "remove": return [255, 0, 0];
                        default:       return [200, 200, 200];
                    }}
                }},
                onClick: info => {{
                    if (info.object) {{
                        window.parent.postMessage(
                            JSON.stringify({{type:'select', index: info.object.properties.index}}),
                            '*'
                        );
                    }}
                }}
            }}) ]
        }});
        </script>
        """,
        height=800,
    )

# --- Zpracování výběru bodu ---
params = st.experimental_get_query_params()
if "selected" in params:
    try:
        idx = int(params["selected"][0])
        if idx in st.session_state.selected_points:
            st.session_state.selected_points.remove(idx)
        else:
            st.session_state.selected_points.add(idx)
    except ValueError:
        pass
    st.experimental_set_query_params()
    st.experimental_rerun()
