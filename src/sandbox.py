import streamlit as st
import pandas as pd
import json
from streamlit.components.v1 import html

# Nacteni dat (zachovani kompatibility s puvodnim kodem)
from src.io import load_project_json

#st.set_page_config(layout="wide")

# Inicalizace session state
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test.json"
    st.session_state.trees = load_project_json(file_path)

if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

if "rectangle_polygons" not in st.session_state:
    st.session_state.rectangle_polygons = []

# Pracovni DataFrame
df = st.session_state.trees
if "color" not in df.columns:
    df["color"] = ["red"] * len(df)

# JSON data pro deck.gl
geojson_features = []
for idx, row in df.iterrows():
    geojson_features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row["lon"], row["lat"]],
        },
        "properties": {
            "id": row["id"],
            "index": idx,
            "color": row["color"],
        },
    })

geojson = {
    "type": "FeatureCollection",
    "features": geojson_features,
}

# Zasle data jako JSON do deck.gl komponenty
html(
    f"""
    <div id="deck-map" style="width:100%; height:100vh;"></div>
    <script src="https://unpkg.com/deck.gl@8.7.0/dist.min.js"></script>
    <script>
    const geojson = {json.dumps(geojson)};
    
    const deckgl = new deck.DeckGL({{
        container: 'deck-map',
        map: false,
        views: [new deck.OrthographicView()],
        initialViewState: {{
            target: [0, 0, 0],
            zoom: 0,
        }},
        controller: true,
        layers: [
            new deck.GeoJsonLayer({{
                id: 'geojson-layer',
                data: geojson,
                pickable: true,
                filled: true,
                pointRadiusMinPixels: 5,
                getFillColor: f => f.properties.color === "green" ? [0,255,0] : [255,0,0],
                onClick: info => {{
                    if (info.object) {{
                        const idx = info.object.properties.index;
                        window.parent.postMessage(JSON.stringify({{"type": "select", "index": idx}}), '*');
                    }}
                }}
            }})
        ]
    }});
    </script>
    """,
    height=700
)


# Zachytit JS event z onClick a aktualizovat Session
selected = st.experimental_get_query_params().get("selected")
if selected:
    try:
        idx = int(selected[0])
        if idx in st.session_state.selected_points:
            st.session_state.selected_points.remove(idx)
            df.at[idx, "color"] = "red"
        else:
            st.session_state.selected_points.add(idx)
            df.at[idx, "color"] = "green"
    except:
        pass

# Export zmenenych dat
st.download_button("📥 Export JSON", data=df.to_json(orient="records"), file_name="modified_points.json")
