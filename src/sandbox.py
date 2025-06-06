import streamlit as st
import pandas as pd
import json
from shapely.geometry import Point, Polygon
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, BoxSelectTool, TapTool, HoverTool, LabelSet
from bokeh.models.tools import BoxZoomTool, ResetTool, WheelZoomTool, PanTool
from streamlit_bokeh3_events import streamlit_bokeh3_events
import src.io as io

# --- Načtení a inicializace ---
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    )
    st.session_state.trees = io.load_project_json(file_path)

df = st.session_state.trees

if "colormap" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    )
    st.session_state.colormap = io.load_colormap(file_path)

# Inicializace výběru
if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

# --- Výpočet rozsahu ---
min_x, max_x = df["x"].min(), df["x"].max()
min_y, max_y = df["y"].min(), df["y"].max()
x_range = (min_x - 10, max_x + 10)
y_range = (min_y - 10, max_y + 10)

# --- Převod df do Bokeh ColumnDataSource ---
df["id"] = df.index
df["color"] = df["management_status"].map({
    "target": "green",
    "remove": "red"
}).fillna("gray")
df["alpha"] = 0.8
df["size"] = 8
df["label"] = df["tree_id"].astype(str) if "tree_id" in df.columns else df.index.astype(str)
source = ColumnDataSource(df)

# --- Rozložení layoutu ---
col_buttons, col_map = st.columns([1, 5], gap="small")

with col_buttons:
    st.markdown("__Select management__")

    if st.button("Target tree"):
        if st.session_state.selected_points:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "target"
            st.experimental_rerun()

    if st.button("Remove"):
        if st.session_state.selected_points:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "remove"
            st.experimental_rerun()

    if st.button("Unselect"):
        st.session_state.selected_points.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.button("Exportovat změny")

with col_map:
    # --- Vytvoření Bokeh figury ---
    p = figure(
        title="Pick trees on map",
        x_range=x_range,
        y_range=y_range,
        #width=1000,
        sizing_mode="stretch_width",
        height=800,
        tools="",
        toolbar_location="above"
    )
    p.add_tools(BoxSelectTool(), TapTool(), HoverTool(tooltips=[("Tree", "@label")]), BoxZoomTool(), ResetTool(), WheelZoomTool(), PanTool())

    # --- Body stromů ---
    renderer = p.circle(
        x='x',
        y='y',
        size='size',
        fill_color='color',
        fill_alpha='alpha',
        line_color='black',
        source=source,
        name="tree_points"
    )

    # --- Labely stromů ---
    labels = LabelSet(
        x='x',
        y='y',
        text='label',
        x_offset=5,
        y_offset=5,
        source=source,
        text_font_size="8pt"
    )
    p.add_layout(labels)

    # --- Event catcher ---
    result = streamlit_bokeh3_events(
        events="tap",
        bokeh_plot=p,
        key="map_events",
        debounce_time=0,
        override_height=600  # nižší, aby nepřetékal
        #sizing_mode="stretch_both"  #  reaguje na velikost
    )

    # --- Zpracování kliknutí ---
    if result and "tap" in result:
        tap_x = result["tap"]["x"]
        tap_y = result["tap"]["y"]
        d2 = (df["x"] - tap_x) ** 2 + (df["y"] - tap_y) ** 2
        idx = int(d2.idxmin())
        if idx in st.session_state.selected_points:
            st.session_state.selected_points.remove(idx)
        else:
            st.session_state.selected_points.add(idx)
       
