import streamlit as st
import pandas as pd
from shapely.geometry import Point, Polygon
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, BoxSelectTool, TapTool, HoverTool, LabelSet,
    BoxZoomTool, ResetTool, WheelZoomTool, PanTool, CustomJS, Range1d
)
from streamlit_bokeh3_events import streamlit_bokeh3_events
import src.io as io
import src.utils as utils

# --- Načtení dat ---
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    st.session_state.trees = io.load_project_json(file_path)
    st.session_state.colormap = io.load_colormap(file_path)

df = utils.prepare_tree_dataframe(st.session_state.trees, st.session_state.colormap)
#species_colormap = st.session_state.colormap.get("species", {})
#management_colormap = st.session_state.colormap.get("management", {})

# Inicializace výběru
if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

# Výpočet rozsahu nebo obnova z uloženého zoomu
min_x, max_x = df["x"].min(), df["x"].max()
min_y, max_y = df["y"].min(), df["y"].max()
default_x_range = (min_x - 10, max_x + 10)
default_y_range = (min_y - 10, max_y + 10)

x_range = st.session_state.get("view_range", {}).get("x", default_x_range)
y_range = st.session_state.get("view_range", {}).get("y", default_y_range)

# --- Vykreslovací atributy ---
df["color"] = df["species"].map(species_colormap).fillna("#aaaaaa")
df["line_color"] = df["management_status"].apply(
    lambda status: management_colormap.get(status) if status in ["target", "remove"] else None
)
df["line_width"] = df["management_status"].apply(
    lambda status: 2 if status in ["target", "remove"] else 0
)

# Převod do ColumnDataSource
df["id"] = df.index
df["alpha"] = 0.8
df["size"] = 8
df["label"] = df["tree_id"].astype(str) if "tree_id" in df.columns else df.index.astype(str)
source = ColumnDataSource(df)

# --- Rozvržení ---
col_buttons, col_map = st.columns([1, 5], gap="small")

with col_buttons:
    st.markdown("__Select management__")

    if st.button("Target tree"):
        if st.session_state.selected_points:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "target"
            st.session_state.trees = df  # uložit změny!
            st.experimental_rerun()

    if st.button("Remove"):
        if st.session_state.selected_points:
            for idx in st.session_state.selected_points:
                df.at[idx, "management_status"] = "remove"
            st.session_state.trees = df  # uložit změny!
            st.experimental_rerun()

    if st.button("Unselect"):
        st.session_state.selected_points.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.button("Exportovat změny")

with col_map:
    p = figure(
        title="Pick trees on map",
        x_range=Range1d(*x_range),
        y_range=Range1d(*y_range),
        sizing_mode="stretch_width",
        height=800,
        tools="",
        toolbar_location="above"
    )
    p.add_tools(BoxSelectTool(), TapTool(), HoverTool(tooltips=[("Tree", "@label")]),
                BoxZoomTool(), ResetTool(), WheelZoomTool(), PanTool())

    p.circle(
        x='x', y='y', size='size',
        fill_color='color', fill_alpha='alpha',
        line_color='line_color', line_width='line_width',
        source=source, name="tree_points"
    )

    labels = LabelSet(
        x='x', y='y', text='label',
        x_offset=5, y_offset=5,
        source=source, text_font_size="8pt"
    )
    p.add_layout(labels)

    # JS callback pro rozsah
    range_callback = CustomJS(code="""
        document.dispatchEvent(new CustomEvent("RANGE_EVENT", {
            detail: {
                x_range: [cb_obj.start, cb_obj.end],
                y_range: [cb_obj.start, cb_obj.end]
            }
        }))
    """)
    p.x_range.js_on_change("start", range_callback)
    p.y_range.js_on_change("start", range_callback)

    result = streamlit_bokeh3_events(
        events="tap,RANGE_EVENT",
        bokeh_plot=p,
        key="map_events",
        debounce_time=0,
        override_height=600
    )

    if result:
        if "RANGE_EVENT" in result:
            r = result["RANGE_EVENT"]
            st.session_state["view_range"] = {
                "x": r.get("x_range", x_range),
                "y": r.get("y_range", y_range)
            }

        if "tap" in result:
            tap_x = result["tap"]["x"]
            tap_y = result["tap"]["y"]
            d2 = (df["x"] - tap_x)**2 + (df["y"] - tap_y)**2
            idx = int(d2.idxmin())
            if idx in st.session_state.selected_points:
                st.session_state.selected_points.remove(idx)
            else:
                st.session_state.selected_points.add(idx)
            st.experimental_rerun()
