import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, BoxSelectTool, TapTool, HoverTool, LabelSet,
    BoxZoomTool, ResetTool, WheelZoomTool, PanTool, CustomJS, Range1d
)
from streamlit_bokeh3_events import streamlit_bokeh3_events
import src.io as io
import src.utils as utils

# --- Konstanty ---
FILE_PATH = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"

# --- Inicializace session_state ---
if "trees" not in st.session_state:
    st.session_state.trees = io.load_project_json(FILE_PATH)
    st.session_state.colormap = io.load_colormap(FILE_PATH)

if "tree_updates" not in st.session_state:
    st.session_state.tree_updates = {}  # {index: management_status}

if "selected_points" not in st.session_state:
    st.session_state.selected_points = set()

# --- Příprava DataFrame ---
# Načteme původní JSON → DataFrame
df = utils.prepare_tree_dataframe(
    st.session_state.trees,
    st.session_state.colormap
)
# Aplikace uživatelských aktualizací
for idx, status in st.session_state.tree_updates.items():
    if idx in df.index:
        df.at[idx, "management_status"] = status

# Mapování barev (ověřujeme existenci klíčů)
species_map = st.session_state.colormap.get("species", {}) or {}
management_map = st.session_state.colormap.get("management", {}) or {}

df["color"] = df["species"].map(species_map).fillna("#aaaaaa")
df["line_color"] = df["management_status"].map(lambda s: management_map.get(s) if s in ["target", "remove"] else None)
df["line_width"] = df["management_status"].map(lambda s: 2 if s in ["target", "remove"] else 0)

# Přidání Bokeh atributů
df["id"] = df.index
df["alpha"] = 0.8
df["size"] = 8
df["label"] = df["tree_id"].astype(str) if "tree_id" in df.columns else df.index.astype(str)
source = ColumnDataSource(df)

# Rozsahy a zoom
min_x, max_x = df["x"].min(), df["x"].max()
min_y, max_y = df["y"].min(), df["y"].max()
default_x = (min_x - 10, max_x + 10)
default_y = (min_y - 10, max_y + 10)
view = st.session_state.get("view_range", {})
xr = view.get("x", default_x)
yr = view.get("y", default_y)

# --- Rozvržení stránky ---
col_buttons, col_map = st.columns([1, 5], gap="small")

with col_buttons:
    st.markdown("__Select management__")

    if st.button("Target tree"):
        for idx in st.session_state.selected_points:
            st.session_state.tree_updates[idx] = "target"
        st.rerun()

    if st.button("Remove"):
        for idx in st.session_state.selected_points:
            st.session_state.tree_updates[idx] = "remove"
        st.rerun()

    if st.button("Unselect"):
        st.session_state.selected_points.clear()
        st.rerun()

    st.markdown("---")
    if st.button("Exportovat změny"):
        # Uložení upravených dat
        updated = utils.update_trees(
            st.session_state.trees,
            st.session_state.tree_updates
        )
        io.save_project_json(FILE_PATH, updated)
        st.success("Změny byly úspěšně exportovány.")

with col_map:
    p = figure(
        title="Pick trees on map",
        x_range=Range1d(*xr),
        y_range=Range1d(*yr),
        sizing_mode="stretch_width",
        height=800,
        tools="",
        toolbar_location="above"
    )
    # Přidání nástrojů
    p.add_tools(
        BoxSelectTool(), TapTool(), HoverTool(tooltips=[("Tree", "@label")]),
        BoxZoomTool(), ResetTool(), WheelZoomTool(), PanTool()
    )

    # Kružnice
    p.circle(
        x='x', y='y', size='size',
        fill_color='color', fill_alpha='alpha',
        line_color='line_color', line_width='line_width',
        source=source, name="tree_points"
    )

    # Popisky
    labels = LabelSet(
        x='x', y='y', text='label',
        x_offset=5, y_offset=5,
        source=source, text_font_size="10pt"
    )
    p.add_layout(labels)

    # JS callback pro RANGE_EVENT (zahrnuje obě osy)
    range_cb = CustomJS(
        args=dict(x_range=p.x_range, y_range=p.y_range),
        code="""
            document.dispatchEvent(new CustomEvent("RANGE_EVENT", {
                detail: {
                    x_range: [x_range.start, x_range.end],
                    y_range: [y_range.start, y_range.end]
                }
            }));
        """
    )
    p.x_range.js_on_change("start", range_cb)
    p.x_range.js_on_change("end", range_cb)
    p.y_range.js_on_change("start", range_cb)
    p.y_range.js_on_change("end", range_cb)

    # Zpracování událostí
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
                "x": r.get("x_range", xr),
                "y": r.get("y_range", yr)
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
            st.rerun()
