import streamlit as st
from src.io import *
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
import pandas as pd

# Inicializace dat
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat.json"
    st.session_state.trees = load_project_json(file_path)

# Mapování barev
color_mapping_status = {
    "Target tree": "red",
    "Remove": "green",
    "Unselect": "blue",
    "Valid": "gray",
}
color_mapping_species = {
    "Species A": "orange",
    "Species B": "purple",
    "Species C": "cyan",
}

# Přepínač pro zobrazení barev
color_by = st.radio("Color by:", ["status", "species"], index=0, horizontal = True)

# Přidání barev na základě výběru
if color_by == "status":
    st.session_state.trees["color"] = st.session_state.trees["status"].map(color_mapping_status)
else:
    if "species" not in st.session_state.trees.columns:
        st.session_state.trees["species"] = pd.Series(
            ["Species A", "Species B", "Species C"]
        ).sample(len(st.session_state.trees), replace=True).tolist()
    st.session_state.trees["color"] = st.session_state.trees["species"].map(color_mapping_species)

# Zdroj dat pro Bokeh
source = ColumnDataSource(st.session_state.trees)

# Vytvoření interaktivního grafu
p = figure(
    title="Interaktivní graf stromů",
    tools="tap,pan,wheel_zoom,box_zoom,reset",
    width=700,
    height=500,
    x_axis_label="X (lokální souřadnice)",
    y_axis_label="Y (lokální souřadnice)",
)

# Přidání bodů
p.circle(
    x="x",
    y="y",
    size=10,
    source=source,
    color="color",
    line_color=None,
    legend_field=color_by,
)

# Přidání hover tool
p.add_tools(HoverTool(tooltips=[("ID", "@id"), ("Status", "@status"), ("Species", "@species"), ("DBH", "@dbh"), ("Height", "@height")]))

# Interaktivní klikání na body
tap_tool = TapTool()
p.add_tools(tap_tool)

# Zobrazení grafu ve Streamlit
st.bokeh_chart(p, use_container_width=True)

# Výběr bodu kliknutím
if "selected" not in st.session_state:
    st.session_state.selected = None

selected_indices = source.selected.indices
if selected_indices:
    selected_id = st.session_state.trees.iloc[selected_indices[0]]["id"]
    st.session_state.selected = selected_id
    st.write(f"Vybraný strom ID: {selected_id}")

    # Možnost změnit status vybraného bodu
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Target tree"):
            st.session_state.trees.loc[
                st.session_state.trees["id"] == selected_id, "status"
            ] = "Target tree"
    with col2:
        if st.button("Remove"):
            st.session_state.trees.loc[
                st.session_state.trees["id"] == selected_id, "status"
            ] = "Remove"
    with col3:
        if st.button("Unselect"):
            st.session_state.trees.loc[
                st.session_state.trees["id"] == selected_id, "status"
            ] = "Unselect"

    # Aktualizace barev bodů
    if st.session_state.trees["status"].map(color_mapping_status).notnull().all():
        source.data = st.session_state.trees

# Uložení aktuálního výřezu jako navštívené oblasti
if st.button("Uložit aktuální výřez"):
    x_range = p.x_range.start, p.x_range.end
    y_range = p.y_range.start, p.y_range.end
    st.session_state.visited_areas.append({"x_min": x_range[0], "x_max": x_range[1], "y_min": y_range[0], "y_max": y_range[1]})
    st.success("Výřez byl uložen jako navštívená oblast.")
