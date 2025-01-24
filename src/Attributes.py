import streamlit as st
import pandas as pd
import json
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from src.io import *


if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat.json"
    st.session_state.trees = load_project_json(file_path)


col1, col2, col3 = st.columns(3)
with col1:
    target_tree_btn = st.button("Target tree")
with col2:
    remove_btn = st.button("Remove")
with col3:
    unselect_btn = st.button("Unselect")

# Konfigurace tabulky
gb = GridOptionsBuilder.from_dataframe(st.session_state.trees)
gb.configure_selection("multiple", use_checkbox=True)  # Povolit výběr více řádků
gb.configure_grid_options(domLayout="normal")  # Standardní rozložení
gb.configure_columns(["status"], editable=True)
gb.configure_side_bar(filters_panel=True, defaultToolPanel='filters')
grid_options = gb.build()

# Zobrazení interaktivní tabulky
response = AgGrid(
    st.session_state.trees,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    theme="streamlit",
    enable_enterprise_modules=False,
    fit_columns_on_grid_load=True,
)

# Získání vybraných řádků
selected_rows = response.get("selected_rows",[])

#st.text(type(selected_rows))

# Zpracování tlačítek
if selected_rows is not None and not selected_rows.empty:  # Zkontrolujeme, zda DataFrame není prázdný
    selected_ids = selected_rows["id"].tolist()  # Extrahujeme sloupec "id" jako seznam

    if target_tree_btn:
        st.session_state.trees.loc[
            st.session_state.trees["id"].isin(selected_ids), "status"
        ] = "Target tree"
        st.success(f"Status byl aktualizován na 'Target tree' pro ID: {selected_ids}")
    elif remove_btn:
        st.session_state.trees.loc[
            st.session_state.trees["id"].isin(selected_ids), "status"
        ] = "Remove"
        st.success(f"Status byl aktualizován na 'Remove' pro ID: {selected_ids}")
    elif unselect_btn:
        st.session_state.trees.loc[
            st.session_state.trees["id"].isin(selected_ids), "status"
        ] = "Unselect"
        st.success(f"Status byl aktualizován na 'Unselect' pro ID: {selected_ids}")

response.set