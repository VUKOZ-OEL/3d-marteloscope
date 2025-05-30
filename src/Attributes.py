import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from src.io import load_project_json

# Načti data do session_state
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
    st.session_state.trees = load_project_json(file_path)

if "universal_app_key" not in st.session_state:
    st.session_state.universal_app_key = "jnkqb34o78qy3rhef "

if "grid_counter" not in st.session_state:
    st.session_state.grid_counter = 0

grid_key = f"{st.session_state.universal_app_key}_grid_{st.session_state.grid_counter}"


# Přidej sloupec management_status, pokud chybí
if "management_status" not in st.session_state.trees.columns:
    st.session_state.trees["management_status"] = "none"

st.title("Tabulka stromů")

# Definuj sloupce
visible_columns = ["id", "label", "species", "management_status", "dbh", "height"]
hidden_columns = ["x", "y", "latlon", "status", "position", "dbhPosition", "lat", "lon"]

# Najdi neznámé sloupce a přidej je na konec
all_columns = list(st.session_state.trees.columns)
additional_columns = [col for col in all_columns if col not in visible_columns + hidden_columns]
display_columns = visible_columns + additional_columns

# Konfigurace AgGrid
gb = GridOptionsBuilder.from_dataframe(st.session_state.trees[display_columns])
gb.configure_selection("multiple", use_checkbox=False)
gb.configure_grid_options(domLayout="normal")
gb.configure_columns(["management_status"], editable=True)
gb.configure_side_bar(filters_panel=True, defaultToolPanel='filters')
grid_options = gb.build()

# Tlačítka
col1, col2, col3, col4 = st.columns(4)
with col1:
    target_tree_btn = st.button("Target tree")
with col2:
    remove_btn = st.button("Remove")
with col3:
    unselect_btn = st.button("Unselect")
with col4:
    export_btn = st.button("💾 Exportovat změny")

# Zobraz AgGrid - přímo session_state data
response = AgGrid(
    st.session_state.trees[display_columns],
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    theme="streamlit",
    enable_enterprise_modules=False,
    fit_columns_on_grid_load=True,
    reload_data=True,                                # ← přidáno
    key=grid_key + "_grid", # ← přidáno
)


selected_rows = response.get("selected_rows")

# pokud je tlačítko stisknuto a máme vybrané pozice, aktualizujeme status
if target_tree_btn and selected_rows is not None:
    
    raw_indices = selected_rows.iloc[:,0].tolist()
    col_idx = st.session_state.trees.columns.get_loc("management_status")
    st.session_state.trees.iloc[raw_indices, col_idx] = "Target tree"
    st.session_state.grid_counter += 1
    st.rerun()

if remove_btn and selected_rows is not None:
    
    raw_indices = selected_rows.iloc[:,0].tolist()
    col_idx = st.session_state.trees.columns.get_loc("management_status")
    st.session_state.trees.iloc[raw_indices, col_idx] = "Remove"
    st.session_state.grid_counter += 1
    st.rerun()

if unselect_btn and selected_rows is not None:

    raw_indices = selected_rows.iloc[:,0].tolist()
    col_idx = st.session_state.trees.columns.get_loc("management_status")
    st.session_state.trees.iloc[raw_indices, col_idx] = "Untouched"
    st.session_state.grid_counter += 1
    st.rerun()





# Export změněného DataFrame do JSON
if export_btn:
    export_path = "exported_trees.json"
    st.session_state.trees.to_json(export_path, orient="records", force_ascii=False, indent=2)
    st.success(f"Data byla exportována do souboru `{export_path}`")
