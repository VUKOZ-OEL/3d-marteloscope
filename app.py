import streamlit as st
import src.io_utils as iou


# Set Page Title
st.set_page_config(page_title="3D-Marteloscope", page_icon=":material/nature_people:",layout="wide")

# Dash page with summary
dash_page = st.Page("src/Dashboard.py", title="Plot Info", icon=":material/dashboard:")

# Tree & Crown stats
tree_page = st.Page("src/Tree_stats.py", title="Tree Statistics", icon=":material/nature:")
canopy_page = st.Page("src/Canopy_stats.py", title="Canopy Volume", icon=":material/forest:")
space_page = st.Page("src/Space_comp.py", title="Space Competition", icon=":material/workspaces:")
light_page = st.Page("src/Light_comp.py", title="Light Competition", icon=":material/light_mode:")
heatmap_page = st.Page("src/Heatmaps.py", title="Heatmaps", icon=":material/blur_on:")

# Map, Att tab, pygpage
map_page = st.Page("src/Map.py", title="Plot Map", icon=":material/map:")
att_page = st.Page("src/Attributes.py", title="Attribute Table", icon=":material/data_table:")
analytics_page = st.Page("src/Analytics.py", title="Analytics - Experimental", icon=":material/addchart:")

# Simulation results 
simul_page = st.Page("src/Simulation.py", title="Simulation", icon=":material/clock_arrow_up:")
simul_detail_page = st.Page("src/Simulation_detail.py", title="Deatiled view", icon=":material/frame_inspect:")
test_page = st.Page("src/sandbox.py", title="COMMENTS")



file_path = "data/test_project.json"
st.session_state.trees = iou.load_project_json(file_path)
st.session_state.plot_info = iou.load_plot_info(file_path)


# Define common labels:
st.session_state.Before = "Original Stand"
st.session_state.After = "Managed Stand"
st.session_state.Removed = "Removed from Stand"


st.session_state.plot_title_font = dict(size=18, color="#33343f", weight="bold")

pages = {
    "Summary": [
        dash_page,
    ],
    "Results:": [
        tree_page,
        canopy_page,
        space_page,
        light_page,
        heatmap_page,
        map_page,
        att_page,
        #analytics_page,
    ],
    "Future Outlook": [
        simul_page,
        simul_detail_page,
        test_page,
    ],
}

pg = st.navigation(pages)

# Spuštění aplikace
pg.run()