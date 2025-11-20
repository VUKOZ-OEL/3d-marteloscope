import streamlit as st
import src.io_utils as iou


# Set Page Title
st.set_page_config(page_title="3D-Marteloscope", page_icon=":material/nature_people:",layout="wide")

# Dash page with summary
dash_page = st.Page("src/Dashboard.py", title="Info & controls", icon=":material/dashboard:")

# Tree & Crown stats
summary_page = st.Page("src/Summary.py", title="Summary", icon=":material/info:")
tree_page = st.Page("src/Tree_stats.py", title="Tree Statistics", icon=":material/nature:")
canopy_page = st.Page("src/Canopy_stats.py", title="Canopy Occupancy", icon=":material/forest:")
space_page = st.Page("src/Space_comp.py", title="Space Competition", icon=":material/join:")
light_page = st.Page("src/Light_comp.py", title="Sky View Factor", icon=":material/light_mode:")
heatmap_page = st.Page("src/Heatmaps.py", title="Heatmaps", icon=":material/blur_on:")

# Map, Att tab, pygpage
map_page = st.Page("src/Map.py", title="Plot Map", icon=":material/map:")
att_page = st.Page("src/Attributes.py", title="Attribute Table", icon=":material/data_table:")
analytics_page = st.Page("src/Analytics.py", title="Analytics - Experimental", icon=":material/addchart:")

# Simulation results 
simul_page = st.Page("src/Simulation.py", title="Simulation", icon=":material/clock_arrow_up:")
simul_detail_page = st.Page("src/Simulation_detail.py", title="Deatiled view", icon=":material/frame_inspect:")
add_atts_page = st.Page("src/Add_attributes_prj.py", title="Add attributes", icon=":material/list_alt_add:")
colors_page = st.Page("src/Colors_settings.py", title="Colors", icon=":material/colors:")
# Sandbox
test_page2 = st.Page("src/Tree_stats_v2.py", title="Tree stats old")
test_page = st.Page("src/sandbox.py", title="sandbox")
comment_page = st.Page("src/comments.py", title="comments")

file_path = "data/test_project.json"

# Init data
if not st.session_state.get("data_initialized"):
    st.session_state.project_file = file_path
    st.session_state.trees = iou.load_project_json(file_path)
    st.session_state.plot_info = iou.load_plot_info(file_path)
    st.session_state.color_palette = iou.load_color_pallete(file_path)
    st.session_state.data_initialized = True

# Define common labels:
st.session_state.Before = "Before Cut"
st.session_state.After = "After Cut"
st.session_state.Removed = "Harvested"

st.session_state.Management = "Cutting purpose"
st.session_state.Species = "Species"

st.session_state.plot_title_font = dict(size=18, color="#33343f", weight="bold")

pages = {
    "Main": [
        dash_page,
    ],
    "Basic Results:": [
        summary_page,
        map_page,
        heatmap_page,
        tree_page,

    ],
    "Expert Results": [
        canopy_page,
        space_page,
        light_page,
        #att_page,
        #analytics_page,
    ],
    "Growt Simulation": [
                simul_page,
        simul_detail_page,
    ],
    "Settings":[
        add_atts_page,
        colors_page,
    ],
    "Sanbox":[
        test_page2,
        comment_page,
        test_page,
    ]
}

pg = st.navigation(pages)

# Spuštění aplikace
pg.run()