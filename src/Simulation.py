# pages/Simulation.py
import streamlit as st
import pandas as pd
import src.io_utils as iou

from src.simul_utils import (
    prepare_tree_dataset,
    load_simulation,
    build_species_maps
)

from src.simul_plots import (
    fig_sim_vol_height_ba
)


st.markdown("##### Forest Growth Simulation")

root = "C:/Users/krucek/Documents/iLand/test/rep_out"

color_pallete = iou.load_color_pallete("data/test_project.json")
code2latin, code2color, code2label = build_species_maps(color_pallete)


c_left,left_empty, c_mid,right_empty, c_right = st.columns([3,1, 4,1, 3])

with c_left:
    st.markdown("####")
    run_simul = st.button("**Start simulation**", icon=":material/play_arrow:", width="stretch",type="primary")

with c_mid:
    st.markdown("**Set lenght of Simulation:**")
    year = st.slider("", min_value=0, max_value=100, value=30, step=5)

with c_right:
    st.markdown("**Options:**")
    mortality = st.toggle("Mortality")
    regeneration = st.toggle("Regeneration")

st.divider()

if "sim_trees" not in st.session_state:
    with st.spinner("Loading and processing outcomes of forest growth simulation, please wait.", show_time=True):
        #st.markdown("The simulation was repeted 100 times to ensure reproducible results.")
        sim_trees = load_simulation(root)
        st.session_state.sim_trees = sim_trees
        sim_trees["species_label"] = sim_trees["species"].map(lambda c: code2label.get(str(c).lower(), str(c).title()))
        sim_trees["species_color"] = sim_trees["species"].map(lambda c: code2color.get(str(c).lower()))
        st.plotly_chart(fig_sim_vol_height_ba(sim_trees, code2color, code2label, smooth_window=3), use_container_width=True)
        print(sim_trees)

#sim_trees.to_feather("C:/Users/krucek/Documents/iLand/test/rep_out/sim_trees.feather")





