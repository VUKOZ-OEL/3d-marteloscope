import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.express as px
import src.io_utils as iou
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Dict, List, Union
import json

# ---------- DATA ----------

plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- HLAVIČKA ----------
st.markdown(
    f"### :orange[**{plot_info['name'].iloc[0]}**]",
)


# ---------- OVLÁDÁNÍ ----------
c1, c2 = st.columns([2,2])

with c1:
    st.markdown("###")
    st.markdown("##### Overview:")
    st.markdown("")
    st.markdown(f"""
    - **Forest type:** {plot_info['forest_type'].iloc[0]}
    - **Number of trees:** {plot_info['no_trees'].iloc[0]}
    - **Wood volume:** {plot_info['volume'].iloc[0]} m³
    - **Area:** {plot_info['size_ha'].iloc[0]} ha
    - **Altitude:** {plot_info['altitude'].iloc[0]} m
    - **Precipitation:** {plot_info['precipitation'].iloc[0]} mm/year
    - **Average temperature:** {plot_info['temperature'].iloc[0]} °C
    - **Established:** {plot_info['established'].iloc[0]}
    - **Location:** {plot_info['state'].iloc[0]}
    - **Owner:** {plot_info['owner'].iloc[0]}
    """)


with c2:
    st.markdown("###")
    st.markdown("##### Controls:")
    st.markdown("")
    st.selectbox("**Management examples:**",["some example","another"])
    st.button("**Load example**", icon=":material/model_training:")
    st.markdown("###")
    st.button("Export results",icon=":material/file_save:")
    st.button("Clear management",icon=":material/delete:")
    st.markdown("... other settings")