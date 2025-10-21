import streamlit as st
import pandas as pd
import src.io_utils as iou

c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### Current project file:")
    st.markdown(st.session_state.project_file)

with c2:
    if st.button("**Save project**", width="stretch",type="primary", icon=":material/save:"):
        print("save project")
    if st.button("**Save project as**", width="stretch",type="primary", icon=":material/save_as:"):
        print("save project as")



uploaded_files = st.file_uploader(
    "Import data", accept_multiple_files=True, type="csv"
)