import streamlit as st
#from streamlit_multipage import MultiPage

st.set_page_config(page_title="3D-Marteloscope", page_icon=":material/forest:",layout="wide")

dash_page = st.Page("src/Dashboard.py", title="Dashboard", icon=":material/home:")
map_page = st.Page("src/Map.py", title="Map", icon=":material/map:")
att_page = st.Page("src/Attributes.py", title="Attribute table", icon=":material/data_table:")
results_page = st.Page("src/Results.py", title="Results", icon=":material/monitoring:")
analytics_page = st.Page("src/Analytics.py", title="Analytics", icon=":material/addchart:")
test_page = st.Page("src/sandbox.py", title="_TESTs_", icon=":material/abc:")

#pg = st.navigation([dash_page, map_page,att_page,analytics_page,])

pages = {
    "Info & control": [
        dash_page,
    ],
    "Select:": [
        map_page,
        att_page,
    ],
    "Explore": [
        results_page,
        analytics_page,
        test_page,
    ],
}

pg = st.navigation(pages)
st.sidebar.button("Show in 3D")

# Spuštění aplikace
pg.run()