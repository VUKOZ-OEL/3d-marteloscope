from pygwalker.api.streamlit import StreamlitRenderer
import pandas as pd
import streamlit as st
from src.io import load_project_json

st.set_page_config(page_title="PyGWalker + Streamlit", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    data = load_project_json(path)
    df = pd.json_normalize(data) if not isinstance(data, pd.DataFrame) else data
    # Najdeme sloupce obsahující seznamy a převedeme je na tuple
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, list)).any():
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return df

@st.cache_resource
def get_renderer(df: pd.DataFrame) -> StreamlitRenderer:
    return StreamlitRenderer(df, spec_io_mode="rw", spec="./pyg_spec.json")

file_path = r"C:\Users\krucek\OneDrive - vukoz.cz\DATA\_GS-LCR\LS-Krivoklat\3df_project\Krivoklat_test_SAVE.json"
df = load_data(file_path)

renderer = get_renderer(df)
renderer.explorer()
