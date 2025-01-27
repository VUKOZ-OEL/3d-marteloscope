import streamlit as st
from src.io import *
from bokeh.plotting import figure
#from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS
import pandas as pd



# Přepínač pro zobrazení barev
color_by = st.radio("Color by:", ["status", "species"], index=0, horizontal = True)
