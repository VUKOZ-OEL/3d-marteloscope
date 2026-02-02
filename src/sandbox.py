# src/Simulation.py
import sys
import ctypes, os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import src.io_utils as iou
import src.simul_utils as sut

from src.i18n import t
import shutil
from src.species_dict import species_dict

root = "C:/Users/krucek/Documents/iLand/test/rep_out"



# ------------------------------------------------------------------------------
# Get directory of the running script
dll_path = "C:/Users/krucek/Documents/GitHub/VUK/3d-forest/out/install/x64-Debug/bin/ILandModel.dll"

os.add_dll_directory(dll_path)

# Load the shared library
iland = ctypes.CDLL(str(dll_path))

# Define argument and return types
iland.runilandmodel.argtypes = [ctypes.c_char_p, ctypes.c_int]
iland.runilandmodel.restype = ctypes.c_int

# Call the function
years = 10

xml_path = b"C:\\Users\\krucek\\Documents\\iLand\\test\\Pokojna_hora.xml"
out_db = Path("C:\\Users\\krucek\\Documents\\iLand\\test\\output\\output.sqlite")
temp_db = Path("C:\\Users\\krucek\\Documents\\iLand\\test\\output\\temp.sqlite")

all_living = []

for i in range(1, 10):
    print(f"Running replication {i}/{10}")

    # --- spusť iLand ---
    iland.runilandmodel(xml_path, years)

    shutil.copyfile(out_db, temp_db)

    # --- průběžně načti ---
    living= sut.read_single_sqlite(temp_db, rep_id=f"rep_{i:03d}")

    all_living.append(living)

if not all_living:
    st.error("No simulation outputs loaded.")
    st.stop()

sim_trees = pd.concat(all_living, ignore_index=True)

def _agg_ci(df, group_cols, value_col, rep_col="replication"):
    per_rep = (
        df.groupby([rep_col] + group_cols, as_index=False)[value_col]
          .sum()
          .rename(columns={value_col: "value"})
    )

    stats = (
        per_rep.groupby(group_cols, as_index=False)["value"]
        .agg(
            mean="mean",
            low=lambda x: np.nanquantile(x, 0.025),
            high=lambda x: np.nanquantile(x, 0.975),
        )
    )
    return stats


stats = _agg_ci(sim_trees, ["year", "species"], "volume_m3")

print(stats)
