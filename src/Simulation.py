# src/Simulation.py
import ctypes
import os
from pathlib import Path
import shutil

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import src.io_utils as iou
import src.simul_utils2 as sut
from src.i18n import t,t_help
from src.species_dict import species_dict


# =============================================================================
# HEADER
# =============================================================================
st.markdown(f"#### {t('simulation_header')}")

trees = st.session_state.trees
project_file = st.session_state.project_file

out_csv = sut.export_iland_trees_csv(
    trees=trees,
    species_dict=species_dict,
    project_file=project_file,
)
print(out_csv)

# =============================================================================
# iLand DLL
# =============================================================================
dll_path = "C:/Users/krucek/Documents/GitHub/VUK/3d-forest/out/install/x64-Debug/bin/ILandModel.dll"
os.add_dll_directory(os.path.dirname(dll_path))

iland = ctypes.CDLL(dll_path)
iland.runilandmodel.argtypes = [ctypes.c_char_p, ctypes.c_int]
iland.runilandmodel.restype = ctypes.c_int


# =============================================================================
# CONFIG
# =============================================================================


xml_path = Path("C:\\Users\\krucek\\Documents\\iLand\\test\\Pokojna_hora.xml")
out_db = Path("C:/Users/krucek/Documents/iLand/test/output/output.sqlite")
temp_db = Path("C:/Users/krucek/Documents/iLand/test/output/temp.sqlite")


# =============================================================================
# UI
# =============================================================================
c1, _, c2,_,c3, _, c4 = st.columns([2,0.25,2,0.25,2,0.25, 3])

with c1:
    st.markdown("####")
    run_simul = st.button(
        f"**{t('button_resater_simulation')}**",
        icon=":material/play_arrow:",
        width="stretch",
        type="primary",
    )

with c2:
    st.markdown(f"**{t('simulation_period')}**")
    years = st.slider("", min_value=0, max_value=100, value=30, step=5)

with c3:
    st.markdown(f"**{t('replications')}**")
    n_rep = st.slider("", min_value=1, max_value=100, value=10, step=1)

with c4:
    st.markdown(f"**{t('simulation_options')}**")
    c4l,c4r= st.columns([1,1])

    with c4l:
        mortality = st.toggle(t("mortality_box"), value = True)
    with c4r:
        regeneration = st.toggle(t("regeneration_box"), value = True)



st.divider()


# =============================================================================
# RUN SIMULATION (MONTE CARLO)
# =============================================================================
if run_simul:
    all_living = []

    progress = st.progress(0.0, text="Running Monte Carlo simulations…")

    sut.set_iland_mortality_regeneration(xml_path,mortality,regeneration)

    for i in range(1, n_rep + 1):
        progress.progress(i / n_rep, text=f"Replication {i} / {n_rep}")

        iland.runilandmodel(str(xml_path).encode("utf-8"), years)
        shutil.copyfile(out_db, temp_db)

        living = sut.read_single_sqlite_living(
            temp_db,
            rep_id=f"rep_{i:03d}"
        )
        if not living.empty:
            all_living.append(living)

    progress.empty()

    if not all_living:
        st.error("Simulation produced no outputs.")
        st.stop()

    sim_trees = pd.concat(all_living, ignore_index=True)

    # ---- enrich ----
    trees = st.session_state.get("trees")
    sim_trees = sut.map_species_label(sim_trees, species_dict)
    sim_trees = sut.mark_target_trees(sim_trees, trees)

    sim_trees["year"] = pd.to_numeric(sim_trees["year"], errors="coerce")

    st.session_state.sim_trees = sim_trees


# =============================================================================
# STOP HERE if no simulation yet
# =============================================================================
if "sim_trees" in st.session_state:

    sim_trees = st.session_state.sim_trees


    # =============================================================================
    # COLORS (FIXED & CORRECT)
    # =============================================================================
    palette = st.session_state.get("color_palette") or iou.load_color_palette(
        st.session_state.get("project_file", "data/test_project.json")
    )

    species_colors = palette.get("species", {})
    management_colors = palette.get("management", {})

    mg_colors = {
        "Target tree": management_colors.get("Target tree", "#F4EE00"),
        "Untouched": management_colors.get("Untouched", "#A6A6A6"),
        "SUM": "#000000",
    }


    # =============================================================================
    # FAN CHART PLOTTER
    # =============================================================================
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"


    def fig_fan(stats: pd.DataFrame, group_col: str, color_map: dict, title: str):
        fig = go.Figure()

        for g in stats[group_col].unique():
            s = stats[stats[group_col] == g].sort_values("year")
            color = color_map.get(g, "#999999")

            for lo, hi, a in [("q5", "q95", 0.15), ("q25", "q75", 0.25)]:
                fig.add_trace(go.Scatter(
                    x=s["year"], y=s[hi],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=s["year"], y=s[lo],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=_hex_to_rgba(color, a),
                    showlegend=False,
                    hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=s["year"], y=s["q50"],
                mode="lines",
                name=g,
                line=dict(color=color, width=4 if g == "SUM" else 2),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Volume (m³)",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig


    # =============================================================================
    # BUILD FAN DATA
    # =============================================================================
    df = sim_trees[sim_trees["year"] <= years]

    fan_species = pd.concat([
        sut.agg_fan(df, ["year", "species_label"]).assign(group=lambda d: d["species_label"]),
        sut.agg_fan(df, ["year"]).assign(group="SUM"),
    ])

    fan_mgmt = pd.concat([
        sut.agg_fan(df, ["year", "management_group"]).assign(group=lambda d: d["management_group"]),
        sut.agg_fan(df, ["year"]).assign(group="SUM"),
    ])


    print(fan_species)

    # =============================================================================
    # PLOTS
    # =============================================================================
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.plotly_chart(
            fig_fan(fan_species, "group", species_colors, "Volume by Species (fan chart)"),
            width="stretch",
        )

    with col2:
        st.plotly_chart(
            fig_fan(fan_mgmt, "group", mg_colors, "Volume by Management (fan chart)"),
            width="stretch",
        )

with st.expander(label=t("expander_help_label"),icon=":material/help:"):
    st.markdown(t_help("simulation_help"))