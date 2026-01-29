# src/Simulation.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

import src.io_utils as iou
from src.simul_utils import load_simulation, build_species_maps


st.markdown("##### Forest Growth Simulation")

root = "C:/Users/krucek/Documents/iLand/test/rep_out"

# --- palette for species (from project json) ---
# Prefer session palette if available (set in app.py), otherwise fall back
project_file = st.session_state.get("project_file", "data/test_project.json")
color_pallete = st.session_state.get("color_palette") or iou.load_color_palette(project_file)
code2latin, code2color, code2label = build_species_maps(color_pallete)


# =========================
# UI
# =========================
c_left, left_empty, c_mid, right_empty, c_right = st.columns([3, 1, 4, 1, 3])

with c_left:
    st.markdown("####")
    run_simul = st.button(
        "**Start simulation**",
        icon=":material/play_arrow:",
        width="stretch",
        type="primary",
    )

with c_mid:
    st.markdown("**Set lenght of Simulation:**")
    year = st.slider("", min_value=0, max_value=100, value=30, step=5)

with c_right:
    st.markdown("**Options:**")
    mortality = st.toggle("Mortality")
    regeneration = st.toggle("Regeneration")

st.divider()


# =========================
# Helpers
# =========================
def _get_trees_df() -> pd.DataFrame:
    if "trees" not in st.session_state or st.session_state.trees is None:
        return pd.DataFrame()
    return st.session_state.trees.copy()


def _enrich_sim_with_trees(sim_trees: pd.DataFrame) -> pd.DataFrame:
    """
    Join info from session_state.trees by id and add:
      - speciesColorhex
      - management_status
      - managementColorHex
    Also adds:
      - species_label (for legends)
    """
    out = sim_trees.copy()

    # always add label
    out["species_label"] = out["species"].map(
        lambda c: code2label.get(str(c).lower(), str(c).title())
    )

    trees = _get_trees_df()
    if trees.empty or "id" not in trees.columns or "id" not in out.columns:
        # fallback colors from palette if possible
        out["speciesColorhex"] = out["species"].map(lambda c: code2color.get(str(c).lower()))
        if "management_status" not in out.columns:
            out["management_status"] = pd.NA
        if "managementColorHex" not in out.columns:
            out["managementColorHex"] = pd.NA
        return out

    # ---- speciesColorhex from trees (try common names) ----
    if "speciesColorhex" not in trees.columns:
        if "speciesColorHex" in trees.columns:
            trees["speciesColorhex"] = trees["speciesColorHex"]
        elif "species_color" in trees.columns:
            trees["speciesColorhex"] = trees["species_color"]
        elif "speciesColor" in trees.columns:
            trees["speciesColorhex"] = trees["speciesColor"]
        else:
            trees["speciesColorhex"] = pd.NA

    # ---- management_status from trees (try common names) ----
    if "management_status" not in trees.columns:
        if "managementStatus" in trees.columns:
            trees["management_status"] = trees["managementStatus"]
        elif "management" in trees.columns:
            trees["management_status"] = trees["management"]
        elif "treatment" in trees.columns:
            trees["management_status"] = trees["treatment"]
        else:
            trees["management_status"] = pd.NA

    # ---- managementColorHex from trees (try common names) ----
    if "managementColorHex" not in trees.columns:
        if "managementColorhex" in trees.columns:
            trees["managementColorHex"] = trees["managementColorhex"]
        elif "management_color" in trees.columns:
            trees["managementColorHex"] = trees["management_color"]
        elif "managementColor" in trees.columns:
            trees["managementColorHex"] = trees["managementColor"]
        else:
            trees["managementColorHex"] = pd.NA

    trees_join = (
        trees[["id", "speciesColorhex", "management_status", "managementColorHex"]]
        .drop_duplicates("id")
        .copy()
    )

    out = out.merge(trees_join, on="id", how="left")

    # fallback speciesColorhex if missing: use palette via species code
    out["speciesColorhex"] = out.apply(
        lambda r: r["speciesColorhex"]
        if pd.notna(r.get("speciesColorhex")) and str(r.get("speciesColorhex")).strip() != ""
        else code2color.get(str(r["species"]).lower()),
        axis=1,
    )

    return out


def _fig_volume_by_species(sim_trees: pd.DataFrame, max_year: int) -> go.Figure:
    df = sim_trees.copy()
    df = df[df["year"] <= max_year]

    # SUM volume across trees per year & species
    g = (
        df.groupby(["year", "species_label"], as_index=False)["volume_m3"]
        .sum()
        .rename(columns={"volume_m3": "volume_sum_m3"})
    )

    # color map per species_label
    c = (
        df[["species_label", "speciesColorhex"]]
        .dropna()
        .drop_duplicates("species_label")
        .set_index("species_label")["speciesColorhex"]
        .to_dict()
    )

    # total line
    total = (
        g.groupby("year", as_index=False)["volume_sum_m3"]
        .sum()
        .rename(columns={"volume_sum_m3": "total_volume_sum_m3"})
        .sort_values("year")
    )

    fig = go.Figure()

    for sp in sorted(g["species_label"].unique()):
        s = g[g["species_label"] == sp].sort_values("year")
        # legenda jen pro nenulové křivky
        if s.empty or float(s["volume_sum_m3"].fillna(0).max()) <= 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=s["year"],
                y=s["volume_sum_m3"],
                mode="lines",
                name=sp,
                line=dict(color=c.get(sp)),
                hovertemplate="Year=%{x}<br>Volume SUM=%{y:.2f} m³<extra></extra>",
            )
        )

    # SUM line (overall) - vždy zobraz
    fig.add_trace(
        go.Scatter(
            x=total["year"],
            y=total["total_volume_sum_m3"],
            mode="lines",
            name="SUM",
            line=dict(width=4),
            hovertemplate="Year=%{x}<br>Total SUM=%{y:.2f} m³<extra></extra>",
        )
    )

    fig.update_layout(
        title="Volume (SUM) by Species",
        xaxis_title="Year",
        yaxis_title="Volume SUM (m³)",
        legend_title_text="Species",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _fig_volume_by_management(sim_trees: pd.DataFrame, max_year: int) -> go.Figure:
    df = sim_trees.copy()
    df = df[df["year"] <= max_year]

    if "management_status" not in df.columns:
        df["management_status"] = pd.NA

    g = (
        df.groupby(["year", "management_status"], as_index=False)["volume_m3"]
        .sum()
        .rename(columns={"volume_m3": "volume_sum_m3"})
    )

    c = (
        df[["management_status", "managementColorHex"]]
        .dropna()
        .drop_duplicates("management_status")
        .set_index("management_status")["managementColorHex"]
        .to_dict()
    )

    total = (
        g.groupby("year", as_index=False)["volume_sum_m3"]
        .sum()
        .rename(columns={"volume_sum_m3": "total_volume_sum_m3"})
        .sort_values("year")
    )

    fig = go.Figure()

    preferred = ["Target tree", "Untouched"]
    all_statuses = [s for s in g["management_status"].dropna().unique()]
    statuses_sorted = [s for s in preferred if s in all_statuses] + [
        s for s in sorted(all_statuses) if s not in preferred
    ]

    for ms in statuses_sorted:
        s = g[g["management_status"].eq(ms)].sort_values("year")
        # legenda jen pro nenulové křivky
        if s.empty or float(s["volume_sum_m3"].fillna(0).max()) <= 0:
            continue

        name = str(ms)
        fig.add_trace(
            go.Scatter(
                x=s["year"],
                y=s["volume_sum_m3"],
                mode="lines",
                name=name,
                line=dict(color=c.get(ms)),
                hovertemplate="Year=%{x}<br>Volume SUM=%{y:.2f} m³<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=total["year"],
            y=total["total_volume_sum_m3"],
            mode="lines",
            name="SUM",
            line=dict(width=4),
            hovertemplate="Year=%{x}<br>Total SUM=%{y:.2f} m³<extra></extra>",
        )
    )

    fig.update_layout(
        title="Volume (SUM) by Management Status",
        xaxis_title="Year",
        yaxis_title="Volume SUM (m³)",
        legend_title_text="Management",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# =========================
# Load + render
# =========================
# (volitelně) reload při kliknutí na Start simulation
if run_simul:
    st.session_state.pop("sim_trees", None)

if "sim_trees" not in st.session_state:
    with st.spinner(
        "Loading and processing outcomes of forest growth simulation, please wait.",
        show_time=True,
    ):
        sim_trees = load_simulation(root)
        sim_trees = _enrich_sim_with_trees(sim_trees)
        st.session_state.sim_trees = sim_trees

sim_trees = st.session_state.sim_trees

# ensure numeric year
if "year" in sim_trees.columns:
    sim_trees["year"] = pd.to_numeric(sim_trees["year"], errors="coerce")

# =========================
# Charts side-by-side
# =========================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.plotly_chart(_fig_volume_by_species(sim_trees, max_year=year), use_container_width=True)

with col2:
    st.plotly_chart(_fig_volume_by_management(sim_trees, max_year=year), use_container_width=True)
