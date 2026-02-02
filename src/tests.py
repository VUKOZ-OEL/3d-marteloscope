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




st.markdown(f"#### {t('simulation_header')}")

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

for i in range(1, 10):
    print(f"Running replication {i}/{10}")

    
    if out_db.exists():
        out_db.unlink()

    # --- spusť iLand ---
    iland.runilandmodel(xml_path, years)

    shutil.copyfile(out_db, temp_db)

    # --- průběžně načti ---
    living, death = sut.read_single_sqlite(temp_db, rep_id=f"rep_{i:03d}")

    all_living.append(living)
    if death is not None:
        all_death.append(death)

view(living)



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
        f"**{t('button_resater_simulation')}**",
        icon=":material/play_arrow:",
        width="stretch",
        type="primary",
    )

with c_mid:
    st.markdown(f"**{t('simulation_period')}**")
    year = st.slider("", min_value=0, max_value=100, value=30, step=5)

with c_right:
    st.markdown(f"**{t('simulation_options')}**")
    mortality = st.toggle(t('mortality_box'))
    regeneration = st.toggle(t('regeneration_box'))

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

    Fixes:
      - pokud se v simulaci objeví stromy mimo vstupní set (nebo dojde k recyklaci ID),
        zařadí se do kategorie management_status = 'Regeneration'
    """
    out = sim_trees.copy()

    # always add label
    out["species_label"] = out["species"].map(
        lambda c: code2label.get(str(c).lower(), str(c).title())
    )

    trees = _get_trees_df()
    input_ids = set()
    if not trees.empty and "id" in trees.columns:
        input_ids = set(pd.to_numeric(trees["id"], errors="coerce").dropna().astype(int).tolist())

    # --- helpers for regeneration detection ---
    def _is_regeneration(df: pd.DataFrame) -> pd.Series:
        # 1) Prefer age-based detection (iLand typically provides 'age')
        if "age" in df.columns and "year" in df.columns:
            yr = pd.to_numeric(df["year"], errors="coerce")
            age = pd.to_numeric(df["age"], errors="coerce")
            est_year = yr - age  # establishment year relative to sim start
            return est_year.fillna(-1) > 0  # established after year 0 => regeneration
        # 2) fallback: id not in original input
        if "id" in df.columns and input_ids:
            ids = pd.to_numeric(df["id"], errors="coerce").fillna(-1).astype(int)
            return ~ids.isin(list(input_ids))
        return pd.Series(False, index=df.index)

    regen_mask = _is_regeneration(out)

    # fallback colors from palette if we cannot join
    if trees.empty or "id" not in trees.columns or "id" not in out.columns:
        out["speciesColorhex"] = out["species"].map(lambda c: code2color.get(str(c).lower()))
        out["management_status"] = np.where(regen_mask, "Regeneration", pd.NA)
        out["managementColorHex"] = np.where(regen_mask, "#2ca02c", pd.NA)
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

    # --- Management status fix: only keep input categories; everything else => Regeneration ---
    keep_status = {"Target tree", "Untouched"}

    # if regen by age / missing id in inputs => Regeneration
    out.loc[regen_mask, "management_status"] = "Regeneration"
    out.loc[regen_mask, "managementColorHex"] = "#2ca02c"

    # if not regen, but status is missing or not one of allowed -> likely ID reuse; treat as Regeneration too
    not_keep = ~out["management_status"].isin(list(keep_status))
    out.loc[not_keep, "management_status"] = "Regeneration"
    out.loc[not_keep, "managementColorHex"] = out.loc[not_keep, "managementColorHex"].where(
        out.loc[not_keep, "managementColorHex"].notna() & (out.loc[not_keep, "managementColorHex"].astype(str).str.strip() != ""),
        "#2ca02c",
    )

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


def _agg_ci(df: pd.DataFrame, group_cols: list[str], value_col: str, rep_col: str = "replication") -> pd.DataFrame:
    """Agreguje přes replikace a vrátí mean + 95% interval (2.5–97.5 percentil)."""
    d = df.copy()
    if rep_col not in d.columns:
        d[rep_col] = "rep_1"

    per_rep = (
        d.groupby([rep_col] + group_cols, as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: "value"})
    )

    stats = (
        per_rep.groupby(group_cols, as_index=False)["value"]
        .agg(
            mean="mean",
            low=lambda x: float(np.nanquantile(x, 0.025)) if len(x) else np.nan,
            high=lambda x: float(np.nanquantile(x, 0.975)) if len(x) else np.nan,
        )
    )
    return stats



def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' or 'RRGGBB' to 'rgba(r,g,b,a)'."""
    if not isinstance(hex_color, str):
        return f"rgba(0,0,0,{alpha})"
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return f"rgba(0,0,0,{alpha})"
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return f"rgba(0,0,0,{alpha})"


def _add_line_with_ci(fig: go.Figure, x, y, low, high, name: str, color: str | None, width: int = 2):
    # upper (invisible), then lower with fill to create ribbon
    fig.add_trace(
        go.Scatter(
            x=x,
            y=high,
            mode="lines",
            line=dict(color=color, width=0),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=low,
            mode="lines",
            line=dict(color=color, width=0),
            fill="tonexty",
            fillcolor=(_hex_to_rgba(color, 0.20) if color else "rgba(0,0,0,0.15)"),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
            hovertemplate="Year=%{x}<br>Mean=%{y:.2f} m³<br>95% CI=[%{customdata[0]:.2f}, %{customdata[1]:.2f}] m³<extra></extra>",
            customdata=np.column_stack([low, high]),
        )
    )


def _fig_volume_by_species(sim_trees: pd.DataFrame, max_year: int) -> go.Figure:
    df = sim_trees.copy()
    df = df[df["year"] <= max_year]

    # stats per year & species_label across replications
    stats = _agg_ci(df, ["year", "species_label"], "volume_m3")
    stats = stats.sort_values(["species_label", "year"])

    # color map per species_label
    c = (
        df[["species_label", "speciesColorhex"]]
        .dropna()
        .drop_duplicates("species_label")
        .set_index("species_label")["speciesColorhex"]
        .to_dict()
    )

    # total stats (sum across species per year)
    total_stats = _agg_ci(df, ["year"], "volume_m3").sort_values("year")

    fig = go.Figure()

    for sp in sorted(stats["species_label"].unique()):
        s = stats[stats["species_label"] == sp].sort_values("year")
        if s.empty or float(s["mean"].fillna(0).max()) <= 0:
            continue
        _add_line_with_ci(
            fig,
            x=s["year"],
            y=s["mean"],
            low=s["low"],
            high=s["high"],
            name=sp,
            color=c.get(sp),
            width=2,
        )

    # SUM line with CI
    _add_line_with_ci(
        fig,
        x=total_stats["year"],
        y=total_stats["mean"],
        low=total_stats["low"],
        high=total_stats["high"],
        name="SUM",
        color=None,
        width=4,
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

    stats = _agg_ci(df, ["year", "management_status"], "volume_m3")
    stats = stats.sort_values(["management_status", "year"])

    c = (
        df[["management_status", "managementColorHex"]]
        .dropna()
        .drop_duplicates("management_status")
        .set_index("management_status")["managementColorHex"]
        .to_dict()
    )
    c.setdefault("Regeneration", "#2ca02c")

    total_stats = _agg_ci(df, ["year"], "volume_m3").sort_values("year")

    fig = go.Figure()

    preferred = ["Target tree", "Untouched", "Regeneration"]
    all_statuses = [s for s in stats["management_status"].dropna().unique()]
    statuses_sorted = [s for s in preferred if s in all_statuses] + [
        s for s in sorted(all_statuses) if s not in preferred
    ]

    for ms in statuses_sorted:
        s = stats[stats["management_status"].eq(ms)].sort_values("year")
        if s.empty or float(s["mean"].fillna(0).max()) <= 0:
            continue

        name = str(ms)
        _add_line_with_ci(
            fig,
            x=s["year"],
            y=s["mean"],
            low=s["low"],
            high=s["high"],
            name=name,
            color=c.get(ms),
            width=2,
        )

    _add_line_with_ci(
        fig,
        x=total_stats["year"],
        y=total_stats["mean"],
        low=total_stats["low"],
        high=total_stats["high"],
        name="SUM",
        color=None,
        width=4,
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
        sim_trees = load_simulation(root, keep_replication=True)
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



# =========================
# Debug (temporary)
# =========================
with st.expander("Debug: CI inputs / replication", expanded=False):
    st.write("Columns:", list(sim_trees.columns))
    if "replication" in sim_trees.columns:
        st.write("Unique replications:", int(sim_trees["replication"].nunique()))
        st.write("Replications:", sorted(sim_trees["replication"].dropna().unique().tolist())[:20])
    else:
        st.warning("Column 'replication' is missing -> CI will collapse to a single line (low==high).")

    # ---- Species CI table ----
    try:
        _df = sim_trees[sim_trees["year"] <= year].copy()
        _sp_stats = _agg_ci(_df, ["year", "species_label"], "volume_m3")
        _sp_stats["ci_width"] = _sp_stats["high"] - _sp_stats["low"]
        st.write("Species CI: rows with width>0:", int((_sp_stats["ci_width"] > 0).sum()), "/", int(len(_sp_stats)))
        st.dataframe(_sp_stats.sort_values("ci_width", ascending=False).head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Species CI debug failed: {e}")

    # ---- Management CI table ----
    try:
        _mg = sim_trees[sim_trees["year"] <= year].copy()
        if "management_status" not in _mg.columns:
            _mg["management_status"] = pd.NA
        _mg_stats = _agg_ci(_mg, ["year", "management_status"], "volume_m3")
        _mg_stats["ci_width"] = _mg_stats["high"] - _mg_stats["low"]
        st.write("Management CI: rows with width>0:", int((_mg_stats["ci_width"] > 0).sum()), "/", int(len(_mg_stats)))
        st.dataframe(_mg_stats.sort_values("ci_width", ascending=False).head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Management CI debug failed: {e}")

    # ---- Deeper check: do per-rep totals actually vary? ----
    try:
        _df = sim_trees[sim_trees["year"] <= year].copy()
        _df["volume_m3"] = pd.to_numeric(_df["volume_m3"], errors="coerce")
        per_rep = (
            _df.groupby(["replication", "year", "species_label"], as_index=False)["volume_m3"]
              .sum()
              .rename(columns={"volume_m3": "value"})
        )

        # std across replications for each (year, species)
        spread = (
            per_rep.groupby(["year", "species_label"], as_index=False)["value"]
                  .agg(n="count", mean="mean", std="std", min="min", max="max")
        )
        spread["range"] = spread["max"] - spread["min"]
        st.write("Max std across replications (year/species):", float(spread["std"].fillna(0).max()))
        st.write("Max range across replications (year/species):", float(spread["range"].fillna(0).max()))
        st.dataframe(spread.sort_values(["range", "std"], ascending=False).head(15), use_container_width=True)

        # show a concrete distribution for the worst (year,species)
        worst = spread.sort_values(["range", "std"], ascending=False).head(1)
        if len(worst):
            wy = int(worst["year"].iloc[0])
            ws = str(worst["species_label"].iloc[0])
            st.write(f"Sample distribution for worst group: year={wy}, species_label='{ws}'")
            dist = per_rep[(per_rep["year"] == wy) & (per_rep["species_label"] == ws)].sort_values("value", ascending=False)
            st.dataframe(dist.head(30), use_container_width=True)
            st.write("Unique values in this group:", int(dist["value"].nunique()))
    except Exception as e:
        st.error(f"Deeper per-rep variance debug failed: {e}")

    # ---- Extra check: are replications actually different at the tree-level? ----
    try:
        if "replication" in sim_trees.columns:
            base = sim_trees.copy()
            base["year"] = pd.to_numeric(base["year"], errors="coerce")
            base["id"] = pd.to_numeric(base["id"], errors="coerce")
            base["dbh"] = pd.to_numeric(base.get("dbh"), errors="coerce") if "dbh" in base.columns else np.nan
            base["volume_m3"] = pd.to_numeric(base.get("volume_m3"), errors="coerce") if "volume_m3" in base.columns else np.nan

            # basic per-rep fingerprint
            fp = (
                base.groupby("replication")
                    .agg(rows=("id", "size"),
                         max_year=("year", "max"),
                         sum_vol=("volume_m3", "sum"))
                    .reset_index()
            )
            st.write("Per-replication fingerprint (rows / max_year / sum_vol):")
            st.dataframe(fp.sort_values("replication").head(30), use_container_width=True)
            st.write("Fingerprint variability:",
                     {
                         "rows_unique": int(fp["rows"].nunique()),
                         "max_year_unique": int(fp["max_year"].nunique()),
                         "sum_vol_unique": int(fp["sum_vol"].round(6).nunique()),
                     })

            # pick a (year,id) that appears in many replications and compare dbh/volume across reps
            cand = (
                base.dropna(subset=["id", "year"])
                    .groupby(["year", "id"], as_index=False)
                    .agg(n_rep=("replication", "nunique"))
                    .sort_values("n_rep", ascending=False)
            )
            if len(cand):
                pick = cand.iloc[0]
                py, pid, nrep = int(pick["year"]), int(pick["id"]), int(pick["n_rep"])
                st.write(f"Tree-level variability sample: year={py}, id={pid} (present in {nrep} replications)")
                sample = base[(base["year"] == py) & (base["id"] == pid)][["replication", "dbh", "volume_m3", "species_label", "management_status"]].copy()
                st.dataframe(sample.sort_values("replication").head(50), use_container_width=True)
                st.write("Unique dbh values:", int(sample["dbh"].round(6).nunique()))
                st.write("Unique volume_m3 values:", int(sample["volume_m3"].round(6).nunique()))
    except Exception as e:
        st.error(f"Tree-level replication debug failed: {e}")

with st.expander(label=t("expander_help_label"),icon=":material/help:"):
    st.markdown(t("prediction_help"))