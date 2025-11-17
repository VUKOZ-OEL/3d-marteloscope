# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Streamlit: Crown volume profiles by height
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import math
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.io_utils as iou

st.markdown("#### Explore canopy statistics: Crown volume profiles by height")

# --- Data ---
if "trees" not in st.session_state:
    file_path = ("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json")
    st.session_state.trees = iou.load_project_json(file_path)

df: pd.DataFrame = st.session_state.trees.copy()

# ========== SETTINGS ==========
CHART_HEIGHT = 420
Before  = "Original Stand"
After   = "Managed Stand"
Removed = "Removed from Stand"

# --- UI controls ---
c1, c2, c3 = st.columns([2,2,2])
with c2:
    layers_mode = st.segmented_control(
        "**Show Values by:**",
        options=["Species", "Management"],
        default="Species",
        help="Select one or both. When both are selected, both are shown (Species = solid, Management = dashed).",
        selection_mode="multi",
        width="stretch"
    )

# zajistit list a aspoň jednu vrstvu
if not layers_mode:
    layers_mode = ["Species"]
elif isinstance(layers_mode, str):
    layers_mode = [layers_mode]

primary = layers_mode[0]
overlay = layers_mode[1] if len(layers_mode) > 1 else None

# area
try:
    area_ha = float(st.session_state.plot_info['size_ha'].iloc[0])
    if not np.isfinite(area_ha) or area_ha <= 0:
        area_ha = 1.0
except Exception:
    area_ha = 1.0

# masks
keep_status = {"Target tree", "Untouched"}
if "management_status" in df.columns:
    mask_after   = df["management_status"].isin(keep_status)
    mask_removed = ~mask_after
else:
    mask_after   = pd.Series(False, index=df.index)
    mask_removed = pd.Series(False, index=df.index)

# colors
def _species_colors(df_all: pd.DataFrame) -> dict:
    if "species" not in df_all.columns or "speciesColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(species=lambda d: d["species"].astype(str))
                 .groupby("species")["speciesColorHex"].first())
    return tmp.to_dict()

def _management_colors(df_all: pd.DataFrame) -> dict:
    if "management_status" not in df_all.columns or "managementColorHex" not in df_all.columns:
        return {}
    tmp = (df_all.assign(mgmt=lambda d: d["management_status"].astype(str))
                 .groupby("mgmt")["managementColorHex"].first())
    return tmp.to_dict()

def _nice_upper(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 500.0

    return math.ceil(value / 500.0) * 500.0

def _to_list_of_dicts(v):
    """Bezpečně převede crownVoxelCountShared na list[dict(treeId, count)]."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict) and "treeId" in x and "count" in x]
    if isinstance(v, str):
        import json
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return [
                    x for x in parsed
                    if isinstance(x, dict) and "treeId" in x and "count" in x
                ]
        except Exception:
            return []
    return []


def _safe_get(d, key):
    try:
        return d.get(key, None)
    except Exception:
        return None


def _as_counts(seq) -> list[int]:
    if isinstance(seq, (list, tuple, np.ndarray, pd.Series)):
        return [int(x) if pd.notna(x) else 0 for x in seq]
    if isinstance(seq, str):
        try:
            return _as_counts(ast.literal_eval(seq))
        except Exception:
            return []
    return []

def crown_volume_per_tree(df_sub: pd.DataFrame) -> pd.Series:
    if "crownVoxelSize" not in df_sub.columns or "crownVoxelCountPerMeters" not in df_sub.columns:
        return pd.Series(0.0, index=df_sub.index, dtype=float)
    voxel_size_m = pd.to_numeric(df_sub["crownVoxelSize"], errors="coerce")
    voxel_vol = voxel_size_m.pow(3)
    counts_series = df_sub["crownVoxelCountPerMeters"].apply(_as_counts)
    counts_sum = counts_series.apply(lambda lst: float(np.nansum(lst)) if len(lst) else 0.0)
    vol = counts_sum * voxel_vol.fillna(0.0)
    return vol.fillna(0.0).astype(float)

def expand_crown_volume(df_sub: pd.DataFrame, group_col: str) -> pd.DataFrame:
    required = {"crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize", group_col}
    missing = required - set(df_sub.columns)
    if missing:
        st.warning("Missing columns for crown volume profile: " + ", ".join(sorted(missing)))
        return pd.DataFrame(columns=["height_bin", group_col, "volume"])

    out_h, out_g, out_v = [], [], []
    for _, row in df_sub.iterrows():
        grp = str(row.get(group_col))
        h0  = pd.to_numeric(row.get("crownStartHeight"), errors="coerce")
        vs  = pd.to_numeric(row.get("crownVoxelSize"),   errors="coerce")
        seq = row.get("crownVoxelCountPerMeters")
        if not (np.isfinite(h0) and np.isfinite(vs) and vs > 0):
            continue
        counts = _as_counts(seq)
        if not counts:
            continue
        base = int(math.floor(float(h0)))
        voxel_vol = float(vs) ** 3
        for i, c in enumerate(counts):
            try:
                c_int = int(c)
            except Exception:
                continue
            if c_int <= 0:
                continue
            hb = base + i
            out_h.append(hb)
            out_g.append(grp)
            out_v.append(c_int * voxel_vol)

    if not out_h:
        return pd.DataFrame(columns=["height_bin", group_col, "volume"])

    long = pd.DataFrame({"height_bin": out_h, group_col: out_g, "volume": out_v})
    return long.groupby(["height_bin", group_col], as_index=False)["volume"].sum()

def profiles_by(df_sub: pd.DataFrame, H: int, group_col: str, color_map: dict) -> pd.DataFrame:
    long = expand_crown_volume(df_sub, group_col)
    if group_col not in df_sub.columns:
        groups = []
    else:
        groups = sorted(df_sub[group_col].astype(str).unique().tolist())
    if not groups:
        full_idx = pd.MultiIndex.from_product([range(0, H), []], names=["height_bin", group_col])
        return pd.DataFrame(index=full_idx).reset_index().assign(volume=0.0, color="#AAAAAA")
    full_idx = pd.MultiIndex.from_product([range(0, H), groups], names=["height_bin", group_col])
    prof = (long.set_index(["height_bin", group_col])["volume"]
                .reindex(full_idx, fill_value=0.0)
                .reset_index())
    prof["color"] = prof[group_col].map(color_map).fillna("#AAAAAA")
    return prof

# ===============================================================
# =============== ÚPRAVA: SUM PROFIL + TRACES ===================
# ===============================================================
def sum_profile(df_sub: pd.DataFrame, H: int) -> np.ndarray:
    """Vrací sumu objemů po výškových metrech pro celý subset."""
    prof = expand_crown_volume(df_sub.assign(__sum__="Sum"), "__sum__")
    if prof.empty:
        return np.zeros(H, dtype=float)
    d = (prof.groupby("height_bin")["volume"]
             .sum()
             .reindex(range(0, H), fill_value=0.0))
    return d.to_numpy(dtype=float)
# ===============================================================


def render_crown_volume_profiles(df_all: pd.DataFrame, primary: str, overlay: str | None):
    need = {"height", "crownStartHeight", "crownVoxelCountPerMeters", "crownVoxelSize"}
    miss = need - set(df_all.columns)
    if miss:
        st.warning("Missing columns: " + ", ".join(sorted(miss)))
        return

    hmax = pd.to_numeric(df_all["height"], errors="coerce").max()
    if not np.isfinite(hmax):
        st.warning("Invalid values in 'height'.")
        return
    H = max(1, int(math.ceil(float(hmax))))
    y_centers = np.arange(0, H, 1) + 0.5

    species_cmap = _species_colors(df_all)
    mgmt_cmap    = _management_colors(df_all)

    df_before  = df_all
    df_after   = df_all[mask_after]
    df_removed = df_all[mask_removed]

    def _colormap(col: str):
        return species_cmap if col == "species" else mgmt_cmap

    def _resolve_col(name: str) -> str:
        return "species" if name == "Species" else "management_status"

    primary_col  = _resolve_col(primary)
    overlay_col  = _resolve_col(overlay) if overlay else None
    if overlay_col == primary_col:
        overlay_col = None

    prof_before  = profiles_by(df_before,  H, primary_col, _colormap(primary_col))
    prof_after   = profiles_by(df_after,   H, primary_col, _colormap(primary_col))
    prof_removed = profiles_by(df_removed, H, primary_col, _colormap(primary_col))

    if overlay_col is not None:
        overlay_before  = profiles_by(df_before,  H, overlay_col, _colormap(overlay_col))
        overlay_after   = profiles_by(df_after,   H, overlay_col, _colormap(overlay_col))
        overlay_removed = profiles_by(df_removed, H, overlay_col, _colormap(overlay_col))
    else:
        overlay_before = overlay_after = overlay_removed = None

    group_order = (prof_before.groupby(primary_col)["volume"]
                              .sum()
                              .sort_values(ascending=False)
                              .index.tolist())

    def global_max_volume():
        vals = []

        # primary profiles
        vals.extend([prof_before["volume"].max(),
                    prof_after["volume"].max(),
                    prof_removed["volume"].max()])

        # overlay profiles
        if overlay_col is not None:
            vals.extend([
                overlay_before["volume"].max(),
                overlay_after["volume"].max(),
                overlay_removed["volume"].max()
            ])

        # SUM křivky
        vals.extend([
            sum_profile(df_before,  H).max(),
            sum_profile(df_after,   H).max(),
            sum_profile(df_removed, H).max(),
        ])

        return max(vals)
    
    x_upper = _nice_upper(global_max_volume())

    def _sum_per_ha(dfx: pd.DataFrame) -> float:
        return float(crown_volume_per_tree(dfx).sum()) / float(area_ha if area_ha > 0 else 1.0)

    title_before  = f"{Before} · Σ { _sum_per_ha(df_before):.0f} m³/ha"
    title_after   = f"{After} · Σ { _sum_per_ha(df_after):.0f} m³/ha"
    title_removed = f"{Removed} · Σ { _sum_per_ha(df_removed):.0f} m³/ha"

    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=(title_before, title_after, title_removed),
        horizontal_spacing=0.06
    )

    if "plot_title_font" in st.session_state:
        fig.update_layout(
            annotations=[
                dict(text=ann.text, x=ann.x, y=ann.y, xref=ann.xref, yref=ann.yref,
                     showarrow=False, font=st.session_state.plot_title_font)
                for ann in fig.layout.annotations
            ]
        )

    def add_panel(fig_obj, prof_df: pd.DataFrame, col: int, show_legend: bool, group_col: str):
        is_species = (group_col == "species")
        width   = 6 if is_species else 4
        dash    = None if is_species else "dash"
        opacity = 0.65 if is_species else 0.8

        order = (prof_df.groupby(group_col)["volume"]
                           .sum()
                           .sort_values(ascending=False)
                           .index.tolist())
        for grp in order:
            d = (prof_df[prof_df[group_col] == grp]
                 .set_index("height_bin")
                 .reindex(range(0, H), fill_value=0.0))
            x_series = d["volume"].to_numpy(dtype=float)
            if np.allclose(x_series, 0.0):
                continue
            col_hex = (d["color"].iloc[0] if "color" in d.columns else "#AAAAAA")
            fig_obj.add_trace(
                go.Scatter(
                    x=x_series,
                    y=y_centers,
                    mode="lines",
                    name=str(grp),
                    legendgroup=f"{group_col}-{grp}",
                    showlegend=show_legend,
                    line=dict(shape="spline", width=width, dash=dash),
                    opacity=opacity,
                    hovertemplate=f"{group_col}: {grp}<br>Height: %{{y:.0f}} m<br>Volume: %{{x:.0f}} m³<extra></extra>",
                    marker_color=col_hex
                ),
                row=1, col=col
            )

    # primary
    add_panel(fig, prof_before,  col=1, show_legend=True,  group_col=primary_col)
    add_panel(fig, prof_after,   col=2, show_legend=False, group_col=primary_col)
    add_panel(fig, prof_removed, col=3, show_legend=False, group_col=primary_col)

    # overlay
    if overlay_before is not None:
        add_panel(fig, overlay_before,  col=1, show_legend=True,  group_col=overlay_col)
        add_panel(fig, overlay_after,   col=2, show_legend=False, group_col=overlay_col)
        add_panel(fig, overlay_removed, col=3, show_legend=False, group_col=overlay_col)

    # SUM křivky
    sum_before  = sum_profile(df_before,  H)
    sum_after   = sum_profile(df_after,   H)
    sum_removed = sum_profile(df_removed, H)

    for col_i, arr in [(1, sum_before), (2, sum_after), (3, sum_removed)]:
        fig.add_trace(
            go.Scatter(
                x=arr,
                y=y_centers,
                mode="lines",
                name="Sum",
                legendgroup="Sum",
                showlegend=(col_i == 1),
                line=dict(color="#555555", width=4, dash="dot"),
                hovertemplate="Sum<br>Height: %{y:.0f} m<br>Volume: %{x:.0f} m³<extra></extra>",
            ),
            row=1, col=col_i
        )

    fig.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        hovermode="y unified"
    )
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="Crown volume [m³]", row=1, col=c,
                         rangemode="tozero", range=[0, x_upper])

    fig.update_yaxes(title_text="Height above ground [m]", row=1, col=1,
                     rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=2, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig.update_yaxes(row=1, col=3, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)

    st.plotly_chart(fig, use_container_width=True)

    # ===============================================================
    # ===========  NOVÉ GRAFY: PROFILY SDÍLENÉHO PROSTORU ===========
    # ===============================================================

    def build_shared_profiles(group_col: str, color_map: dict):
        needed = {"id", "height", "CBH_m", "crownVoxelSize", "crownVoxelCountShared", group_col}
        missing = needed - set(df_all.columns)
        if missing:
            st.info(
                "Shared-space profiles not shown – missing columns: "
                + ", ".join(sorted(missing))
            )
            return None

        # meta pro stromy
        meta = pd.DataFrame({
            "id": pd.to_numeric(df_all["id"], errors="coerce").astype("Int64"),
            "height": pd.to_numeric(df_all["height"], errors="coerce"),
            "CBH_m": pd.to_numeric(df_all["CBH_m"], errors="coerce"),
            group_col: df_all[group_col].astype(str),
        })
        if group_col == "species":
            meta["color"] = df_all.get("speciesColorHex", "#AAAAAA")
        else:
            meta["color"] = df_all.get("managementColorHex", "#AAAAAA")

        # flagy stavu porostu (stejná indexace jako df_all)
        meta["after_flag"] = mask_after.values
        meta["removed_flag"] = mask_removed.values

        valid = np.isfinite(meta["CBH_m"]) & np.isfinite(meta["height"])
        meta["centroid"] = np.where(
            valid,
            (meta["CBH_m"] + meta["height"]) / 2.0,
            np.nan
        )

        meta = meta.dropna(subset=["id", "centroid"]).copy()

        # rozbalení crownVoxelCountShared -> páry stromů
        base = df_all[["id", "crownVoxelSize", "crownVoxelCountShared"]].copy()
        base["focal_id"] = pd.to_numeric(base["id"], errors="coerce").astype("Int64")
        base["_shared_list"] = base["crownVoxelCountShared"].apply(_to_list_of_dicts)
        base = base.dropna(subset=["focal_id"])
        base = base.explode("_shared_list", ignore_index=True)
        base = base.dropna(subset=["_shared_list"])

        base["neighbor_id"] = pd.to_numeric(
            base["_shared_list"].apply(lambda d: _safe_get(d, "treeId")),
            errors="coerce"
        ).astype("Int64")
        base["count"] = (
            pd.to_numeric(
                base["_shared_list"].apply(lambda d: _safe_get(d, "count")),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .clip(lower=0)
        )
        base["voxel_vol"] = pd.to_numeric(base["crownVoxelSize"], errors="coerce").pow(3)
        base["shared_volume"] = (base["count"] * base["voxel_vol"]).fillna(0.0)

        base = base[(base["shared_volume"] > 0) & base["neighbor_id"].notna()]
        if base.empty:
            return None

        # join metadat pro focální a sousední strom
        meta_f = meta.rename(columns={
            "id": "focal_id",
            "centroid": "centroid_f",
            group_col: f"{group_col}_f",
            "color": "color_f",
            "after_flag": "after_f",
            "removed_flag": "removed_f",
        })
        meta_n = meta.rename(columns={
            "id": "neighbor_id",
            "centroid": "centroid_n",
            group_col: f"{group_col}_n",
            "color": "color_n",
            "after_flag": "after_n",
            "removed_flag": "removed_n",
        })

        base = (base
                .merge(meta_f, on="focal_id", how="left")
                .merge(meta_n, on="neighbor_id", how="left"))

        base = base[np.isfinite(base["centroid_f"]) & np.isfinite(base["centroid_n"])]
        if base.empty:
            return None

        # výška sdíleného objemu = průměr centroidů
        base["pair_centroid"] = (base["centroid_f"] + base["centroid_n"]) / 2.0
        base["height_bin"] = np.floor(base["pair_centroid"]).astype(int)
        base = base[(base["height_bin"] >= 0) & (base["height_bin"] < H)]

        # skupina podle FOKÁLNÍHO stromu
        base[group_col] = base[f"{group_col}_f"].astype(str)
        base["color"] = base["color_f"].astype(str)
        base["after_f"] = base["after_f"].fillna(False)
        base["after_n"] = base["after_n"].fillna(False)
        base["removed_f"] = base["removed_f"].fillna(False)

        # stavy porostu
        before_pairs  = base.copy()
        after_pairs   = base[base["after_f"] & base["after_n"]].copy()
        removed_pairs = base[base["removed_f"]].copy()

        groups = sorted(
            pd.concat([
                before_pairs[group_col],
                after_pairs[group_col],
                removed_pairs[group_col],
            ], axis=0)
            .dropna()
            .astype(str)
            .unique()
            .tolist()
)

        def agg_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
            if not groups:
                return pd.DataFrame(columns=["height_bin", group_col, "volume", "color"])
            full_idx = pd.MultiIndex.from_product([range(0, H), groups],
                                                  names=["height_bin", group_col])
            if pairs.empty:
                prof = (pd.DataFrame(index=full_idx)
                          .reset_index()
                          .assign(volume=0.0))
            else:
                long = (pairs.groupby(["height_bin", group_col], as_index=False)["shared_volume"]
                            .sum()
                            .rename(columns={"shared_volume": "volume"}))
                prof = (long.set_index(["height_bin", group_col])["volume"]
                            .reindex(full_idx, fill_value=0.0)
                            .reset_index())
            prof["color"] = prof[group_col].map(color_map).fillna("#AAAAAA")
            return prof

        prof_before_sh  = agg_pairs(before_pairs)
        prof_after_sh   = agg_pairs(after_pairs)
        prof_removed_sh = agg_pairs(removed_pairs)

        total_before = float(before_pairs["shared_volume"].sum()) / float(area_ha if area_ha > 0 else 1.0)
        total_after  = float(after_pairs["shared_volume"].sum())  / float(area_ha if area_ha > 0 else 1.0)
        total_removed= float(removed_pairs["shared_volume"].sum())/ float(area_ha if area_ha > 0 else 1.0)

        return prof_before_sh, prof_after_sh, prof_removed_sh, total_before, total_after, total_removed

    shared_result = build_shared_profiles(primary_col, _colormap(primary_col))
    if shared_result is None:
        return

    sh_before, sh_after, sh_removed, sh_sum_before, sh_sum_after, sh_sum_removed = shared_result

    shared_title_before  = f"{Before} · Shared Σ {sh_sum_before:.0f} m³/ha"
    shared_title_after   = f"{After} · Shared Σ {sh_sum_after:.0f} m³/ha"
    shared_title_removed = f"{Removed} · Shared Σ {sh_sum_removed:.0f} m³/ha"

    def max_shared_volume():
        return max(
            sh_before["volume"].max(),
            sh_after["volume"].max(),
            sh_removed["volume"].max()
        )

    x_upper_sh = _nice_upper(max_shared_volume())

    fig_sh = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        subplot_titles=(shared_title_before, shared_title_after, shared_title_removed),
        horizontal_spacing=0.06
    )

    if "plot_title_font" in st.session_state:
        fig_sh.update_layout(
            annotations=[
                dict(text=ann.text, x=ann.x, y=ann.y, xref=ann.xref, yref=ann.yref,
                     showarrow=False, font=st.session_state.plot_title_font)
                for ann in fig_sh.layout.annotations
            ]
        )

    # profily sdíleného objemu – opět podle primary_col
    add_panel(fig_sh, sh_before,  col=1, show_legend=True,  group_col=primary_col)
    add_panel(fig_sh, sh_after,   col=2, show_legend=False, group_col=primary_col)
    add_panel(fig_sh, sh_removed, col=3, show_legend=False, group_col=primary_col)

    fig_sh.update_layout(
        height=CHART_HEIGHT,
        margin=dict(l=10, r=10, t=60, b=60),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        hovermode="y unified"
    )
    for c in (1, 2, 3):
        fig_sh.update_xaxes(
            title_text="Shared crown volume [m³]", row=1, col=c,
            rangemode="tozero", range=[0, x_upper_sh]
        )

    fig_sh.update_yaxes(title_text="Height above ground [m]", row=1, col=1,
                        rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig_sh.update_yaxes(row=1, col=2, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)
    fig_sh.update_yaxes(row=1, col=3, rangemode="tozero", range=[0, float(H)], tick0=0, dtick=5)

    st.plotly_chart(fig_sh, use_container_width=True)


render_crown_volume_profiles(df, primary=primary, overlay=overlay)
