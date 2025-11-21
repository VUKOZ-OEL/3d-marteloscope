# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import src.io_utils as iou
from shapely import wkt

# =========================================================
# PARSE WKT → coordinates + area
# =========================================================
def parse_polygon(wkt_str):
    """Parse WKT polygon into (x,y,area). Supports Polygon & MultiPolygon."""
    try:
        geom = wkt.loads(wkt_str)
    except Exception:
        return None, None, None

    # Merge MultiPolygon → Polygon (union)
    if geom.geom_type == "MultiPolygon":
        geom = geom.buffer(0)

    if geom.geom_type != "Polygon":
        return None, None, None

    area = float(geom.area)
    x, y = geom.exterior.xy
    return list(x), list(y), area


# =========================================================
# HEX → RGBA
# =========================================================
def hex_to_rgba(hex_color, alpha=0.3):
    if not isinstance(hex_color, str):
        return f"rgba(120,120,120,{alpha})"
    hex_color = hex_color.strip()
    if hex_color.startswith("#"):
        hex_color = hex_color[1:]
    if len(hex_color) != 6:
        return f"rgba(120,120,120,{alpha})"

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# =========================================================
# LABELY → Z SESSION STATE
# =========================================================
label_before  = st.session_state.Before
label_after   = st.session_state.After
label_removed = st.session_state.Removed
colorBySpp_label  = st.session_state.Species
colorByMgmt_label = st.session_state.Management

FILTER_BEFORE  = "before"
FILTER_AFTER   = "after"
FILTER_REMOVED = "removed"

COLOR_SPP  = "color_species"
COLOR_MGMT = "color_mgmt"


# =========================================================
# LOAD DATA
# =========================================================
if "trees" not in st.session_state:
    file_path = (
        "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/"
        "PokojnaHora_3df/PokojnaHora.json"
    )
    st.session_state.trees = iou.load_project_json(file_path)

df_all = st.session_state.trees.copy()

# ---- store original minima BEFORE normalization ----
original_x_min = df_all["x"].min()
original_y_min = df_all["y"].min()

# ---- normalize tree coordinates ----
df_all["x"] = df_all["x"] - original_x_min
df_all["y"] = df_all["y"] - original_y_min

for col in ["species", "speciesColorHex", "management_status",
            "managementColorHex", "label"]:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str)

df_all["dbh"]    = pd.to_numeric(df_all["dbh"], errors="coerce")
df_all["height"] = pd.to_numeric(df_all["height"], errors="coerce")


# =========================================================
# MASKS
# =========================================================
def _make_masks(d):
    keep = {"Target tree", "Untouched"}
    ms = d["management_status"].astype(str)
    mask_after   = ms.isin(keep)
    mask_removed = ~mask_after
    mask_before  = pd.Series(True, index=d.index)
    return {
        FILTER_BEFORE:  mask_before,
        FILTER_AFTER:   mask_after,
        FILTER_REMOVED: mask_removed,
    }

masks = _make_masks(df_all)


# =========================================================
# TOP UI
# =========================================================
c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])

with c2:
    dist_mode = st.segmented_control(
        "**Show Data for:**",
        [FILTER_BEFORE, FILTER_AFTER, FILTER_REMOVED],
        default=FILTER_BEFORE,
        format_func=lambda v: {
            FILTER_BEFORE:  label_before,
            FILTER_AFTER:   label_after,
            FILTER_REMOVED: label_removed,
        }.get(v, v),
        width="stretch",
    )

with c4:
    color_mode = st.segmented_control(
        "**Color by**",
        [COLOR_SPP, COLOR_MGMT],
        default=COLOR_SPP,
        format_func=lambda v: {
            COLOR_SPP:  colorBySpp_label,
            COLOR_MGMT: colorByMgmt_label,
        }.get(v, v),
        width="stretch",
    )


# =========================================================
# LEFT SIDE PANEL
# =========================================================
mask = masks[dist_mode]
df = df_all[mask].copy()

c21, _, c23, _ = st.columns([2, 0.5, 10, 0.5])

with c21:

    st.markdown("**Show:**")
    show_text     = st.checkbox("**Label**", value=False)
    show_polys    = st.checkbox("**Crown Projections**", value=False)
    invert_colors = st.checkbox("**Invert Crown Colors**", value=False)

    # DBH filter
    dbh_vals = df_all["dbh"].dropna()
    if dbh_vals.empty:
        min_dbh, max_dbh = 0, 100
    else:
        min_dbh, max_dbh = int(dbh_vals.min()), int(dbh_vals.max())

    dbh_range = st.slider(
        "**Filter DBH [cm]**", min_dbh, max_dbh, (min_dbh, max_dbh)
    )

    # Height filter
    h_vals = df_all["height"].dropna()
    if h_vals.empty:
        min_h, max_h = 0, 50
    else:
        min_h, max_h = int(h_vals.min()), int(h_vals.max())

    height_range = st.slider(
        "**Filter Height [m]**", min_h, max_h, (min_h, max_h)
    )

    size_min = st.slider("Scale Min Point Size", 1, 20,
                         st.session_state.get("size_min", 6))
    size_max = st.slider("Scale Max Point Size", 20, 60,
                         st.session_state.get("size_max", 28))


# =========================================================
# APPLY FILTERS TO DF
# =========================================================
with c23:
    df = df[
        (df["dbh"]   >= dbh_range[0]) & (df["dbh"]   <= dbh_range[1]) &
        (df["height"]>= height_range[0]) & (df["height"]<= height_range[1])
    ]

    if df.empty:
        st.info("No data for given filters.")
        st.stop()

    color_col = "speciesColorHex" if color_mode == COLOR_SPP else "managementColorHex"
    group_col = "species"         if color_mode == COLOR_SPP else "management_status"

    def _valid_hex(c):
        return c if isinstance(c, str) and c.startswith("#") else "#888888"

    colors = df[color_col].apply(_valid_hex)

    # sizes
    if dbh_vals.empty:
        global_min_dbh, global_max_dbh = 1, 1
    else:
        global_min_dbh = dbh_vals.min()
        global_max_dbh = dbh_vals.max()

    dbh_for_size = df["dbh"].fillna(global_min_dbh)

    if global_min_dbh == global_max_dbh:
        sizes = np.full(len(df), (size_min + size_max) / 2.0)
    else:
        sizes = np.interp(
            dbh_for_size,
            (global_min_dbh, global_max_dbh),
            (size_min, size_max),
        )

    # HOVER TREE INFO (oprava: použít Series.round(), ne built-in round)
    dbh_rounded = df["dbh"].round(0).astype("Int64")

    customdata = np.column_stack([
        df["label"].astype(str),
        dbh_rounded.to_numpy(),
        df["species"].astype(str),
        df["management_status"].astype(str),
    ])

    hovertemplate = (
        "Label: %{customdata[0]}<br>"
        "DBH: %{customdata[1]} cm<br>"
        "Species: %{customdata[2]}<br>"
        "Management: %{customdata[3]}<extra></extra>"
    )

    # =========================================================
    # FIGURE
    # =========================================================
    fig = go.Figure()

    # =========================================================
    # POLYGONS (CROWN PROJECTIONS)
    # =========================================================
    if show_polys and "planar_projection_poly" in df_all.columns:

        poly_df = df_all[masks[dist_mode]].dropna(subset=["planar_projection_poly"])

        poly_x_min_vals = []
        poly_x_max_vals = []
        poly_y_min_vals = []
        poly_y_max_vals = []

        # choose attribute for polygon colors
        if invert_colors:
            if color_mode == COLOR_SPP:
                poly_group_col = "management_status"
                cmap = poly_df.groupby("management_status")["managementColorHex"].first().to_dict()
            else:
                poly_group_col = "species"
                cmap = poly_df.groupby("species")["speciesColorHex"].first().to_dict()
        else:
            if color_mode == COLOR_SPP:
                poly_group_col = "species"
                cmap = poly_df.groupby("species")["speciesColorHex"].first().to_dict()
            else:
                poly_group_col = "management_status"
                cmap = poly_df.groupby("management_status")["managementColorHex"].first().to_dict()

        # ---- DRAW POLYGONS ----
        for _, row in poly_df.iterrows():
            wkt_str = row["planar_projection_poly"]
            xs, ys, area = parse_polygon(wkt_str)
            if xs is None:
                continue

            # POSITION FIX — use original minima
            xs = np.array(xs) - original_x_min
            ys = np.array(ys) - original_y_min

            # collect polygon extents
            poly_x_min_vals.append(xs.min())
            poly_x_max_vals.append(xs.max())
            poly_y_min_vals.append(ys.min())
            poly_y_max_vals.append(ys.max())


            key = row[poly_group_col]
            base_color = cmap.get(key, "#77AADD")
            rgba_color = hex_to_rgba(base_color, alpha=0.30)

            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                line=dict(width=0),
                fillcolor=rgba_color,
                hoverinfo="skip",
                showlegend=False,
            ))

        # ---- POLYGON LEGEND (ONLY WHEN INVERT ON) ----
        if invert_colors:
            added = set()
            for key, hex_color in cmap.items():
                if key in added:
                    continue
                added.add(key)

                rgba = hex_to_rgba(hex_color, alpha=0.8)

                fig.add_trace(go.Scatter(
                    x=[-9999], y=[-9999],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=rgba,
                        line=dict(width=0),
                    ),
                    name=f"Crown – {key}",
                    hoverinfo="skip",
                    showlegend=True,
                ))

    # =========================================================
    # GRID 10 m (extends into negative space)
    # =========================================================

    # EXTENTS from tree points
    tree_x_min = df_all["x"].min()
    tree_x_max = df_all["x"].max()
    tree_y_min = df_all["y"].min()
    tree_y_max = df_all["y"].max()

    # EXTENTS from polygons
    poly_x_min = min(poly_x_min_vals) if show_polys else tree_x_min
    poly_x_max = max(poly_x_max_vals) if show_polys else tree_x_max
    poly_y_min = min(poly_y_min_vals) if show_polys else tree_y_min
    poly_y_max = max(poly_y_max_vals) if show_polys else tree_y_max

    # Combine
    x_min_full = min(tree_x_min, poly_x_min)
    x_max_full = max(tree_x_max, poly_x_max)
    y_min_full = min(tree_y_min, poly_y_min)
    y_max_full = max(tree_y_max, poly_y_max)

    # Expand to nearest tens and add buffer
    gx_min = np.floor(x_min_full / 10.0) * 10 - 10
    gx_max = np.ceil (x_max_full / 10.0) * 10 + 10
    gy_min = np.floor(y_min_full / 10.0) * 10 - 10
    gy_max = np.ceil (y_max_full / 10.0) * 10 + 10

    grid_x = np.arange(gx_min, gx_max+1, 10)
    grid_y = np.arange(gy_min, gy_max+1, 10)

    # DRAW GRID
    for gx in grid_x:
        fig.add_trace(go.Scatter(
            x=[gx, gx],
            y=[gy_min, gy_max],
            mode="lines",
            line=dict(color="lightgray", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    for gy in grid_y:
        fig.add_trace(go.Scatter(
            x=[gx_min, gx_max],
            y=[gy, gy],
            mode="lines",
            line=dict(color="lightgray", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Update axes so full grid is visible
    fig.update_xaxes(range=[gx_min, gx_max], showgrid=False)
    fig.update_yaxes(range=[gy_min, gy_max], showgrid=False,
                 scaleanchor="x", scaleratio=1)


    # =========================================================
    # TREE POINTS
    # =========================================================
    scatter_cls = go.Scatter if show_text else go.Scattergl
    mode = "markers+text" if show_text else "markers"

    from collections import defaultdict
    buckets = defaultdict(list)
    for idx, (gv, col) in enumerate(zip(df[group_col], colors)):
        buckets[(str(gv), col)].append(idx)

    for (legend_name, hex_color), idxs in buckets.items():
        fig.add_trace(
            scatter_cls(
                x=df["x"].iloc[idxs],
                y=df["y"].iloc[idxs],
                mode=mode,
                name=legend_name,
                text=df["label"].iloc[idxs] if show_text else None,
                textposition="top center",
                marker=dict(
                    size=sizes[idxs],
                    color=hex_color,
                    line=dict(width=0),   # no outline
                ),
                customdata=customdata[idxs],
                hovertemplate=hovertemplate,
            )
        )

    fig.update_xaxes(range=[0, gx_max], showgrid=False)
    fig.update_yaxes(
        range=[0, gy_max],
        showgrid=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        height=700,
        dragmode="pan",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

# SAVE SIZE SETTINGS
st.session_state.size_min = size_min
st.session_state.size_max = size_max
