import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import src.io_utils as iou

# ---------- LABELY Z SESSION STATE (CO VIDÍ UŽIVATEL) ----------
label_before = st.session_state.Before  # např. "Before"
label_after = st.session_state.After  # např. "After"
label_removed = st.session_state.Removed  # např. "Removed"

colorBySpp_label = st.session_state.Species  # např. "Species"
colorByMgmt_label = st.session_state.Management  # např. "Management"

# ---------- INTERNÍ KLÍČE (STÁLÉ, NEZÁVISLÉ NA TEXTECH) ----------
FILTER_BEFORE = "before"
FILTER_AFTER = "after"
FILTER_REMOVED = "removed"

COLOR_SPP = "color_species"
COLOR_MGMT = "color_mgmt"

# ---------- LOAD DATA ----------
if "trees" not in st.session_state:
    file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
    st.session_state.trees = iou.load_project_json(file_path)

df_all = st.session_state.trees.copy()
# ---------- NORMALIZACE SOUŘADNIC 0 → MAX ----------
if "x" in df_all.columns and "y" in df_all.columns:
    df_all["x"] = df_all["x"] - df_all["x"].min()
    df_all["y"] = df_all["y"] - df_all["y"].min()

# Standardizace typů
for col in [
    "species",
    "speciesColorHex",
    "management_status",
    "managementColorHex",
    "label",
]:
    if col in df_all.columns:
        df_all[col] = df_all[col].astype(str)

# DBH a výška na čísla
if "dbh" in df_all.columns:
    df_all["dbh"] = pd.to_numeric(df_all["dbh"], errors="coerce")
if "height" in df_all.columns:
    df_all["height"] = pd.to_numeric(df_all["height"], errors="coerce")


# ---------- MASK LOGIC (stejné chování jako SUMMARY, ale s interními klíči) ----------
def _make_masks(d: pd.DataFrame):
    keep_status = {"Target tree", "Untouched"}
    mask_after = d.get("management_status", pd.Series(False, index=d.index)).isin(
        keep_status
    )
    mask_removed = (
        ~mask_after
        if "management_status" in d.columns
        else pd.Series(False, index=d.index)
    )
    mask_before = pd.Series(True, index=d.index)  # vše
    return {
        FILTER_BEFORE: mask_before,
        FILTER_AFTER: mask_after,
        FILTER_REMOVED: mask_removed,
    }


masks = _make_masks(df_all)

# ---------- UI TOP ----------
c1, c2, c3, c4, c5 = st.columns([1, 3, 1, 3, 1])

with c2:
    dist_mode = st.segmented_control(
        "**Show Data for:**",
        options=[FILTER_BEFORE, FILTER_AFTER, FILTER_REMOVED],
        default=FILTER_BEFORE,
        format_func=lambda v: {
            FILTER_BEFORE: label_before,
            FILTER_AFTER: label_after,
            FILTER_REMOVED: label_removed,
        }.get(v, v),
        width="stretch",
    )

with c4:
    color_mode = st.segmented_control(
        "**Color by**",
        options=[COLOR_SPP, COLOR_MGMT],
        default=COLOR_SPP,
        format_func=lambda v: {
            COLOR_SPP: colorBySpp_label,
            COLOR_MGMT: colorByMgmt_label,
        }.get(v, v),
        width="stretch",
    )

# ---------- APPLY Show Data For FILTER ----------
mask = masks.get(dist_mode, pd.Series(True, index=df_all.index))
df = df_all[mask].copy()

# ---------- LAYOUT ----------
c21, _, c23, _ = st.columns([2, 0.5, 10, 0.5])

with c21:
    st.markdown("##")

    # DBH filter (range z CELÉHO datasetu)
    if "dbh" in df_all.columns:
        dbh_vals_full = pd.to_numeric(df_all["dbh"], errors="coerce").dropna()
        if dbh_vals_full.empty:
            min_dbh, max_dbh = 0, 100
        else:
            min_dbh, max_dbh = int(dbh_vals_full.min()), int(dbh_vals_full.max())
    else:
        dbh_vals_full = pd.Series([], dtype=float)
        min_dbh, max_dbh = 0, 100

    dbh_range = st.slider(
        "**DBH filter [cm]**", min_dbh, max_dbh, (min_dbh, max_dbh), 1
    )

    # Height filter (range z CELÉHO datasetu)
    if "height" in df_all.columns:
        h_vals_full = pd.to_numeric(df_all["height"], errors="coerce").dropna()
        if h_vals_full.empty:
            min_h, max_h = 0, 50
        else:
            min_h, max_h = int(h_vals_full.min()), int(h_vals_full.max())
        height_range = st.slider(
            "**Height filter [m]**", min_h, max_h, (min_h, max_h), 1
        )
    else:
        height_range = None

    st.divider()

    show_crown = st.checkbox("**Show Crown Projections**", value=False)
    show_text = st.checkbox("**Show Label**", value=False)

    st.divider()

    # Size sliders
    size_min = st.session_state.get("size_min", 6)
    size_max = st.session_state.get("size_max", 28)

    new_size_min = st.slider("**Min Point Size (px)**", 1, 20, size_min, 1)
    new_size_max = st.slider("**Max Point Size (px)**", 20, 60, size_max, 1)

# ---------- MAP ----------
with c23:
    # Bezpečná numerika pro DBH / HEIGHT
    if "dbh" in df.columns:
        df["dbh"] = pd.to_numeric(df["dbh"], errors="coerce")
    if "height" in df.columns:
        df["height"] = pd.to_numeric(df["height"], errors="coerce")

    # --- APPLY DBH FILTER ---
    if "dbh" in df.columns:
        df = df[(df["dbh"] >= dbh_range[0]) & (df["dbh"] <= dbh_range[1])]

    # --- APPLY HEIGHT FILTER ---
    if height_range is not None and "height" in df.columns:
        df = df[(df["height"] >= height_range[0]) & (df["height"] <= height_range[1])]

    # Pokud je df prázdné -> info
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        st.info("Pro zadané filtry nejsou k dispozici žádná data k zobrazení.")
    else:
        # ---------- COLORS ----------
        if color_mode == COLOR_SPP:
            color_col = "speciesColorHex"
            group_col = "species"
        else:
            color_col = "managementColorHex"
            group_col = "management_status"

        if color_col not in df.columns:
            colors = pd.Series(["#AAAAAA"] * len(df), index=df.index)
        else:
            colors = df[color_col].apply(
                lambda c: c if isinstance(c, str) and c.startswith("#") else "#AAAAAA"
            )

        # ---------- SIZES based on FULL DBH range (stable) ----------
        if not dbh_vals_full.empty:
            global_dbh_min = dbh_vals_full.min()
            global_dbh_max = dbh_vals_full.max()
        else:
            global_dbh_min = 1
            global_dbh_max = 1

        if "dbh" in df.columns:
            dbh_series = pd.to_numeric(df["dbh"], errors="coerce").fillna(1)
        else:
            dbh_series = pd.Series([1] * len(df), index=df.index)

        if global_dbh_min == global_dbh_max:
            sizes = np.full(len(df), (new_size_min + new_size_max) / 2)
        else:
            sizes = np.interp(
                dbh_series,
                (global_dbh_min, global_dbh_max),
                (new_size_min, new_size_max),
            )

        # ---------- CUSTOMDATA ----------
        # label, dbh, species?, management?
        cd_parts = []
        if "label" in df.columns:
            cd_parts.append(df["label"].astype(str))
        else:
            cd_parts.append(pd.Series([""] * len(df), index=df.index))

        cd_parts.append(dbh_series.astype(float))

        if "species" in df.columns:
            cd_parts.append(df["species"].astype(str))
        if "management_status" in df.columns:
            cd_parts.append(df["management_status"].astype(str))

        customdata = np.column_stack(cd_parts)

        # ---------- HOVER TEMPLATE ----------
        hover_lines = [
            "label: %{customdata[0]}",
            "DBH: %{customdata[1]} cm",
        ]
        idx = 2
        if "species" in df.columns:
            hover_lines.append(f"Species: %{{customdata[{idx}]}}")
            idx += 1
        if "management_status" in df.columns:
            hover_lines.append(f"Management: %{{customdata[{idx}]}}")

        hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

        # ---------- FIGURE ----------
        fig = go.Figure()

        # ---------- GRID based on FULL dataset ----------
        x_max = float(df_all["x"].max())
        y_max = float(df_all["y"].max())

        grid_x = np.arange(0, x_max + 10, 10)
        grid_y = np.arange(0, y_max + 10, 10)

        for gx in grid_x:
            fig.add_trace(
                go.Scatter(
                    x=[gx, gx],
                    y=[0, y_max],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        for gy in grid_y:
            fig.add_trace(
                go.Scatter(
                    x=[0, x_max],
                    y=[gy, gy],
                    mode="lines",
                    line=dict(color="lightgray", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # ---------- POINTS ----------
        scatter_cls = go.Scatter if show_text else go.Scattergl
        mode = "markers+text" if show_text else "markers"

        if group_col in df.columns:
            group_vals = df[group_col].astype(str)
        else:
            group_vals = colors  # fallback – seskupení podle barvy

        from collections import defaultdict

        buckets = defaultdict(list)

        for i, (gv, col) in enumerate(zip(group_vals, colors)):
            buckets[(str(gv), col)].append(i)

        print(df)
        for (legend_name, hex_color), idxs in buckets.items():
            fig.add_trace(
                scatter_cls(
                    x=df["x"].iloc[idxs],
                    y=df["y"].iloc[idxs],
                    mode=mode,
                    name=legend_name,
                    text=df["label"].iloc[idxs]
                    if show_text and "label" in df.columns
                    else None,
                    textposition="top center",
                    marker=dict(size=sizes[idxs], color=hex_color, sizemode="diameter"),
                    customdata=customdata[idxs],
                    hovertemplate=hovertemplate,
                )
            )

        # ---------- AXES ----------
        fig.update_xaxes(showgrid=False, range=[0, x_max])
        fig.update_yaxes(
            showgrid=False, range=[0, y_max], scaleanchor="x", scaleratio=1
        )

        fig.update_layout(
            height=700,
            dragmode="pan",
            margin=dict(l=10, r=10, t=10, b=10),
        )

        st.plotly_chart(fig, width="stretch")

# ---------- SAVE SIZES ----------
st.session_state.size_min = new_size_min
st.session_state.size_max = new_size_max
