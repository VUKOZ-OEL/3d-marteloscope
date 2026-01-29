import streamlit as st
import pandas as pd
import numpy as np
import src.io_utils as iou
from src.i18n import t

trees_df = st.session_state.trees.copy()

# --- HLAVIČKA / BUTTONS ---
c1, c2, c3 = st.columns([3, 1, 2])

with c1:
    st.markdown(f"##### {t('page_add_attributes_title')}:")

    uploaded_file = st.file_uploader(
        label="",
        type="csv",
        help=t("file_uploader_details"),  # bez příkladu formátu
    )

with c3:
    st.button(
        f"**{t('btn_save_project')}**",
        type="primary",
        icon=":material/save:",
        use_container_width=True,
    )
    st.button(
        f"**{t('btn_save_project_as')}**",
        type="secondary",
        icon=":material/save_as:",
        use_container_width=True,
    )
    do_import = st.button(
        t("btn_import_attributes"),
        type="secondary",
        icon=":material/add_column_right:",
        use_container_width=True,
    )

# === ZPRACOVÁNÍ PO NAHRÁNÍ ===
if uploaded_file is not None:
    import_df = pd.read_csv(uploaded_file)

    # 1) Urči ID sloupec v importu (preferuj 'ID', jinak první sloupec)
    import_id_col = "ID" if "ID" in import_df.columns else import_df.columns[0]
    if import_id_col != "ID":
        import_df = import_df.rename(columns={import_id_col: "ID"})
    import_df["ID"] = import_df["ID"].astype(str).str.strip()
    import_df = import_df.drop_duplicates(subset=["ID"], keep="last")

    # 2) Urči ID v projektových datech (preferuj 'ID', jinak první sloupec)
    trees_id_col = "ID" if "ID" in trees_df.columns else trees_df.columns[0]
    trees_df["_ID_JOIN"] = trees_df[trees_id_col].astype(str).str.strip()

    # 3) Zjisti chybějící ID (v trees jsou, v CSV chybí)
    missing_ids = sorted(set(trees_df["_ID_JOIN"]) - set(import_df["ID"]))

    if missing_ids:
        st.error(t("error_missing_ids", count=len(missing_ids)))
        st.code(", ".join(missing_ids[:50]) + (" ..." if len(missing_ids) > 50 else ""))

    # 4) Rozdělení sloupců: duplicitní (přepíšou se), nové (přidají se)
    dup_cols = [c for c in import_df.columns if c != "ID" and c in trees_df.columns]
    new_cols = [c for c in import_df.columns if c != "ID" and c not in trees_df.columns]
    all_cols = dup_cols + new_cols

    if not all_cols:
        st.info(t("info_no_columns_to_import"))
    else:
        if dup_cols:
            st.warning(t("warn_overwrite_columns") + " " + ", ".join(dup_cols))

        # 5) Sloučení pro preview
        merged = trees_df.merge(
            import_df[["ID"] + all_cols],
            left_on="_ID_JOIN",
            right_on="ID",
            how="left",
            suffixes=("", "_import"),
        )

        # DŮLEŽITÉ: ID ve výstupu je VŽDY z projektových dat
        preview_df = pd.DataFrame({"ID": trees_df["_ID_JOIN"].values})
        for c in all_cols:
            preview_df[c] = merged[c].values

        # NaN -> <NA>
        preview_df = preview_df.convert_dtypes()
        preview_df = preview_df.where(pd.notna(preview_df), pd.NA)

        # --- Stylování (pandas Styler) ---
        missing_set = set(map(str, missing_ids))
        ROW_MISSING_BG = "background-color: rgba(255,0,0,0.10)"   # missing ID
        COL_DUP_BG     = "background-color: rgba(255,215,0,0.25)" # dup sloupce

        def highlight_rows(row: pd.Series):
            _id = row.get("ID", "")
            miss = pd.isna(_id) or str(_id) in missing_set or str(_id) == ""
            return [ROW_MISSING_BG if miss else ""] * len(row)

        def highlight_cols(col: pd.Series):
            name = col.name
            if name in dup_cols:
                return [COL_DUP_BG] * len(col)
            return [""] * len(col)

        base_styles = [
            dict(selector="th", props=[("max-width", "140px"), ("overflow", "hidden")]),
            dict(
                selector="td",
                props=[
                    ("max-width", "160px"),
                    ("overflow", "hidden"),
                    ("text-overflow", "ellipsis"),
                    ("white-space", "nowrap"),
                ],
            ),
            dict(selector="table", props=[("table-layout", "fixed")]),
        ]

        small_styler = (
            preview_df.head(3)
            .style
            .apply(highlight_rows, axis=1)
            .apply(highlight_cols, axis=0)
            .set_table_styles(base_styles)
        )

        full_styler = (
            preview_df
            .style
            .apply(highlight_rows, axis=1)
            .apply(highlight_cols, axis=0)
            .set_table_styles(base_styles)
        )

        # 6) „Minimalizovaný“ náhled
        st.markdown(f"##### {t('preview_title')}")
        st.dataframe(
            small_styler,
            use_container_width=False,
            height=160,
        )

        # 7) Plný náhled v expanderu
        with st.expander(t("preview_full")):
            st.dataframe(
                full_styler,
                use_container_width=True,
                height=600,
            )

        # 8) Samotný import po kliknutí
        if do_import:
            id_series = trees_df[trees_id_col].astype(str).str.strip()
            for col in all_cols:
                mapper = dict(zip(import_df["ID"], import_df[col]))
                trees_df[col] = id_series.map(mapper)

            st.session_state.trees = trees_df.drop(columns=["_ID_JOIN"], errors="ignore")
            st.success(
                t(
                    "success_import",
                    count=len(all_cols),
                    cols=", ".join(all_cols),
                )
            )

with st.expander(label=t("expander_help_label"),icon=":material/help:"):
    st.markdown(t("add_att_help"))