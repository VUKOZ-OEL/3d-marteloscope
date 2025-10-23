import streamlit as st
import pandas as pd
import numpy as np
import src.io_utils as iou

trees_df = st.session_state.trees.copy()

# --- HLAVIČKA / BUTTONS ---
c1, c2, c3 = st.columns([3, 1, 2])
with c1:
    st.markdown("##### Add attributes to current project :")
    uploaded_file = st.file_uploader("", type="csv", help="""
                **File format:**
                ```
                ID, Attribute_1, Attribute_2, ...
                id, value, value, ...
                ..., ..., ..., ...
                ```
                The file must contain a header.  
                If the *ID* column is not present, the first column is considered as Tree IDs.  
                Use **.** as decimal separator.
                """)

  
with c3:
    st.button("**Save project**", type="primary", icon=":material/save:", use_container_width=True)
    st.button("**Save project as**", type="secondary", icon=":material/save_as:", use_container_width=True)
    do_import = st.button("Import attributes", type="secondary", icon=":material/add_column_right:", use_container_width=True)


# === ZPRACOVÁNÍ PO NAHRÁNÍ ===
if uploaded_file is not None:
    import_df = pd.read_csv(uploaded_file)

    # 1) Urči ID sloupec v importu (preferuj 'ID', jinak první sloupec)
    import_id_col = "ID" if "ID" in import_df.columns else import_df.columns[0]
    if import_id_col != "ID":
        import_df = import_df.rename(columns={import_id_col: "ID"})
    import_df["ID"] = import_df["ID"].astype(str).str.strip()
    import_df = import_df.drop_duplicates(subset=["ID"], keep="last")

    # 2) Urči ID v projetkových datech (preferuj 'ID', jinak první sloupec)
    trees_id_col = "ID" if "ID" in trees_df.columns else trees_df.columns[0]
    trees_df["_ID_JOIN"] = trees_df[trees_id_col].astype(str).str.strip()

    # 3) Zjisti chybějící ID (v trees jsou, v CSV chybí)
    missing_ids = sorted(set(trees_df["_ID_JOIN"]) - set(import_df["ID"]))

    if missing_ids:
        st.error(f"{len(missing_ids)} IDs missing in CSV (will show <NA>):")
        st.code(", ".join(missing_ids[:50]) + (" ..." if len(missing_ids) > 50 else ""))

    # 4) Rozdělení sloupců: duplicitní (přepíšou se), nové (přidají se)
    dup_cols = [c for c in import_df.columns if c != "ID" and c in trees_df.columns]
    new_cols = [c for c in import_df.columns if c != "ID" and c not in trees_df.columns]
    all_cols = dup_cols + new_cols

    if not all_cols:
        st.info("No new or overlapping columns to import.")
    else:
        if dup_cols:
            st.warning("Existing columns will be overwritten: " + ", ".join(dup_cols))

        # 5) Sloučení pro preview
        merged = trees_df.merge(
            import_df[["ID"] + all_cols],
            left_on="_ID_JOIN",
            right_on="ID",
            how="left",
            suffixes=("", "_import"),
        )

        # DŮLEŽITÉ: ID ve výstupu je VŽDY z projektových dat
        # (tj. jistota, že ID je vždy přítomno a koresponduje s pořadím řádků v projektu)
        preview_df = pd.DataFrame({"ID": trees_df["_ID_JOIN"].values})
        for c in all_cols:
            preview_df[c] = merged[c].values

        # NaN -> <NA> (konzistentní missing typ)
        preview_df = preview_df.convert_dtypes()
        preview_df = preview_df.where(pd.notna(preview_df), pd.NA)

        # --- Stylování (pandas Styler) ---
        missing_set = set(map(str, missing_ids))
        ROW_MISSING_BG = "background-color: rgba(255,0,0,0.10)"        # červené řádky (missing ID)
        COL_DUP_BG     = "background-color: rgba(255,215,0,0.25)"      # žluté sloupce (duplicitní)
        # POZN: nové sloupce NEMAJÍ žádné podbarvení

        def highlight_rows(row: pd.Series):
            _id = row.get("ID", "")
            miss = pd.isna(_id) or str(_id) in missing_set or str(_id) == ""
            return [ROW_MISSING_BG if miss else ""] * len(row)

        def highlight_cols(col: pd.Series):
            name = col.name
            if name in dup_cols:
                return [COL_DUP_BG] * len(col)
            # nové sloupce nebarvíme
            return [""] * len(col)

        # CSS pro omezení šířky sloupců a tabulky (min/max width)
        # (funguje pro Styler -> HTML rendering)
        base_styles = [
            dict(selector="th", props=[("max-width", "140px"), ("overflow", "hidden")]),
            dict(selector="td", props=[("max-width", "160px"), ("overflow", "hidden"), ("text-overflow", "ellipsis"), ("white-space", "nowrap")]),
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

        # 6) „Minimalizovaný“ náhled (jen 3 řádky, omezené šířky)
        st.markdown("##### Preview of attributes to be imported (first 3 rows)")
        st.dataframe(
            small_styler,
            use_container_width=False,  # drž menší šířku, aby to působilo kompaktně
            height=160,
        )

        # 7) Plný náhled v expanderu + možnost fullscreen (zajistí streamlit)
        with st.expander("Show full preview"):
            st.dataframe(
                full_styler,
                use_container_width=True,
                height=600,
            )

        # 8) Samotný import po kliknutí
        if do_import:
            # mapuj hodnoty z import_df přes ID; chybějící ID -> <NA> zůstane
            id_series = trees_df[trees_id_col].astype(str).str.strip()
            for col in all_cols:
                mapper = dict(zip(import_df["ID"], import_df[col]))
                trees_df[col] = id_series.map(mapper)

            # ulož zpět (odstraníme pomocný join sloupec)
            st.session_state.trees = trees_df.drop(columns=["_ID_JOIN"], errors="ignore")
            st.success(f"Imported/updated {len(all_cols)} column(s): {', '.join(all_cols)}")
