import streamlit as st
import pandas as pd
import src.io_utils as iou

from src.i18n import t, set_lang, get_lang


# ---------- DATA ----------
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()


# ---------- OVLÁDÁNÍ ----------
c1, c2 = st.columns([2, 2])

with c1:
    # ---------- HLAVIČKA ----------
    st.markdown(f"### :orange[**{plot_info['name'].iloc[0]}**]")
    st.markdown("###")

    st.markdown(f"##### {t('overview_header')}")
    st.markdown("")

    st.markdown(
        f"""
- **{t('forest_type')}** {plot_info['forest_type'].iloc[0]}
- **{t('number_of_trees_label')}** {plot_info['no_trees'].iloc[0]}
- **{t('wood_volume_label')}** {plot_info['volume'].iloc[0]} m³
- **{t('area')}** {plot_info['size_ha'].iloc[0]} ha
- **{t('altitude')}** {plot_info['altitude'].iloc[0]} m
- **{t('precipitation')}** {plot_info['precipitation'].iloc[0]} mm/year
- **{t('average_temperature')}** {plot_info['temperature'].iloc[0]} °C
- **{t('established')}** {plot_info['established'].iloc[0]}
- **{t('location')}** {plot_info['state'].iloc[0]}
- **{t('owner')}** {plot_info['owner'].iloc[0]}
- **{t('scan_date')}** {plot_info['scan_date'].iloc[0]}
"""
    )


with c2:
    st.markdown(f"##### {t('choose_existing_selection')}")
    st.markdown("")

    uploaded_file = st.file_uploader(
        f"**{t('import_ext_selection')}**",
        type="csv",
        help=t("import_help_text"),
    )

    st.divider()

    # --- options: map label -> key (key = název sloupce v mgmt_example) ---
    mgmt_options = {
        t("usr_mgmt"): "usr_mgmt",
        t("ph_mgmt_ex_1"): "ph_mgmt_ex_1",
        # později přidáš další: t("ph_mgmt_ex_2"): "ph_mgmt_ex_2", ...
    }

    # aby selectbox ukazoval aktuální výběr
    inv = {v: k for k, v in mgmt_options.items()}
    current_label = inv.get(st.session_state.active_mgmt_selection, t("usr_mgmt"))
    labels = list(mgmt_options.keys())
    default_index = labels.index(current_label) if current_label in labels else 0

    msg = st.session_state.pop("flash_success", None)
    if msg:
        st.success(msg)

    selected_label = st.selectbox(
        f"**{t('management_examples')}**",
        labels,
        index=default_index,
    )
    selected_key = mgmt_options[selected_label]

    if st.button(
        f"**{t('load_example')}**",
        icon=":material/model_training:",
        use_container_width=True,
    ):
        iou.apply_mgmt_selection(selected_key)  # <- tady proběhne bezpečné přepnutí
        # uložit hlášku pro další rerun
        st.session_state.flash_success = t("success_load_mgmt")
        st.session_state.flash_success_ts = pd.Timestamp.utcnow().isoformat()
        st.rerun()
        

    st.divider()

    st.markdown(f"##### {t('project_controls')}")

    st.button(t("export_results"), icon=":material/file_save:")
    st.button(t("btn_clear_management"), icon=":material/delete:")



