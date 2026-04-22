from datetime import datetime
import streamlit as st
import pandas as pd
import tempfile, os
import webbrowser

import src.io_utils as iou
from src.i18n import t, t_help
from src.db_utils import (
    get_sqlite_path_from_session,
    ensure_mgmt_tables,
    list_managements,
    load_management_map,
    save_management_from_trees,
    delete_management,
)


from src.report_utils import generate_all_summary_figs, build_intervention_report_pdf, SummaryVariant

# ---------- PROJECT SWITCHER ----------
# Dropdown + Load tlačítko úplně nahoře na první stránce.
# Po kliknutí na Load dojde ke kompletnímu vyčištění session_state
# (trees, mgmt_example, plot_info, color_palette, user_attributes, …)
# a nový JSON projekt je načten na dalším rerunu z app.py.
_available_projects = st.session_state.get("available_projects", []) or []
_current_project = st.session_state.get("project_file", "")

with st.container(border=True):
    st.markdown(f"##### {t('project_selector_header')}")

    if not _available_projects:
        st.info(t("project_no_files_found"))
    else:
        # Mapování: zobrazený název (basename) -> absolutní cesta.
        # Pokud by se v /data někdy objevily dva soubory se stejným jménem
        # (jiné adresáře), bylo by potřeba label zkombinovat s cestou –
        # tady ale všechny soubory ležící ve /data mají unikátní název.
        _proj_options: dict[str, str] = {os.path.basename(p): p for p in _available_projects}
        _labels = list(_proj_options.keys())

        _current_label = os.path.basename(_current_project) if _current_project else _labels[0]
        _default_idx = _labels.index(_current_label) if _current_label in _labels else 0

        ps_col_a, ps_col_b = st.columns([4, 1], vertical_alignment="bottom")
        with ps_col_a:
            _selected_label = st.selectbox(
                f"**{t('project_selector_label')}**",
                _labels,
                index=_default_idx,
                key="project_selectbox",
                help=t("project_selector_help"),
            )
            st.caption(f"{t('project_current')} `{os.path.basename(_current_project)}`")
        with ps_col_b:
            if st.button(
                t("project_load_btn"),
                icon=":material/refresh:",
                use_container_width=True,
                key="btn_load_project",
                type="primary",
            ):
                _new_path = _proj_options[_selected_label]
                if _new_path != _current_project:
                    # kompletní reload – smaž data-derived session state
                    iou.reset_project_state()
                    st.session_state.project_file = _new_path
                    st.session_state.flash_success = t("project_loaded_success").format(
                        name=_selected_label
                    )
                    st.rerun()

# ---------- DATA ----------
plot_info = st.session_state.plot_info
df: pd.DataFrame = st.session_state.trees.copy()

# ---------- DB PATH ----------
sqlite_path = get_sqlite_path_from_session(st.session_state)
ensure_mgmt_tables(sqlite_path)

# ---------- helper: user cache ----------
def _ensure_usr_cache():
    """
    Uloží aktuální user management_status do usr_mgmt_cache, pokud ještě není.
    Cache je "poslední uživatelský zásah" nezávislý na přepínání scénářů.
    """
    if "usr_mgmt_cache" not in st.session_state or st.session_state["usr_mgmt_cache"] is None:
        tmp = st.session_state.trees[["id", "management_status"]].copy()
        tmp["id"] = pd.to_numeric(tmp["id"], errors="coerce").astype("Int64")
        tmp = tmp.dropna(subset=["id"]).copy()
        tmp["id"] = tmp["id"].astype(int)
        st.session_state["usr_mgmt_cache"] = dict(zip(tmp["id"].tolist(), tmp["management_status"].tolist()))

def _restore_usr_cache():
    _ensure_usr_cache()
    tr = st.session_state.trees.copy()
    tr["id"] = pd.to_numeric(tr["id"], errors="coerce").astype("Int64")
    tr = tr.dropna(subset=["id"]).copy()
    tr["id"] = tr["id"].astype(int)

    cache = st.session_state.get("usr_mgmt_cache", {}) or {}
    # apply only where cache has value (and not None)
    tr["management_status"] = tr["id"].map(cache).fillna(tr["management_status"])
    # refresh colors
    palette = st.session_state.get("color_palette", {})
    tr = iou.refresh_management_colors(tr, palette)
    st.session_state.trees = tr

def _apply_saved_mgmt(mgmt_id: int):
    # before switching away, preserve current user intervention (once)
    _ensure_usr_cache()

    mp = load_management_map(sqlite_path, mgmt_id=int(mgmt_id))
    tr = st.session_state.trees.copy()

    tr["id"] = pd.to_numeric(tr["id"], errors="coerce").astype("Int64")
    tr = tr.dropna(subset=["id"]).copy()
    tr["id"] = tr["id"].astype(int)

    # apply scenario values only where present
    s = tr["id"].map(mp)
    mask = s.notna()
    tr.loc[mask, "management_status"] = s.loc[mask].astype(object)

    palette = st.session_state.get("color_palette", {})
    tr = iou.refresh_management_colors(tr, palette)
    st.session_state.trees = tr


# ---------- OVLÁDÁNÍ ----------
c1, c2 = st.columns([2, 2])



with st.expander(label=t("expander_help_label"),icon=":material/help:"):
    st.markdown(t_help("dashboard_help"))