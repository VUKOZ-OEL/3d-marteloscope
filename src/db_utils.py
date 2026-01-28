from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from typing import Dict

import pandas as pd


# ------------------------------------------------------------
# DB helpers (uprav cestu/conn podle projektu)
# ------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    return sqlite3.connect("app.sqlite", check_same_thread=False)

def ensure_mgmt_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mgmt_streams (
        plot_id     TEXT NOT NULL,
        stream_key  TEXT NOT NULL,
        label       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        PRIMARY KEY (plot_id, stream_key)
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS mgmt_stream_values (
        plot_id     TEXT NOT NULL,
        stream_key  TEXT NOT NULL,
        tree_id     TEXT NOT NULL,
        value       REAL,
        PRIMARY KEY (plot_id, stream_key, tree_id)
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_msv_plot_key ON mgmt_stream_values(plot_id, stream_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_msv_plot_tree ON mgmt_stream_values(plot_id, tree_id)")
    conn.commit()


# ------------------------------------------------------------
# Existing (from earlier): list & save
# ------------------------------------------------------------

def get_saved_mgmt_options(plot_id: str) -> Dict[str, str]:
    conn = get_conn()
    ensure_mgmt_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT label, stream_key FROM mgmt_streams WHERE plot_id = ? ORDER BY created_at DESC",
        (plot_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return {label: key for (label, key) in rows}

def slugify_key(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "selection"

def make_unique_key(conn: sqlite3.Connection, plot_id: str, base_key: str) -> str:
    cur = conn.cursor()
    key = base_key
    i = 1
    while True:
        cur.execute(
            "SELECT 1 FROM mgmt_streams WHERE plot_id = ? AND stream_key = ? LIMIT 1",
            (plot_id, key),
        )
        if cur.fetchone() is None:
            return key
        i += 1
        key = f"{base_key}_{i}"

def save_current_user_selection_into_db(
    plot_id: str,
    label: str,
    trees_df: pd.DataFrame,
    active_selection_key: str,
    tree_id_col: str = "tree_id",
) -> str:
    if not label.strip():
        raise ValueError("label must be non-empty")
    if tree_id_col not in trees_df.columns:
        raise KeyError(f"trees_df missing '{tree_id_col}' column")
    if active_selection_key not in trees_df.columns:
        raise KeyError(f"trees_df missing active selection column '{active_selection_key}'")

    conn = get_conn()
    ensure_mgmt_schema(conn)

    base_key = "usr_" + slugify_key(label)
    stream_key = make_unique_key(conn, plot_id, base_key)

    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO mgmt_streams (plot_id, stream_key, label, created_at) VALUES (?, ?, ?, ?)",
        (plot_id, stream_key, label.strip(), created_at),
    )

    rows = []
    for _, r in trees_df[[tree_id_col, active_selection_key]].iterrows():
        v = r[active_selection_key]
        rows.append((plot_id, stream_key, str(r[tree_id_col]), None if pd.isna(v) else float(v)))

    cur.executemany(
        "INSERT OR REPLACE INTO mgmt_stream_values (plot_id, stream_key, tree_id, value) VALUES (?, ?, ?, ?)",
        rows,
    )

    conn.commit()
    conn.close()
    return stream_key


# ------------------------------------------------------------
# NEW: detect/load/delete DB stream
# ------------------------------------------------------------

def is_db_stream(plot_id: str, stream_key: str) -> bool:
    conn = get_conn()
    ensure_mgmt_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM mgmt_streams WHERE plot_id = ? AND stream_key = ? LIMIT 1",
        (plot_id, stream_key),
    )
    ok = cur.fetchone() is not None
    conn.close()
    return ok

def load_db_stream_into_trees(
    plot_id: str,
    stream_key: str,
    trees_df: pd.DataFrame,
    tree_id_col: str = "tree_id",
) -> pd.DataFrame:
    """
    Načte DB stream a přidá ho do trees_df jako sloupec stream_key (virtuální sloupec).
    """
    if tree_id_col not in trees_df.columns:
        raise KeyError(f"trees_df missing '{tree_id_col}' column")

    conn = get_conn()
    ensure_mgmt_schema(conn)

    q = """
    SELECT tree_id, value
    FROM mgmt_stream_values
    WHERE plot_id = ? AND stream_key = ?
    """
    stream_df = pd.read_sql_query(q, conn, params=(plot_id, stream_key))
    conn.close()

    # map values by tree_id
    mapper = stream_df.set_index("tree_id")["value"] if not stream_df.empty else pd.Series(dtype=float)
    out = trees_df.copy()
    out[stream_key] = out[tree_id_col].astype(str).map(mapper)

    return out

def delete_mgmt_stream(plot_id: str, stream_key: str) -> None:
    """
    Smaže stream z katalogu i všechny hodnoty.
    """
    conn = get_conn()
    ensure_mgmt_schema(conn)
    cur = conn.cursor()
    cur.execute("DELETE FROM mgmt_stream_values WHERE plot_id = ? AND stream_key = ?", (plot_id, stream_key))
    cur.execute("DELETE FROM mgmt_streams WHERE plot_id = ? AND stream_key = ?", (plot_id, stream_key))
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# UPDATED: apply_mgmt_selection – now supports DB streams too
# ------------------------------------------------------------

def apply_mgmt_selection(selected_key: str, plot_id: str) -> None:
    """
    Přepne aktivní management selection.
    - Pokud je selected_key base sloupec, použije existující logiku.
    - Pokud je selected_key DB stream, načte ho do session_state.trees jako sloupec a nastaví aktivní.
    """
    import streamlit as st

    # --- DB stream? ---
    if is_db_stream(plot_id, selected_key):
        st.session_state.trees = load_db_stream_into_trees(
            plot_id=plot_id,
            stream_key=selected_key,
            trees_df=st.session_state.trees,
            tree_id_col="tree_id",  # uprav pokud máš jinak
        )
        st.session_state.active_mgmt_selection = selected_key
        return

    # --- BASE selection (tvoje původní logika) ---
    # TODO: nech tady svůj existující kód, který přepíná usr_mgmt / ph_mgmt_ex_1 ...
    st.session_state.active_mgmt_selection = selected_key
    # (pokud původně děláš i další kroky, ponech je)
