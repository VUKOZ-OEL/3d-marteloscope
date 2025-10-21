import json
import pandas as pd
import numpy as np
import unicodedata
from typing import Dict, List, Union, Any, Optional
import streamlit as st
from pathlib import Path
import sqlite3
import html

__all__ = ["load_project_json","save_project_json","load_color_pallete","load_plot_info","load_simulation_results", "heading_centered","_unique_sorted","show_success"]



# --- Pomocné funkce (samostatné a robustní) ----------------------------------

def norm(s: str) -> str:
    return (s or "").strip().lower()

def _is_hex_color(x: Any) -> bool:
    return isinstance(x, str) and len(x) == 7 and x.startswith("#")

def _rgb_to_hex01(c01: List[float]) -> str:
    """c01 = [r,g,b] v rozsahu 0..1 → '#RRGGBB'"""
    r = max(0, min(255, int(round(c01[0] * 255))))
    g = max(0, min(255, int(round(c01[1] * 255))))
    b = max(0, min(255, int(round(c01[2] * 255))))
    return f"#{r:02X}{g:02X}{b:02X}"

def _to_color01(value: Any) -> List[float] | None:
    """
    Přijme '#RRGGBB' nebo [r,g,b] v 0..1 či 0..255 a vrátí [r,g,b] v 0..1.
    """
    if value is None:
        return None

    # hex řetězec
    if _is_hex_color(value):
        hs = value[1:]
        try:
            r = int(hs[0:2], 16) / 255.0
            g = int(hs[2:4], 16) / 255.0
            b = int(hs[4:6], 16) / 255.0
            return [r, g, b]
        except Exception:
            return None

    # seznam/tuple čísel
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            r, g, b = float(value[0]), float(value[1]), float(value[2])
        except Exception:
            return None
        # 0..1 nebo 0..255?
        if max(r, g, b) <= 1.0:
            return [max(0.0, min(1.0, r)),
                    max(0.0, min(1.0, g)),
                    max(0.0, min(1.0, b))]
        else:
            return [max(0.0, min(1.0, r / 255.0)),
                    max(0.0, min(1.0, g / 255.0)),
                    max(0.0, min(1.0, b / 255.0))]
    return None

def _to_hex(value: Any) -> str | None:
    """Vrátí '#RRGGBB' nebo None."""
    if _is_hex_color(value):
        return value.upper()
    c01 = _to_color01(value)
    return _rgb_to_hex01(c01) if c01 else None

def _unique_sorted(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())

def heading_centered(text: str, color: str = "#2E7D32", level: int = 5):
    """
    Render a centered heading in Streamlit with custom color and level (h1–h6).

    Parameters
    ----------
    text : str
        Heading text.
    color : str, optional
        CSS color (e.g., '#2E7D32' or 'darkgreen'). Default '#2E7D32'.
    level : int, optional
        Heading level 1–6 (maps to <h1>..</h6>). Default 5.
    """
    # clamp level to 1..6
    try:
        lvl = int(level)
    except Exception:
        lvl = 5
    lvl = min(max(lvl, 1), 6)

    safe_text = html.escape(str(text))
    st.markdown(
        f"<h{lvl} style='text-align:center; color:{color}; margin:0;'>{safe_text}</h{lvl}>",
        unsafe_allow_html=True
    )

# --- Hlavní funkce ------------------------------------------------------------
def load_color_pallete(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    # --- 1) Lookupy pro barvy ---
    # species_colors: {latin_name: color}
    species_colors = data.get("species_colors") or []
    sp_map: Dict[str, str] = {}  # norm(latin) -> '#RRGGBB'
    if isinstance(species_colors, list):
        for item in species_colors:
            if not isinstance(item, dict):
                continue
            lat = item.get("latin")
            col = item.get("color")
            hexc = _to_hex(col)
            if lat and hexc:
                sp_map[norm(str(lat))] = hexc

    return sp_map

def load_project_json(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    # --- 1) Lookupy pro barvy ---
    # species_colors: {latin_name: color}
    species_colors = data.get("species_colors") or []
    sp_map: Dict[str, str] = {}  # norm(latin) -> '#RRGGBB'
    if isinstance(species_colors, list):
        for item in species_colors:
            if not isinstance(item, dict):
                continue
            lat = item.get("latin")
            col = item.get("color")
            hexc = _to_hex(col)
            if lat and hexc:
                sp_map[norm(str(lat))] = hexc

    # managementStatus: [{label: "...", color: ...}, ...]
    mgmt_list = data.get("managementStatus") or []
    mg_map: Dict[str, str] = {}  # norm(label) -> '#RRGGBB'
    if isinstance(mgmt_list, list):
        for item in mgmt_list:
            if not isinstance(item, dict):
                continue
            lab = item.get("label")
            col = item.get("color")
            hexc = _to_hex(col)
            if lab and hexc:
                mg_map[norm(str(lab))] = hexc

    # --- 2) Segmenty -> řádky ---
    segments: List[Dict[str, Any]] = data.get("segments", []) or []
    rows: List[Dict[str, Any]] = []

    for seg in segments:
        base: Dict[str, Any] = {}
        base["id"] = seg.get("id")
        base["label"] = seg.get("label", "")

        attrs: Dict[str, Any] = seg.get("treeAttributes", {}) or {}
        # zkopíruj vše z treeAttributes (aby se neztratilo nic, co tam máš)
        for k, v in attrs.items():
            base[k] = v

        # pozice -> x,y
        pos = attrs.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            try:
                base["x"], base["y"] = float(pos[0]), float(pos[1])
            except Exception:
                base.setdefault("x", 0.0)
                base.setdefault("y", 0.0)
        else:
            base.setdefault("x", 0.0)
            base.setdefault("y", 0.0)

        # --- 3) Barvy pro species a management podle požadovaných polí ---
        sp_key = norm(str(attrs.get("species", "")))
        mg_key = norm(str(attrs.get("management_status", "")))

        base["speciesColorHex"] = sp_map.get(sp_key)  # může být None, když nenašlo
        base["managementColorHex"] = mg_map.get(mg_key)

        rows.append(base)

    df = pd.DataFrame(rows)

    # --- 4) Odstranit ground (jako dřív) ---
    if "label" in df.columns:
        mask = df["label"].astype(str).str.contains("ground", case=False, na=False)
        df = df.loc[~mask].reset_index(drop=True)

    return df

def load_plot_info(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    pi = data.get("plot_info") or []

    return(pd.DataFrame(pi))

def load_simulation_results(db_path: str | Path, table: str = "tree") -> pd.DataFrame:
    """
    Load a table (default: 'tree') from a SQLite database into a pandas DataFrame.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as con:
        # check table exists
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        if cur.fetchone() is None:
            raise ValueError(f"Table '{table}' not found in {db_path}")

        # load via pandas
        df = pd.read_sql_query(f"SELECT * FROM {table};", con)

    st.session_state.simulation = df
    return df

"""
def save_project_json(original_path: str, df: pd.DataFrame, output_path: str = None) -> None:
    import json

    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_map = df.set_index("id").to_dict("index")

    for segment in data.get("segments", []):
        sid = segment.get("id")
        if sid in id_map and "treeAttributes" in segment:
            row = id_map[sid]
            attr = segment["treeAttributes"]

            z = 0.0
            if isinstance(attr.get("position"), list) and len(attr["position"]) >= 3:
                z = attr["position"][2]

            if "x" in row and "y" in row:
                attr["position"] = [
                    float(row["x"]) ,
                    float(row["y"]) ,
                    z
                ]

            # Vyloučíme klíče, které už existují i ve vnějším segmentu
            skip_keys = {"id"}
            outer_keys = set(segment.keys())  # např. "label" na úrovni segmentu
            for key, value in row.items():
                if key not in skip_keys and key not in outer_keys:
                    attr[key] = value  # bezpečné přepsání nebo přidání

    output_path = output_path or original_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
"""
def show_success(message: str, timeout: int = 2000):
    """
    Zobrazí úspěchovou zprávu na pár vteřin.
    
    Args:
        message: text zprávy
        timeout: doba zobrazení v ms (2000 = 2 vteřiny)
    """
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div id="tmp-success" style="background:#d4edda;
             color:#155724;
             border:1px solid #c3e6cb;
             padding:0.75rem 1rem;
             border-radius:0.25rem;
             margin:0.5rem 0;">
          ✅ {message}
        </div>
        <script>
        setTimeout(function(){{
            var el = document.getElementById("tmp-success");
            if(el) el.style.display = 'none';
        }}, {timeout});
        </script>
        """,
        unsafe_allow_html=True
    )

import pandas as pd

def save_project_json(original_path: str, df: pd.DataFrame, output_path: str = None) -> None:

    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # mapování řádek podle ID
    id_map = df.set_index("id").to_dict("index")

    for segment in data.get("segments", []):
        sid = segment.get("id")
        if sid in id_map and "treeAttributes" in segment:
            row = id_map[sid]
            attr = segment["treeAttributes"]

            # zachovat původní Z, ale aktualizovat X,Y pokud jsou v DF
            z = 0.0
            if isinstance(attr.get("position"), list) and len(attr["position"]) >= 3:
                z = attr["position"][2]

            if "x" in row and "y" in row:
                attr["position"] = [float(row["x"]), float(row["y"]), z]

            # --- převod HEX -> RGB (0..1) a zapsání do speciesColor/managementColor ---
            if "speciesColorHex" in row and isinstance(row["speciesColorHex"], str):
                rgb = _to_color01(row["speciesColorHex"])
                if rgb is not None:
                    attr["speciesColor"] = rgb

            if "managementColorHex" in row and isinstance(row["managementColorHex"], str):
                rgb = _to_color01(row["managementColorHex"])
                if rgb is not None:
                    attr["managementColor"] = rgb

            # --- kopírování dalších atributů do treeAttributes ---
            # vynecháme klíče, které nechceme vkládat (HEX, id) a ty, co existují ve vnějším segmentu
            skip_keys = {"id", "speciesColorHex", "managementColorHex"}
            outer_keys = set(segment.keys())
            for key, value in row.items():
                if key in skip_keys or key in outer_keys:
                    continue
                # speciesColor / managementColor už jsme nastavili výše z HEX – nepřepisovat z DF,
                # pokud by tam náhodou byly i floatové sloupce stejného jména
                if key in {"speciesColor", "managementColor"}:
                    continue
                attr[key] = value

    output_path = output_path or original_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
