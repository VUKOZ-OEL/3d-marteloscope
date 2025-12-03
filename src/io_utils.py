import json
import pandas as pd
from typing import Dict, List, Union, Any
import streamlit as st
from pathlib import Path
import sqlite3
import html
import os
import numpy as np
import re
import math

__all__ = [
    "load_project_json",
    "save_project_json",
    "load_color_pallete",
    "load_plot_info",
    "load_simulation_results",
    "heading_centered",
    "_unique_sorted",
    "show_success",
]


# --- Pomocné funkce (samostatné a robustní) ----------------------------------


def simplify_polygon(points, min_dist=0.5):
    """Redukuje body polygonu podle minimální vzdálenosti."""
    if len(points) < 3:
        return points

    simplified = [points[0]]
    last_x, last_y = points[0]

    for x, y in points[1:]:
        if math.dist((x, y), (last_x, last_y)) >= min_dist:
            simplified.append((x, y))
            last_x, last_y = x, y

    # uzavřít polygon
    if simplified[0] != simplified[-1]:
        simplified.append(simplified[0])

    return simplified

def ply_to_wkt_polygon(ply_path: str, min_dist=0.5) -> str:
    """
    Načte PLY, extrahuje XY body, zjednoduší polygon a vrátí WKT POLYGON.
    min_dist = minimální rozestup mezi body (v metrech).
    """
    if not os.path.exists(ply_path):
        return None

    xs = []
    ys = []
    header = True

    try:
        with open(ply_path, "r") as f:
            for line in f:
                if header:
                    if line.strip().lower() == "end_header":
                        header = False
                    continue

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                try:
                    xs.append(float(parts[0]))
                    ys.append(float(parts[1]))
                except:
                    continue
    except:
        return None

    if len(xs) < 3:
        return None

    # spojit XY body do listu
    points = list(zip(xs, ys))

    # --- SIMPLIFY POLYGON ---
    points = simplify_polygon(points, min_dist=min_dist)

    # WKT
    coords = ", ".join(f"{x} {y}" for x, y in points)
    return f"POLYGON (({coords}))"


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
            return [max(0.0, min(1.0, r)), max(0.0, min(1.0, g)), max(0.0, min(1.0, b))]
        else:
            return [
                max(0.0, min(1.0, r / 255.0)),
                max(0.0, min(1.0, g / 255.0)),
                max(0.0, min(1.0, b / 255.0)),
            ]
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
        unsafe_allow_html=True,
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


def load_project_json(file_path: str, exclude_from_sql_update: List[str] = None) -> pd.DataFrame:
    """
    Načte JSON, zpracuje atributy a barvy, a synchronizuje s SQLite.
    Hodnoty z JSONu (species, barvy) mají přednost a nejsou přepisovány z SQL.
    """
    if exclude_from_sql_update is None:
        exclude_from_sql_update = []

    # Tyto sloupce jsou "svaté" - pochází z logiky JSONu a SQL je nesmí nikdy přepsat.
    protected_cols = [
        "species", 
        "speciesColorHex", 
        "management_status", 
        "managementColorHex",
        "label" # Label často chceme taky zachovat z JSONu, pokud to není v zadání jinak
    ]
    
    # Rozšíříme seznam výjimek
    exclude_final = list(set(exclude_from_sql_update + protected_cols))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Soubor {file_path} nebyl nalezen.")

    # --- 1. Načtení a parsování JSON ---
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Lookupy s vynuceným int klíčem
    species_list = data.get("species", [])
    sp_id_map = {}
    for item in species_list:
        try: sp_id_map[int(item.get("id"))] = {"latin": item.get("latin", "Unknown"), "color": _to_hex(item.get("color"))}
        except: pass

    mgmt_list = data.get("managementStatus", [])
    mg_id_map = {}
    for item in mgmt_list:
        try: mg_id_map[int(item.get("id"))] = {"label": item.get("label", "Unknown"), "color": _to_hex(item.get("color"))}
        except: pass

    # Zpracování segmentů
    segments = data.get("segments", [])
    rows = []
    for seg in segments:
        base = {}
        base["id"] = seg.get("id")
        base["label"] = seg.get("label", "")

        # IDs
        try: s_id = int(seg.get("speciesId", -1))
        except: s_id = -1
        try: m_id = int(seg.get("managementStatusId", -1))
        except: m_id = -1

        # Data z lookupů
        sp = sp_id_map.get(s_id, {})
        mg = mg_id_map.get(m_id, {})

        base["species"] = sp.get("latin", "Unknown")
        base["speciesColorHex"] = sp.get("color") # Tady musí být HEX nebo None
        
        base["management_status"] = mg.get("label", "Unknown")
        base["managementColorHex"] = mg.get("color")

        SKIP_ATTRS = {"speciesColorHex", "managementColorHex", "species", "management_status"}
        # Atributy
        attrs = seg.get("treeAttributes", {}) or {}
        print(attrs.items())
        for k, v in attrs.items():
            if k == "position":
                continue
            if k in SKIP_ATTRS:
                continue    # ← TADY ZABRÁNÍME PŘEPISU
            base[k] = v

        # Pozice
        pos = attrs.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            base["x"], base["y"] = float(pos[0]), float(pos[1])
            if len(pos) > 2: base["z"] = float(pos[2])
        else:
            base["x"], base["y"] = 0.0, 0.0
        rows.append(base)

    print("ROWS:")
    #print(rows)
    # Vytvoření DataFrame
    df_json = pd.DataFrame(rows)
    
    if df_json.empty:
        return df_json

    # Nastavení indexu a typů
    df_json['id'] = df_json['id'].astype(int)
    df_json.set_index('id', inplace=True, drop=False)

    # Filtrace (Ground / Unsegmented)
    if "label" in df_json.columns:
        mask = df_json["label"].astype(str).str.contains("ground|unsegmented", case=False, na=False)
        df_json = df_json.loc[~mask]

    # --- 2. SQLite Logika ---
    sqlite_path = os.path.splitext(file_path)[0] + ".sqlite"
    table_name = "tree"

    # A) Pokud neexistuje -> Vytvořit
    if not os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            # Uložíme jen základní sloupce
            cols_to_save = [c for c in ["id", "label", "species"] if c in df_json.columns]
            df_json[cols_to_save].to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"SQLite vytvořeno: {sqlite_path}")
            conn.close()
        except Exception as e:
            print(f"Chyba při tvorbě SQLite: {e}")
    
    # B) Pokud existuje -> Merge (Update)
    else:
        try:
            conn = sqlite3.connect(sqlite_path)
            # Ověření tabulky
            if pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", conn).empty:
                conn.close()
                return df_json.reset_index(drop=True)

            # Načtení SQL dat
            df_sql = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()

            if not df_sql.empty:
                # Sjednotit index
                df_sql['id'] = df_sql['id'].astype(int)
                df_sql.set_index('id', inplace=True, drop=False)

                # 1. Identifikace sloupců pro update
                # Vezmeme sloupce z SQL, ale VYŘADÍME ty, které jsou v exclude_final (barvy, species, atd.)
                # Tím zajistíme, že i kdyby v SQL sloupec 'speciesColorHex' byl (a byl prázdný), nepoužije se.
                update_cols = [c for c in df_sql.columns if c in df_json.columns and c not in exclude_final]
                
                if update_cols:
                    # Update přepíše hodnoty v df_json hodnotami z df_sql (tam kde se shoduje ID)
                    df_json.update(df_sql[update_cols])

                # 2. Identifikace nových sloupců (které jsou v SQL, ale ne v JSON)
                # Např. uživatelské poznámky přidané v aplikaci
                new_cols = [c for c in df_sql.columns if c not in df_json.columns]
                
                if new_cols:
                    df_json = df_json.join(df_sql[new_cols])

        except Exception as e:
            print(f"Chyba při merge SQLite: {e}")

    return df_json.reset_index(drop=True)


def load_plot_info(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    pi = data.get("plot_info") or []

    return pd.DataFrame(pi)


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
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,)
        )
        if cur.fetchone() is None:
            raise ValueError(f"Table '{table}' not found in {db_path}")

        # load via pandas
        df = pd.read_sql_query(f"SELECT * FROM {table};", con)

    st.session_state.simulation = df
    return df




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
        unsafe_allow_html=True,
    )


def save_project_json(
    original_path: str, df: pd.DataFrame, output_path: str = None
) -> None:
    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # mapování řádek podle ID
    id_map = df.set_index("id").to_dict("index")

    for segment in data.get("segments", []):
        sid = segment.get("id")
        if sid in id_map and "treeAttributes" in segment:
            row = id_map[sid]
            attr = segment["treeAttributes"]
            print("a")
            if "speciesId" in row:
                segment["speciesId"] = int(row["speciesId"])
                print(segment["speciesId"])
                print(row["speciesId"])

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

            if "managementColorHex" in row and isinstance(
                row["managementColorHex"], str
            ):
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


def load_project_json(file_path: str, exclude_from_sql_update: List[str] = None) -> pd.DataFrame:
    """
    Načte projektový JSON 3DForest, vytvoří DataFrame stromů, sloučí ho se SQLite databází
    a doplní chybějící 2D projekce koruny (PLY → WKT) do SQLite přes UPDATE.
    """

    import os, json, sqlite3
    import pandas as pd

    # --------------------------------------------------------------------------------------
    # 0) Sloupce, které SQL nesmí nikdy přepsat
    # --------------------------------------------------------------------------------------
    if exclude_from_sql_update is None:
        exclude_from_sql_update = []

    protected_cols = [
        "species",
        "speciesColorHex",
        "management_status",
        "managementColorHex",
        "label",
    ]

    exclude_final = list(set(exclude_from_sql_update + protected_cols))


    # --------------------------------------------------------------------------------------
    # 1) Načtení JSON souboru
    # --------------------------------------------------------------------------------------
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    # --------------------------------------------------------------------------------------
    # 2) Lookup tabulky species + management
    # --------------------------------------------------------------------------------------
    sp_id_map = {}
    for item in data.get("species", []):
        try:
            sid = int(item.get("id"))
            sp_id_map[sid] = {
                "latin": item.get("latin", "Unknown"),
                "color": _to_hex(item.get("color")),
            }
        except:
            pass

    mg_id_map = {}
    for item in data.get("managementStatus", []):
        try:
            mid = int(item.get("id"))
            mg_id_map[mid] = {
                "label": item.get("label", "Unknown"),
                "color": _to_hex(item.get("color")),
            }
        except:
            pass


    # --------------------------------------------------------------------------------------
    # 3) Segmenty → DataFrame
    # --------------------------------------------------------------------------------------
    rows = []

    for seg in data.get("segments", []):
        base = {}

        base["id"] = seg.get("id")
        base["label"] = seg.get("label", "")

        # Lookup IDs
        try: s_id = int(seg.get("speciesId"))
        except: s_id = -1
        try: m_id = int(seg.get("managementStatusId"))
        except: m_id = -1

        # Lookup data
        sp = sp_id_map.get(s_id, {})
        mg = mg_id_map.get(m_id, {})

        base["species"] = sp.get("latin", "Unknown")
        base["speciesColorHex"] = sp.get("color")
        base["management_status"] = mg.get("label", "Unknown")
        base["managementColorHex"] = mg.get("color")

        # Atributy stromu
        SKIP_ATTRS = {"speciesColorHex", "managementColorHex", "species", "management_status"}

        attrs = seg.get("treeAttributes", {}) or {}
        for k, v in attrs.items():
            if k == "position":
                continue
            if k in SKIP_ATTRS:
                continue
            base[k] = v

        # Pozice
        pos = attrs.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            base["x"] = float(pos[0])
            base["y"] = float(pos[1])
            if len(pos) > 2:
                base["z"] = float(pos[2])
        else:
            base["x"], base["y"] = 0.0, 0.0

        rows.append(base)

    df_json = pd.DataFrame(rows)

    if df_json.empty:
        return df_json

    df_json["id"] = df_json["id"].astype(int)
    df_json.set_index("id", inplace=True, drop=False)

    # Odstranit ground/unsegmented
    if "label" in df_json.columns:
        df_json = df_json[
            ~df_json["label"].astype(str).str.contains("ground|unsegmented", case=False, na=False)
        ]


    # --------------------------------------------------------------------------------------
    # 4) SQLite MERGE
    # --------------------------------------------------------------------------------------
    sqlite_path = os.path.splitext(file_path)[0] + ".sqlite"
    table_name = "tree"

    # A) Pokud SQLite neexistuje → vytvořit základní tabulku
    if not os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            df_json[["id", "label", "species"]].to_sql(
                table_name, conn, if_exists="replace", index=False
            )
            conn.close()
        except Exception as e:
            print("Chyba při tvorbě SQLite:", e)

    # B) SQLite existuje → merge
    if os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)

            exists = not pd.read_sql(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';",
                conn
            ).empty

            if exists:
                df_sql = pd.read_sql(f"SELECT * FROM {table_name}", conn)

                if not df_sql.empty:
                    df_sql["id"] = df_sql["id"].astype(int)
                    df_sql.set_index("id", inplace=True, drop=False)

                    # Sloupce, které smíme přepsat
                    update_cols = [
                        c for c in df_sql.columns
                        if c in df_json.columns and c not in exclude_final
                    ]

                    if update_cols:
                        df_json.update(df_sql[update_cols])

                    # Nové sloupce z SQL → přidat
                    new_cols = [c for c in df_sql.columns if c not in df_json.columns]
                    if new_cols:
                        df_json = df_json.join(df_sql[new_cols])

            conn.close()

        except Exception as e:
            print("Chyba merge SQLite:", e)


    # --------------------------------------------------------------------------------------
    # 5) POLYGONY (PLY → WKT → SQLite UPDATE)
    # --------------------------------------------------------------------------------------
    project_dir = os.path.dirname(file_path)
    project_name = os.path.splitext(os.path.basename(file_path))[0]

    # Sloupec v DF
    if "planar_projection_poly" not in df_json.columns:
        df_json["planar_projection_poly"] = None

    # Načíst existující polygony ze SQL, pokud jsou
    try:
        conn = sqlite3.connect(sqlite_path)
        sql_cols = pd.read_sql(f"PRAGMA table_info({table_name});", conn)["name"].tolist()

        if "planar_projection_poly" in sql_cols:
            df_poly = pd.read_sql(
                f"SELECT id, planar_projection_poly FROM {table_name}", conn
            )
            df_poly["id"] = df_poly["id"].astype(int)
            df_poly.set_index("id", inplace=True, drop=False)
            df_json.update(df_poly[["planar_projection_poly"]])

        conn.close()
    except:
        pass

    # Najít chybějící WKT polygony
    missing = df_json[df_json["planar_projection_poly"].isna()]

    if not missing.empty:
        # dopočítat WKT z PLY
        for tid in missing["id"]:
            ply_name = f"{project_name}.{tid}.concaveHullProjection.ply"
            ply_path = os.path.join(project_dir, ply_name)

            wkt = ply_to_wkt_polygon(ply_path)
            if wkt:
                df_json.loc[tid, "planar_projection_poly"] = wkt

        # uložit do SQL pomocí UPDATE
        try:
            conn = sqlite3.connect(sqlite_path)

            # přidat sloupec, pokud neexistuje
            sql_cols = pd.read_sql(f"PRAGMA table_info({table_name});", conn)["name"].tolist()
            if "planar_projection_poly" not in sql_cols:
                conn.execute(f"ALTER TABLE {table_name} ADD COLUMN planar_projection_poly TEXT;")
                conn.commit()

            # UPDATE řádek po řádku
            for tid, row in df_json.iterrows():
                poly = row.get("planar_projection_poly")
                if poly and isinstance(poly, str):
                    conn.execute(
                        f"UPDATE {table_name} SET planar_projection_poly = ? WHERE id = ?",
                        (poly, int(tid))
                    )

            conn.commit()
            conn.close()

        except Exception as e:
            print("Chyba ukládání polygonů do SQLite:", e)


    # --------------------------------------------------------------------------------------
    return df_json.reset_index(drop=True)
