import json
from pathlib import Path
import pandas as pd

# --- vstup/výstup ---
IN_JSON  = Path("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/settings.json")
SPECIES_TSV   = Path("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/colors.txt")      # tab-delimited, s hlavičkou
MGMT_TSV    = Path("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/mgmt_colors.txt") 
OUT_JSON = Path("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/settings_up.json")

# ---------- POMOCNÉ ----------
def hex_to_rgb01(hex_str: str):
    """'#RRGGBB' -> [r,g,b] v rozsahu 0–1 (zaokrouhleno)"""
    if not isinstance(hex_str, str):
        return None
    s = hex_str.strip()
    if not s:
        return None
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Neplatná HEX barva: {hex_str!r}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return [round(r, 6), round(g, 6), round(b, 6)]

# ---------- NAČTENÍ JSON ----------
with IN_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

species_list = data.get("tree_species", [])
mgmt_list    = data.get("managementStatus", [])

if not isinstance(species_list, list):
    raise TypeError("V JSONu není očekávané pole 'tree_species' jako list.")
if not isinstance(mgmt_list, list):
    raise TypeError("V JSONu není očekávané pole 'managementStatus' jako list.")

# ---------- UPDATE DRUHŮ (species TSV) ----------
df_sp = pd.read_csv(SPECIES_TSV, sep="\t", dtype=str).fillna("")
df_sp.columns = [c.strip().lower() for c in df_sp.columns]

req_sp = {"latin", "abbreviation"}
missing = req_sp - set(df_sp.columns)
if missing:
    raise ValueError(f"V {SPECIES_TSV.name} chybí sloupce: {missing}. "
                     f"(Povinné: Latin, abbreviation; volitelný: color)")

color_col_present = "color" in df_sp.columns

# index v JSONu: case-insensitive dle latin
idx_by_latin = {(rec.get("latin") or "").strip().casefold(): i
                for i, rec in enumerate(species_list)}

sp_updated, sp_color_updates, sp_not_found = 0, 0, []
for _, row in df_sp.iterrows():
    latin = row["latin"].strip()
    if not latin:
        continue
    key = latin.casefold()
    i = idx_by_latin.get(key)
    if i is None:
        sp_not_found.append(latin)
        continue

    rec = species_list[i]
    # abbreviation — přidej/aktualizuj
    rec["abbreviation"] = row["abbreviation"].strip()

    # barva — přepiš pouze pokud je v TSV neprázdná
    if color_col_present:
        hexcol = row["color"].strip()
        if hexcol:
            rec["color"] = hex_to_rgb01(hexcol)
            sp_color_updates += 1

    sp_updated += 1

# ---------- UPDATE MANAGEMENT (management TSV) ----------
df_m = pd.read_csv(MGMT_TSV, sep="\t", dtype=str).fillna("")
df_m.columns = [c.strip().lower() for c in df_m.columns]

req_m = {"category", "color"}
missing_m = req_m - set(df_m.columns)
if missing_m:
    raise ValueError(f"V {MGMT_TSV.name} chybí sloupce: {missing_m}. (Povinné: Category, color)")

# index v JSONu: case-insensitive dle label
idx_by_label = {(rec.get("label") or "").strip().casefold(): i
                for i, rec in enumerate(mgmt_list)}

mgmt_updated, mgmt_not_found = 0, []
for _, row in df_m.iterrows():
    label = row["category"].strip()
    hexcol = row["color"].strip()
    if not label:
        continue

    key = label.casefold()
    i = idx_by_label.get(key)
    if i is None:
        mgmt_not_found.append(label)
        continue

    rec = mgmt_list[i]
    if hexcol:
        rec["color"] = hex_to_rgb01(hexcol)
        mgmt_updated += 1

# ---------- ULOŽENÍ ----------
with OUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)