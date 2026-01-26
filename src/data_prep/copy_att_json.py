import json
import math
import re
from typing import Any, Dict, List, Tuple, Optional

# =========================
# 1) SEM ZADEJ CESTY
# =========================
SRC_PATH = r"data/pokojna_test_v2_2.json"
DST_PATH = r"c:/users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/3df_sample/pokojna_hora.json"
OUT_PATH = r"c:/users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/3df_sample/pokojna_hora_v2.json"

# =========================
# 1b) NASTAVENI JEDNOTEK DBH
# =========================
# Příklad: z cm na m => 0.01
DBH_SCALE = 0.01          # násobek: new_dbh = old_dbh * DBH_SCALE
DBH_KEYS_TO_SCALE = {"dbh"}  # kdybys měl víc klíčů (např. "dbh_cm"), přidej sem

# =========================
# 2) VYBER KLÍČE K PŘEKOPÍROVÁNÍ
# =========================

# A) klíče na úrovni segmentu (segment["..."])
SEGMENT_KEYS = [
    "managementStatusId",
    "speciesId",
]

# B) klíče uvnitř segment["treeAttributes"]
TREEATTR_KEYS = [
    "crownCenter",
    "crownStartHeight",
    "crownVoxelCount",
    "crownVoxelCountPerMeters",
    "crownVoxelCountShared",
    "crownVoxelSize",
    "dbh",
    "dbhNormal",
    "dbhPosition",
    "height",
    "position",
    "surfaceArea",
    "surfaceAreaProjection",
    "volume",
    "x",
    "y",
    "z",
    "species",
    "management_status",
    "planar_projection_poly",
    "Volume_m3",
    "crown_centroid_height",
    "crown_volume",
    "crown_surface",
    "vertical_crown_proj",
    "horizontal_crown_proj",
    "crown_eccentricity",
    "projection_exposure",
    "projection_exposure_after_mgmt",
    "basal_area_m2",
    "heightXdbh",
    "light_avail",
    "light_comp",
    "crown_height",
]

# =========================
# 3) HELPER FUNKCE
# =========================

_NAN_INF_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(-?Infinity|NaN)(?![A-Za-z0-9_])")

def load_json_lenient(path: str) -> Any:
    raw = open(path, "r", encoding="utf-8").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = _NAN_INF_TOKEN_RE.sub("null", raw)
        return json.loads(raw2)

def replace_non_finite_numbers(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [replace_non_finite_numbers(x) for x in obj]
    if isinstance(obj, dict):
        return {k: replace_non_finite_numbers(v) for k, v in obj.items()}
    return obj

def build_source_index_by_label(source: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for seg in (source.get("segments") or []):
        label = seg.get("label")
        if isinstance(label, str) and label.strip():
            idx[label] = seg
    return idx

def coerce_number(value: Any) -> Optional[float]:
    """
    Pokusí se z value udělat float.
    - číslo -> float
    - string -> float (pokud jde)
    - None/ostatní -> None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None
    return None

def maybe_scale_dbh(key: str, value: Any, warnings: List[str], seg_label: str) -> Any:
    """
    Pokud key patří mezi DBH klíče, přepočítá hodnotu.
    Zachová None, a pokud nejde převést na číslo, nechá původní a zapíše warning.
    """
    if key not in DBH_KEYS_TO_SCALE:
        return value

    num = coerce_number(value)
    if num is None:
        # None/Nečíselné necháme
        if value not in (None,):
            warnings.append(f"Segment '{seg_label}': dbh='{value}' nelze převést na číslo, nechávám beze změny.")
        return value

    return num * DBH_SCALE

def copy_selected_segment_data(
    source: Dict[str, Any],
    target: Dict[str, Any],
    segment_keys_to_copy: List[str],
    treeattr_keys_to_copy: List[str],
) -> Tuple[int, int, List[str]]:
    warnings: List[str] = []
    src_by_label = build_source_index_by_label(source)

    matched = 0
    updated = 0

    for tseg in (target.get("segments") or []):
        tlabel = tseg.get("label")
        if not isinstance(tlabel, str) or not tlabel.strip():
            continue

        sseg = src_by_label.get(tlabel)
        if not sseg:
            continue

        matched += 1
        changed = False

        # --- A) kopírování klíčů na úrovni segmentu ---
        for key in segment_keys_to_copy:
            if key in sseg:
                tseg[key] = sseg[key]
                changed = True

        # --- B) kopírování uvnitř treeAttributes ---
        sta = sseg.get("treeAttributes") or {}
        tta = tseg.get("treeAttributes") or {}

        if not isinstance(sta, dict) or not isinstance(tta, dict):
            warnings.append(f"Segment '{tlabel}': treeAttributes chybí nebo není dict.")
        else:
            for key in treeattr_keys_to_copy:
                if key in sta:
                    val = sta[key]
                    # ✅ TADY se dělá přepočet dbh
                    val = maybe_scale_dbh(key, val, warnings, tlabel)
                    tta[key] = val
                    changed = True

            tseg["treeAttributes"] = tta

        if changed:
            updated += 1

    return matched, updated, warnings

# =========================
# 4) HLAVNÍ BĚH
# =========================

source = load_json_lenient(SRC_PATH)
target = load_json_lenient(DST_PATH)

if not isinstance(source, dict) or not isinstance(target, dict):
    raise ValueError("Kořen obou JSONů musí být objekt (dict).")

matched, updated, warnings = copy_selected_segment_data(
    source,
    target,
    SEGMENT_KEYS,
    TREEATTR_KEYS,
)

target_clean = replace_non_finite_numbers(target)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(target_clean, f, ensure_ascii=False, indent=2, allow_nan=False)

print(f"Hotovo: uložený soubor: {OUT_PATH}")
print(f"Spárováno podle label: {matched}")
print(f"Aktualizováno segmentů: {updated}")
if warnings:
    print("Varování:")
    for w in warnings:
        print(" -", w)
