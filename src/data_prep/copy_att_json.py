import json
import math
import re
from typing import Any, Dict, List, Tuple

# =========================
# 1) SEM ZADEJ CESTY
# =========================
SRC_PATH = r"C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json"        # zdroj (pokojna hora test_v4)
DST_PATH = r"C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora_tests.json"     # cíl (pokojnahora_test)
OUT_PATH = r"C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora_tests_merged.json"

# =========================
# 2) VYBER ATRIBUTY K PŘEKOPÍROVÁNÍ
# (jsou to klíče uvnitř segment["treeAttributes"])
# =========================
ATTRIBUTES_TO_COPY = [
    "management_status",
    "managementStatusId",
    "planar_projection_poly",
    "projection_exposure",
    "volume",
    "surface_area",
    "surfaceAreaProjection",
    "basal_area_m2",
    "heightXdbh",
    "crownCenter",
    "crownStartHeight",
    "dbh",
    "height",
]

# =========================
# 3) HELPER FUNKCE
# =========================

_NAN_INF_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9_])(-?Infinity|NaN)(?![A-Za-z0-9_])')

def load_json_lenient(path: str) -> Any:
    """
    Načte JSON i když obsahuje NaN/Infinity nebo je 'lehce rozbitý' těmito tokeny.
    1) zkusí json.loads (Python toleruje NaN/Infinity),
    2) když to spadne, nahradí NaN/Infinity za null a zkusí znovu.
    """
    raw = open(path, "r", encoding="utf-8").read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raw2 = _NAN_INF_TOKEN_RE.sub("null", raw)
        return json.loads(raw2)

def replace_non_finite_numbers(obj: Any) -> Any:
    """Rekurzivně nahradí float NaN/±inf za None, aby šel vypsat VALID JSON (allow_nan=False)."""
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
    """Index: label -> celý segment ze zdroje."""
    idx: Dict[str, Dict[str, Any]] = {}
    for seg in (source.get("segments") or []):
        label = seg.get("label")
        if isinstance(label, str) and label.strip():
            idx[label] = seg
    return idx

def copy_selected_tree_attributes(
    source: Dict[str, Any],
    target: Dict[str, Any],
    attrs_to_copy: List[str],
) -> Tuple[int, int, List[str]]:
    """
    Zkopíruje vybrané klíče z source.segments[*].treeAttributes do target.segments[*].treeAttributes
    podle shody segment['label'].

    Vrací: (matched_count, updated_count, warnings)
    """
    warnings: List[str] = []
    src_by_label = build_source_index_by_label(source)

    matched = 0
    updated = 0

    for tseg in (target.get("segments") or []):
        tlabel = tseg.get("label")
        if not isinstance(tlabel, str):
            continue

        sseg = src_by_label.get(tlabel)
        if not sseg:
            continue

        matched += 1

        sta = sseg.get("treeAttributes") or {}
        tta = tseg.get("treeAttributes") or {}

        if not isinstance(sta, dict) or not isinstance(tta, dict):
            warnings.append(f"Segment '{tlabel}': treeAttributes chybí nebo není dict.")
            continue

        changed = False
        for key in attrs_to_copy:
            if key in sta:
                tta[key] = sta[key]
                changed = True

        if changed:
            tseg["treeAttributes"] = tta
            updated += 1

    return matched, updated, warnings

# =========================
# 4) HLAVNÍ BĚH
# =========================

source = load_json_lenient(SRC_PATH)
target = load_json_lenient(DST_PATH)

if not isinstance(source, dict) or not isinstance(target, dict):
    raise ValueError("Kořen obou JSONů musí být objekt (dict).")

matched, updated, warnings = copy_selected_tree_attributes(source, target, ATTRIBUTES_TO_COPY)

# Validní JSON output (žádné NaN/Infinity)
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

# Volitelně: rychlá kontrola atributů u prvních pár segmentů
# for seg in (target_clean.get("segments") or [])[:5]:
#     print(seg.get("label"), {k: seg.get("treeAttributes", {}).get(k) for k in ATTRIBUTES_TO_COPY})
