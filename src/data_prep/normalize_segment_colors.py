import json
import os
from typing import Any, Dict, List, Optional, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_rgb01(col: Any) -> Optional[List[float]]:
    """
    Přijme barvu jako [r,g,b] kde může být:
      - float 0..1 (tvůj případ)
      - int 0..255
    Vrátí vždy [r,g,b] float 0..1.
    """
    if not isinstance(col, (list, tuple)) or len(col) < 3:
        return None
    r, g, b = col[0], col[1], col[2]

    # 0..255 int
    if all(isinstance(v, int) for v in (r, g, b)) and max(r, g, b) > 1:
        return [clamp01(r / 255.0), clamp01(g / 255.0), clamp01(b / 255.0)]

    # 0..1 float
    try:
        return [clamp01(r), clamp01(g), clamp01(b)]
    except Exception:
        return None


def rgb01_to_hex(rgb: List[float]) -> str:
    r = int(round(clamp01(rgb[0]) * 255))
    g = int(round(clamp01(rgb[1]) * 255))
    b = int(round(clamp01(rgb[2]) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"


def build_palette_maps(data: Dict[str, Any]) -> Tuple[Dict[int, List[float]], Dict[int, str], Dict[int, List[float]], Dict[int, str]]:
    """
    Vrátí 4 mapy:
      species_rgb: {speciesId: [r,g,b] (0..1)}
      species_hex: {speciesId: "#RRGGBB"}
      mgmt_rgb:    {managementStatusId: [r,g,b]}
      mgmt_hex:    {managementStatusId: "#RRGGBB"}
    """
    species_rgb: Dict[int, List[float]] = {}
    species_hex: Dict[int, str] = {}
    mgmt_rgb: Dict[int, List[float]] = {}
    mgmt_hex: Dict[int, str] = {}

    for sp in data.get("species", []) or []:
        if not isinstance(sp, dict):
            continue
        try:
            sid = int(sp.get("id"))
        except Exception:
            continue
        col = normalize_rgb01(sp.get("color"))
        if col is None:
            continue
        species_rgb[sid] = col
        species_hex[sid] = rgb01_to_hex(col)

    for mg in data.get("managementStatus", []) or []:
        if not isinstance(mg, dict):
            continue
        try:
            mid = int(mg.get("id"))
        except Exception:
            continue
        col = normalize_rgb01(mg.get("color"))
        if col is None:
            continue
        mgmt_rgb[mid] = col
        mgmt_hex[mid] = rgb01_to_hex(col)

    return species_rgb, species_hex, mgmt_rgb, mgmt_hex


def fix_segment_colors(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Projde segments a:
      - seg["color"] nastaví dle speciesId (pokud existuje v paletě)
      - seg["treeAttributes"]["speciesColorHex"] nastaví dle speciesId
      - seg["treeAttributes"]["managementColorHex"] nastaví dle managementStatusId
    """
    species_rgb, species_hex, mgmt_rgb, mgmt_hex = build_palette_maps(data)

    stats = {
        "segments_total": 0,
        "seg_color_updated": 0,
        "species_hex_updated": 0,
        "mgmt_hex_updated": 0,
        "missing_speciesId": 0,
        "missing_mgmtId": 0,
        "speciesId_not_in_palette": 0,
        "mgmtId_not_in_palette": 0,
    }

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        return data, stats

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        stats["segments_total"] += 1

        # treeAttributes vždy jako dict
        ta = seg.get("treeAttributes")
        if not isinstance(ta, dict):
            ta = {}
            seg["treeAttributes"] = ta

        # --- speciesId -> seg["color"] + speciesColorHex ---
        sid = seg.get("speciesId")
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = None
            stats["missing_speciesId"] += 1

        if sid_int is not None:
            rgb = species_rgb.get(sid_int)
            hx = species_hex.get(sid_int)
            if rgb is None or hx is None:
                stats["speciesId_not_in_palette"] += 1
            else:
                seg["color"] = rgb
                stats["seg_color_updated"] += 1

                ta["speciesColorHex"] = hx
                stats["species_hex_updated"] += 1

        # --- managementStatusId -> managementColorHex ---
        mid = seg.get("managementStatusId")
        try:
            mid_int = int(mid)
        except Exception:
            mid_int = None
            stats["missing_mgmtId"] += 1

        if mid_int is not None:
            hx = mgmt_hex.get(mid_int)
            if hx is None:
                stats["mgmtId_not_in_palette"] += 1
            else:
                ta["managementColorHex"] = hx
                stats["mgmt_hex_updated"] += 1

    return data, stats


def main(input_json: str, output_json: Optional[str] = None) -> None:
    if not os.path.exists(input_json):
        raise FileNotFoundError(input_json)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed, stats = fix_segment_colors(data)

    if output_json is None:
        root, ext = os.path.splitext(input_json)
        output_json = f"{root}.colors_fixed{ext}"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print("✅ Hotovo")
    print("Uloženo:", output_json)
    print("Statistiky:", stats)


if __name__ == "__main__":
    INPUT = r"C:/users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v3.json"
    OUTPUT = r"C:/users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v4.json"
    main(INPUT,OUTPUT)
