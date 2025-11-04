import json
from pathlib import Path

# --- cesty ---
SETTINGS_PATH = Path("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/settings.json")
FOREST_PATH   = Path("data/test_project.json")
OUT_PATH      = Path("data/test_project_up.json")

# --- načtení JSON souborů ---
with SETTINGS_PATH.open("r", encoding="utf-8") as f:
    settings = json.load(f)
with FOREST_PATH.open("r", encoding="utf-8") as f:
    forest = json.load(f)

# --- vytvoření mapy abbreviation → (color, abbreviation) ---
species = settings.get("tree_species", [])
abbr_map = {}
for s in species:
    abbr = (s.get("abbreviation") or "").strip().lower()
    if not abbr:
        continue
    abbr_map[abbr] = {
        "color": s.get("color"),
        "abbreviation": s.get("abbreviation")
    }

# --- pomocná rekurzivní funkce pro průchod strukturou ---
def update_segments(seg):
    """Projde segment nebo seznam segmentů a aktualizuje barvy stromů podle abbreviation."""
    if isinstance(seg, list):
        for s in seg:
            update_segments(s)
        return
    if not isinstance(seg, dict):
        return

    # pokud obsahuje stromy
    if "trees" in seg and isinstance(seg["trees"], list):
        for t in seg["trees"]:
            abbr = (t.get("abbreviation") or t.get("species") or "").strip().lower()
            if abbr in abbr_map:
                ref = abbr_map[abbr]
                if ref.get("color") is not None:
                    t["color"] = ref["color"]
                t["abbreviation"] = ref["abbreviation"]

    # pokud obsahuje vnořené segmenty
    if "segments" in seg:
        update_segments(seg["segments"])

# --- aktualizace stromů ---
update_segments(forest.get("segments", []))

# --- uložení ---
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(forest, f, ensure_ascii=False, indent=2)

print(f"Aktualizace dokončena. Uloženo do: {OUT_PATH}")
