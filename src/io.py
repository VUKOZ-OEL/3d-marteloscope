import json
import pandas as pd
from typing import List, Dict, Union

__all__ = ["load_project_json","save_project_json"]
"""
def load_project_json(file_path: str) -> pd.DataFrame:

    print(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    # Extrakce "segments > treeAttributes"
    segments: List[Dict[str, Union[int, str]]] = data.get("segments", [])
    tree_attributes: List[Dict[str, Union[float, int, str]]] = []

    for segment in segments:
        if "treeAttributes" in segment:
            attributes = segment["treeAttributes"]
            selected_attributes = {
                "id": int(segment.get("id", 0)),  # ID je vždy int
                "label": str(segment.get("label", "")),  # Label je string
                "dbh": float(attributes.get("dbh", 0.0))/100,  # DBH je float
                "height": float(attributes.get("height", 0.0))/10000,  # Výška je float
                "status": str(attributes.get("status", "Unknown")),  # Status je string
                "x": float(attributes.get("position", [0.0, 0.0, 0.0])[0])/10000,  # X pozice
                "y": float(attributes.get("position", [0.0, 0.0, 0,0])[1])/10000,  # Y pozice
                "lat": float(attributes.get("latlon", [0.0, 0.0])[1]),  # lat pozice
                "lon": float(attributes.get("latlon", [0.0, 0.0])[0]),  # lon pozice
            }
            tree_attributes.append(selected_attributes)

    # Vytvoření pandas DataFrame
    return pd.DataFrame(tree_attributes)
"""

import json
import pandas as pd
from typing import List, Dict, Union

def load_project_json(file_path: str) -> pd.DataFrame:

    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Union[Dict, List]] = json.load(f)

    segments: List[Dict] = data.get("segments", [])
    tree_attributes: List[Dict] = []

    for segment in segments:
        if "treeAttributes" in segment:
            base = {}
            attributes = segment["treeAttributes"]

            # Vždy přidáme ID a label z úrovně segmentu
            base["id"] = int(segment.get("id", 0))
            base["label"] = str(segment.get("label", ""))

            # Všechny atributy z treeAttributes přidáme
            for key, value in attributes.items():
                base[key] = value

            # Extrahujeme polohu zvlášť jako x, y
            if isinstance(attributes.get("position"), list) and len(attributes["position"]) >= 2:
                base["x"] = float(attributes["position"][0]) / 10000
                base["y"] = float(attributes["position"][1]) / 10000
            else:
                base["x"], base["y"] = 0.0, 0.0

            # Extrahujeme lat/lon
            if isinstance(attributes.get("latlon"), list) and len(attributes["latlon"]) >= 2:
                base["lon"] = float(attributes["latlon"][0])
                base["lat"] = float(attributes["latlon"][1])
            else:
                base["lon"], base["lat"] = 0.0, 0.0

            # Volitelně: přepočty (nepřepisujeme původní hodnoty)
            if "dbh" in base:
                base["dbh"] = float(base["dbh"]) / 100
            if "height" in base:
                base["height"] = float(base["height"]) / 10000

            tree_attributes.append(base)

    return pd.DataFrame(tree_attributes)


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
                    float(row["x"]) * 10000,
                    float(row["y"]) * 10000,
                    z
                ]
            if "lon" in row and "lat" in row:
                attr["latlon"] = [
                    float(row["lon"]),
                    float(row["lat"])
                ]
            if "dbh" in row:
                attr["dbh"] = float(row["dbh"]) * 100
            if "height" in row:
                attr["height"] = float(row["height"]) * 10000

            # Vyloučíme klíče, které už existují i ve vnějším segmentu
            skip_keys = {"id", "x", "y", "lat", "lon", "dbh", "height"}
            outer_keys = set(segment.keys())  # např. "label" na úrovni segmentu
            for key, value in row.items():
                if key not in skip_keys and key not in outer_keys:
                    attr[key] = value  # bezpečné přepsání nebo přidání

    output_path = output_path or original_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
