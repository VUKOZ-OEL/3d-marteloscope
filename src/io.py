import json
import pandas as pd
from typing import List, Dict, Union

__all__ = ["load_project_json"]

def load_project_json(file_path: str) -> pd.DataFrame:

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
                "dbh": float(attributes.get("dbh", 0.0)),  # DBH je float
                "height": float(attributes.get("height", 0.0)),  # Výška je float
                "status": str(attributes.get("status", "Unknown")),  # Status je string
                "x": float(attributes.get("position", [0.0, 0.0, 0.0])[0])/10000,  # X pozice
                "y": float(attributes.get("position", [0.0, 0.0, 0,0])[1])/10000,  # Y pozice
                "lat": float(attributes.get("latlon", [0.0, 0.0])[1]),  # lat pozice
                "lon": float(attributes.get("latlon", [0.0, 0.0])[0]),  # lon pozice
            }
            tree_attributes.append(selected_attributes)

    # Vytvoření pandas DataFrame
    return pd.DataFrame(tree_attributes)
