from pydeck import View
import pandas as pd
import json
import src.io_utils as iou
# import src.data_prep.species as spp
# import src.colors2json as c2j


# "managementStatusId": 0,
# "speciesId": 0,


def write_json(original_path: str, df: pd.DataFrame, output_path: str = None) -> None:

    # --- Load JSON ---
    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # dataframe -> dict podle id
    id_map = df.set_index("id").to_dict("index")

    # klíče, které se mají přepisovat do segmentu (outer level)
    outer_rewrite_keys = {"label", "managementStatusId", "speciesId"}

    for segment in data.get("segments", []):
        sid = segment.get("id")
        if sid not in id_map or "treeAttributes" not in segment:
            continue

        row = id_map[sid]
        attr = segment["treeAttributes"]

        # ---- (1) Zachování Z-ové hodnoty ----
        z = 0.0
        if isinstance(attr.get("position"), list) and len(attr["position"]) >= 3:
            z = attr["position"][2]

        # ---- (2) Přepis pozice pokud df obsahuje x,y ----
        if "x" in row and "y" in row:
            attr["position"] = [float(row["x"]), float(row["y"]), z]

        # ---- (3) Přepis OUTER KEYS, pokud jsou v DF ----
        # label, managementStatusId, speciesId
        for key in outer_rewrite_keys:
            if key in row and row[key] is not None:
                segment[key] = row[key]

        # ---- (4) Tyto klíče NESMÍ být v treeAttributes ----
        for forbidden in outer_rewrite_keys:
            if forbidden in attr:
                del attr[forbidden]

        # ---- (5) Ostatní atributy – přepis jen pokud nejsou outer keys segmentu ----
        skip_keys = {"id"} | outer_rewrite_keys
        outer_keys = set(segment.keys())

        for key, value in row.items():
            if key in skip_keys:
                continue
            if key in outer_keys:
                continue
            attr[key] = value  # bezpečné přepsání nebo přidání

    # --- save ---
    output_path = output_path or original_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



file_path_2 = "data/pokojna_test_v2_2.json"
out_file = "data/pokojna_test_v2_3.json"

trees = iou.load_project_json(file_path)
view(trees)
trees["dbh"]


