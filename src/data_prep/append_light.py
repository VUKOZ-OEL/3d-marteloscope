import pandas as pd
import json
import src.io_utils as iou
#import src.data_prep.species as spp
import src.data_prep.colors2json as c2j

def write_json(original_path: str, df: pd.DataFrame, output_path: str = None) -> None:
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


file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
light_path = "C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/test_poses.json"

trees = iou.load_project_json(file_path)
light = pd.read_json(light_path)
light = light.rename(columns={
    "ID": "id",
    "free_space_pct": "alight_avail",
    "shade_breakdown": "light_comp",
})
trees = trees.merge(light, on="id", how="left")

write_json("c:/Users/krucek/Documents/GitHub/3d-marteloscope/data/test_project.json", trees, "c:/Users/krucek/Documents/GitHub/3d-marteloscope/data/test_project.json" )
