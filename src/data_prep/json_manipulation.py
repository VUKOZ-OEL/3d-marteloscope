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



file_path_2 = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_mod.json"
# out_file = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_mod.json"
settings_file = (
    "C:/Users/krucek/Documents/GitHub/VUK/3d-marteloscope/settings_with_colors.json"
)

file_path_3 = (
    "C:/Users/krucek/Documents/GitHub/VUK/3d-marteloscope/data/test_project.json"
)
out_file = (
    "C:/Users/krucek/Documents/GitHub/VUK/3d-marteloscope/data/pokojna_test_v2.json"
)
trees = iou.load_project_json(file_path)
trees2 = iou.load_project_json(file_path_2)

trees3 = iou.load_project_json(file_path_3)
trees3.to_feather("C:/Users/krucek/Documents/GitHub/VUK/3d-marteloscope/data/test_project.json.feather")

trees["speciesId"] = trees2["speciesId"]

trees["managementStatusId"] = 0

idx = trees.sample(frac=0.1).index
trees.loc[idx, "managementStatusId"] = 1
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 2
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 3
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 4
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 5
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 6
idx = trees.sample(frac=0.05).index
trees.loc[idx, "managementStatusId"] = 7
 	
trees = trees.drop(columns=["speciesColorHex", "managementColorHex"])

write_json(file_path, trees, out_file)

#polys = pd.read_feather("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/planar_projections.feather")
qsm = pd.read_feather(
    "C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/qsm2json.feather"
)
qsm = qsm.rename(columns={"id": "label"})
# polys = polys.rename(columns={"geometry": "planar_projection_poly"})

df_merged = trees.merge(qsm, left_on="label", right_on="label", how="left")


trees["volume"] = trees["Volume_m3"]
trees["dbhPosition"] = trees["position"]
trees["dbhPosition"] = trees["dbhPosition"].apply(lambda v: [v[0], v[1], v[2] + 1.3])
trees["dbh"] = trees["dbh"] / 100

write_json(file_path, trees, out_file)


trees.to_feather(
    "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora.json.trees.feather"
)


trees["dbh"] = trees["dbh_raycl"] * 100

trees["management_status"] = "Untouched"
trees.loc[trees.index[1:23], "management_status"] = "Target tree"
trees.loc[trees.index[25:100], "management_status"] = "Competition"
trees.loc[trees.index[101:123], "management_status"] = "Maturity"
trees.loc[trees.index[124:223], "management_status"] = "Sanitary"
trees.loc[trees.index[300:353], "management_status"] = "Promote rare species"
trees.loc[trees.index[354:400], "management_status"] = "Promote regeneration"

trees.to_feather(
    "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v10.feather"
)

trees

write_json(file_path, trees, out_file)


file_path2 = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v10.json"
out_file2 = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v11.json"
settings_file = (
    "C:/Users/krucek/Documents/GitHub/3d-marteloscope/settings_with_colors.json"
)
# c2j.add_palettes(file_path2,out_file2,settings_file)






trees3 = iou.load_project_json("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json")
print(trees3)
trees3.to_feather("C:/Users/krucek/Documents/GitHub/VUK/3d-marteloscope/data/pokojna_test_v2.json.feather")


df_martelo = iou.load_project_json("data/pokojna_test_v2.json")
df_3df = iou.load_project_json("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json")

print(df_martelo)
print(df_3df)

df_3df["species"] = df_martelo["species"]
df_3df["management_status"] = df_martelo["management_status"]

write_json("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json",
            df_3df,
            "C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v3.json")


view(df_3df)


df_martelo = iou.load_project_json("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json")
df_martelo.to_feather("data/test_project.json.feather")

print(df_martelo)
species_id = pd.read_feather("data/id_label_species_id.feather")
print(species_id)
df_martelo["label"] = species_id["label"]


write_json("C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v2.json",
            df_martelo,
            "C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test_v3.json")


pp1 = "d:/_gs_lcr/pokojna_hora/pokojna_hora_v2.json"
#pp2 = "data/test_project_v2.json"
pp2 = "d:/_gs_lcr/pokojna_hora/pokojna_hora.json"
prj = iou.load_project_json(pp1)
prj2 = iou.load_project_json(pp2)


view(prj)
view(prj2)

prj["dbhPosition"] = prj2["dbhPosition"]
prj["position"] = prj2["position"]
write_json(pp1, prj, "d:/_gs_lcr/pokojna_hora/pokojna_hora_v3.json")
