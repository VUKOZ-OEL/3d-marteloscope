import pandas as pd
import json
import src.io_utils as iou
#import src.colors2json as c2j


file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/PokojnaHora.json"
out_file = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_mod.json"
SETTINGS_PATH = ("c:/Users/krucek/Documents/GitHub/VUKOZ/3d-forest/.config/settings.json")


trees = iou.load_project_json(file_path)

trees["volume"] = trees["Volume_m3"]
trees["dbhPosition"] = trees["position"]
trees["dbhPosition"] = trees["dbhPosition"].apply(
    lambda v: [v[0], v[1], v[2] + 1.3]
)
trees["dbh"] = trees["dbh"] / 100

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)

# --- načtení species do DataFrame ---
df_species = pd.DataFrame(settings["tree_species"])[["latin", "abbreviation", "color", "id"]]
df_species.rename(columns={"id": "speciesId","latin": "species"}, inplace=True)

df_merged = trees.merge(
    df_species,
    how="left",
    left_on="species",
    right_on="species"
)

df_merged["speciesId"]

iou.save_project_json(file_path,df_merged,out_file)
save_project_json(file_path,df_merged,out_file)
