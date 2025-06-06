import pandas as pd
import json
import src.io as io
import src.data_prep.species as spp



file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json"
trees = io.load_project_json(file_path)

#trees["management_status"] = "untouched"
#trees["species"] = trees["label"].apply(spp.map_species_codes2latin)
#trees["dbh"] = trees["dbh"].where(trees["dbh"] < 200, 0)

trees["surfaceAreaProjection"] = 0
trees["surfaceArea"] = 0
trees["volume"] = 0

io.save_project_json(file_path,trees,"c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/LS-Krivoklat/3df_project/Krivoklat_test_SAVE.json")



colors = io.load_colormap(file_path)
colors.keys()
colors["management"]

