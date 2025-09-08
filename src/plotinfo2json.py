import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


input_file = Path("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v11.json")
output_file = Path("c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/_PokojnaHora_v12.json")
settings_file = Path("c:/Users/krucek/Documents/GitHub/3d-marteloscope/pokojna_info.json")

# načtení JSONu
with input_file.open("r", encoding="utf-8") as f:
    data = json.load(f)

with settings_file.open("r", encoding="utf-8") as fs:
    cols = json.load(fs)


cdf = pd.DataFrame(cols["plot_info"])
# přidání do dat
data["plot_info"] = cols["plot_info"]


# uložení
with output_file.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Plot info added {output_file}")
