import json
import pandas as pd
import numpy as np
import os
import sqlite3
from typing import Dict, Union, List, Any, Optional


file_path = "c:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/PokojnaHora_3df/pokojna_test.json"

trees = load_project_json(file_path)
