import numpy as np
import pandas as pd
import rasterio
from functools import lru_cache
from shapely.geometry import Point
from pyproj import Transformer
from shapely.geometry import Point
from scipy.spatial import cKDTree


# -----------------------------
# 1) Textura půdy z rastrových TIFFů
# -----------------------------
def _sample_one(
    src, 
    x, 
    y
):
    return float(next(src.sample([(x, y)]))[0])

def extract_soil_texture_data(
    lat: float, 
    lon: float, 
    site_name: str
) -> pd.DataFrame:
    
    soil_depth = 80 # hodnota pro kterou mame reaktivitu pudy

    with rasterio.open("soil_data/Silt1.tif") as silt_src, \
         rasterio.open("soil_data/Clay.tif") as clay_src:

        # Připrav transformery jen jednou
        tr_silt = Transformer.from_crs("EPSG:4326", silt_src.crs, always_xy=True)
        tr_clay = Transformer.from_crs("EPSG:4326", clay_src.crs, always_xy=True)

        x_silt, y_silt = tr_silt.transform(lon, lat)
        x_clay, y_clay = tr_clay.transform(lon, lat)

        pctSilt = _sample_one(silt_src, x_silt, y_silt)
        pctClay = _sample_one(clay_src, x_clay, y_clay)

    pctSilt = int(round(pctSilt))
    pctClay = int(round(pctClay))

    # základní výpočet písku
    pctSand = 100 - (pctSilt + pctClay)

    # když po zaokrouhlení silt+clay > 100 → dorovnej z větší složky a sand dej na 0
    if pctSand < 0:
        deficit = -pctSand
        if pctSilt >= pctClay:
            pctSilt = max(0, pctSilt - deficit)
        else:
            pctClay = max(0, pctClay - deficit)
        pctSand = 0

    # pojistky do [0,100]
    pctSilt = int(np.clip(pctSilt, 0, 100))
    pctClay = int(np.clip(pctClay, 0, 100))
    pctSand = int(np.clip(pctSand, 0, 100))

    # finální dorovnání, kdyby numerika něco posunula (vždy upravíme sand)
    total = pctSand + pctSilt + pctClay
    if total != 100:
        pctSand += (100 - total)

    out = pd.DataFrame({
        "id": [1],
        "Site": [site_name],
        "pctSand": [int(np.rint(pctSand))],
        "pctSilt": [int(np.rint(pctSilt))],
        "pctClay": [int(np.rint(pctClay))],
        "soilDepth": [soil_depth],
    })
    return out


# -----------------------------
# 2) Chemie půdy (CZ území)
# -----------------------------
# ---------- helpers ----------
def _num(x: pd.Series | float | str | None) -> pd.Series:
    
    return pd.to_numeric(x, errors="coerce")

def _stock(soil_mass, pct):
    if pd.isna(soil_mass) or pd.isna(pct):
        return np.nan
    return soil_mass * (pct / 100.0)

def _cn(c_t, n_t):
    if pd.isna(c_t) or pd.isna(n_t) or n_t <= 0:
        return np.nan
    return c_t / n_t

def _load_chemistry_df(path: str = "soil_data/Soil_chemistry_CZE.csv") -> pd.DataFrame:
    """
    Načte CSV soubor se středníkem jako oddělovačem.
    Načítání proběhne jen jednou za běh procesu (cache přes lru_cache).
    """
    return pd.read_csv(path, sep=";")


@lru_cache(maxsize=1)
def _build_kdtree():
    df = _load_chemistry_df()
    xy = df[["JTSK.x", "JTSK.y"]].to_numpy(dtype=float)
    tree = cKDTree(xy)
    return tree, df

# ---------- main functions funkce ----------
def extract_soil_chemistry_czeterritory(point_geom_wgs84: Point, point_crs="EPSG:4326") -> pd.DataFrame:
    # Načteme KD-strom a DataFrame
    tree, df = _build_kdtree()

    # Převod souřadnic na JTSK
    transformer = Transformer.from_crs(point_crs, "EPSG:5514", always_xy=True)
    x, y = transformer.transform(point_geom_wgs84.x, point_geom_wgs84.y)

    # Najdeme index nejbližšího bodu
    dist, idx = tree.query([x, y], k=1)
    if np.isinf(dist) or idx is None:
        return pd.DataFrame([{"Site_name": np.nan, "C": np.nan, "N": np.nan, "CN": np.nan}])

    obj_id = df.iloc[idx]["OBJECTID"]
    site_chem = df[df["OBJECTID"] == obj_id]

    # Vybereme první řádek pro každý horizont
    def pick(horizon):
        sub = site_chem.loc[site_chem["Horizon"] == horizon]
        if sub.empty:
            return np.nan, np.nan, np.nan
        r = sub.iloc[0]
        return _num(r["SoilMass"]), _num(r["C"]), _num(r["N"])

    Soil_FH, C_FH, N_FH = pick("FH")
    Soil_L1, C_L1, N_L1 = pick("0-30 cm")
    Soil_L2, C_L2, N_L2 = pick("30-80 cm")

    # Přepočet na zásoby (t/ha)
    C_FH_t, N_FH_t = _stock(Soil_FH, C_FH), _stock(Soil_FH, N_FH)
    C_L1_t, N_L1_t = _stock(Soil_L1, C_L1), _stock(Soil_L1, N_L1)
    C_L2_t, N_L2_t = _stock(Soil_L2, C_L2), _stock(Soil_L2, N_L2)

    # Součty
    C_tot = np.round(np.nansum([C_FH_t, C_L1_t, C_L2_t]), 2)
    N_tot = np.round(np.nansum([N_FH_t, N_L1_t, N_L2_t]), 2)
    CN_tot = np.round(_cn(C_tot, N_tot), 2)

    return pd.DataFrame([{
        "Site_name": np.nan,  # vyplní wrapper
        "C": float(C_tot),
        "N": float(N_tot),
        "CN": float(CN_tot),
    }])

def extract_soil_chemistry_data(lat: float, lon: float, site_name: str) -> pd.DataFrame:
    point_wgs84 = Point(lon, lat)
    out = extract_soil_chemistry_czeterritory(point_wgs84, "EPSG:4326")
    out.loc[:, "Site_name"] = site_name
    return out
