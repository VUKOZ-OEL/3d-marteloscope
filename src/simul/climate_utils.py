import os
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point

# NNames of variables in folders and in netCDF files
VAR_MAP = {
    "Temperature_average": "tas",
    "Temperature_min": "tasmin",
    "Temperature_max": "tasmax",
    "Precipitation": "pr",
    "Relative_humidity": "hurs",
    "Solar_radiation": "rsds"
}

# Return climatic variable name as is in netCDF file
def get_var_name(var: str) -> str:
    return VAR_MAP.get(var, None)

# Extract data from one netCDF file for one specific point
# source - folder with climatic scenario datasets (netCDF files)
# point - lat and lon of the point
# var - climatic variable to be extracted
def extract_pointvalues_from_netcdf(source, point, var):

    # get folder with netCDF files
    folder_route = os.path.join(source, var)
    
    # get names of all files in the folder
    files = sorted([f for f in os.listdir(folder_route) if f.endswith(".nc")])

    # output inicialization
    output = pd.DataFrame(columns=["time", "value"])

    # loading data from netCDF files
    chunks = []
    for nc_file in files:
        file_path = os.path.join(folder_route, nc_file)
        ds = xr.open_dataset(file_path)
        
        # return the name of the variable in netCDF file
        var_netcdf = get_var_name(var)

        # data extraction (bilineární interpolation)
        try:
            vals = ds[var_netcdf].interp(lat=point.y, lon=point.x, method="linear")
        except Exception:
            vals = ds[var_netcdf].sel(lat=point.y, lon=point.x, method="nearest")
        
        temporary_output = pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values),
            "value": vals.values.flatten()
        })

         # skip empty data
        if not temporary_output.empty and not temporary_output["value"].isna().all():
            chunks.append(temporary_output)
        else:
            pass

         # Connect the data
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.DataFrame(columns=["time", "value"])

# Extract one variable (both historical and predicted data)
# point - lat and lon of the point
# scn - scenario of climatic predictions
# var - climatic variable to be extracted
# historical - if historical data will be included (may significantly increase the time needed for the extraction)
def extract_variable(point, scn, var, historical=False):

    # if historical data are needed, then extract all historical data
    if historical:
        output_historical = extract_pointvalues_from_netcdf("climate_data/historical", point, var)
    else:
        output_historical = pd.DataFrame(columns=["time", "value"])

    # sets the source of climatic datasets
    if scn == "RCP 4.5":
        source_root = "climate_data/RCP_45"
    elif scn == "RCP 8.5":
        source_root = "climate_data/RCP_85"
    else:
        raise ValueError("Neznámý scénář: {scn}")

    # extract climatic data based on the scenario
    output_prediction = extract_pointvalues_from_netcdf(source_root, point, var)

    return pd.concat([output_historical, output_prediction], ignore_index=True)

# Calculates VPD
# temperature - mean daily temperature
# humidity - mean relative humidity
def calculate_vpd(temperature, humidity):
    temperature = temperature - 273.15  # převod K → °C
    es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))  # kPa
    ea = es * (humidity / 100)
    vpd_kPa = es - ea
    return vpd_kPa

# Extract climatic data for iLand model
# lat - site latitude
# lon - site longitude
# scn - scenario of climatic predictions
# include_historical_data - if historical data will be included (may significantly increase the time needed for the extraction)
def extract_site_climate(lat, lon, scenario, include_historical_data=False):
    point = Point(lon, lat)

    print("...extracting average temperatures")
    tavg = extract_variable(point, scenario, "Temperature_average", include_historical_data)
    print("...extracting minimal temperatures")
    tmin = extract_variable(point, scenario, "Temperature_min", include_historical_data)
    print("...extracting maximal temperatures")
    tmax = extract_variable(point, scenario, "Temperature_max", include_historical_data)
    print("...extracting precipitations")
    prcp = extract_variable(point, scenario, "Precipitation", include_historical_data)
    print("...extracting relative humidity")
    rhum = extract_variable(point, scenario, "Relative_humidity", include_historical_data)
    print("...extracting solar radiation")
    srad = extract_variable(point, scenario, "Solar_radiation", include_historical_data)

    df = pd.DataFrame({
        "year": pd.to_datetime(tavg["time"]).dt.year,
        "month": pd.to_datetime(tavg["time"]).dt.month,
        "day": pd.to_datetime(tavg["time"]).dt.day,
        "mean_temp": np.round(tavg["value"] - 273.15, 2),
        "min_temp": np.round(tmin["value"] - 273.15, 2),
        "max_temp": np.round(tmax["value"] - 273.15, 2),
        "prec": np.round(prcp["value"] * 86400, 2),
        "rad": np.round(srad["value"] * 86400 / 1e6, 4),
        "vpd": np.round(calculate_vpd(tavg["value"], rhum["value"]), 10),
    })

    return df
