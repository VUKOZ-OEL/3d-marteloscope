
# Promenne nutne pro beh iLandu pro danny site
site_name = "Krivoklat"
lat = 49.9599942
lon = 13.8295778
initial_date = 2025
climate_scenario = "RCP 4.5"
initial_forest_file = "krivoklat_test_trees_iland.txt"

from climate_utils import extract_site_climate
from soil_utils import extract_soil_texture_data
from soil_utils import extract_soil_chemistry_data
# from data_preparation import write_sql_file
from data_preparation import write_sqlite_climate_hard
from xml_file_utils import write_iland_minimal_project_xml

from iLand_run import prepare_iland_files
from iLand_run import run_multiple_iland
from outputs_graphs import read_outputs_tree_level
from outputs_graphs import show_tree_deaths_by_species
from outputs_graphs import show_tree_deaths_by_year_species
from outputs_graphs import show_deaths_volume_by_species
from outputs_graphs import show_deaths_volume_by_year_species
from outputs_graphs import show_trees_by_species_presence
from outputs_graphs import show_trees_by_species_vol
from output_forest_development import prepare_dataset
from output_forest_development import run_forest_development

# Krivoklat
print("------------- preparing Krivoklat -------------")
prepare_iland_files(
    site_name = "Krivoklat",
    lat = 49.9599942,
    lon = 13.8295778,
    initial_date = 2025,
    climate_scenario = "RCP 4.5"
)

# Buchlovice
print("------------- preparing Buchlovice -------------")
prepare_iland_files(
    site_name = "Buchlovice",
    lat = 49.1169861,
    lon = 17.2982428,
    initial_date = 2025,
    climate_scenario = "RCP 4.5"
)

# Pokojna_hora
print("------------- preparing Pokojna_hora -------------")
prepare_iland_files(
    site_name = "Pokojna_hora",
    lat = 49.3323475,
    lon = 16.6927517,
    initial_date = 2025,
    climate_scenario = "RCP 4.5"
)

# Klepacov_1
print("------------- preparing Klepacov_1 -------------")
prepare_iland_files(
    site_name = "Klepacov_1",
    lat = 49.344908476555609,
    lon = 16.673507129932091,
    initial_date = 2025,
    climate_scenario = "RCP 4.5"
)

# Klepacov_2
print("------------- preparing Klepacov_2 -------------")
prepare_iland_files(
    site_name = "Klepacov_2",
    lat = 49.342482979567727,
    lon = 16.671187825639663,
    initial_date = 2025,
    climate_scenario = "RCP 4.5"
)



print("Running iLand")
run_multiple_iland(
    # project_file = "iLand/krivoklat_project_pokus_2.xml",
    # project_file = "iLand/pokus_minimal.xml",
    # project_file = "iLand/pokus_created_upraveno.xml",
    project_file = "iLand/pokus_created.xml",
    base_out = f"iLand_outputs/{site_name}",
    years = 30)

# Cesta ke složce s výstupy z iLandu — uprav podle toho, kam je ukládáš
output_folder = f"iLand_outputs/{site_name}/outputs"

# Načti data
outputs = read_outputs_tree_level(output_folder)

# pousti forest development
# df_forest_development = prepare_dataset(outputs)
# run_forest_development(df_forest_development)


# Smrti stromu dle druhu na yaklade jejich prezence
show_tree_deaths_by_species(
    outputs,
    scale="absolute",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_tree_deaths_by_species(
    outputs,
    scale="relative",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_tree_deaths_by_species(
    outputs,
    scale="relative_toForest",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)

# Smrti stromu dle druhu na zaklade jejich prezence dle jednotlivych let
show_tree_deaths_by_year_species(
    outputs,
    scale="absolute",   # nebo "absolute" / "relative"
    preview=True,
    save=False                   # dej True pro export do JPG 300 dpi
)
show_tree_deaths_by_year_species(
    outputs,
    scale="relative",   # nebo "absolute" / "relative"
    preview=True,
    save=False                   # dej True pro export do JPG 300 dpi
)
show_tree_deaths_by_year_species(
    outputs,
    scale="relative_toForest",   # nebo "absolute" / "relative"
    preview=True,
    save=False                   # dej True pro export do JPG 300 dpi
)

# Smrti objemu biomasy dle druhu na zaklade jejich prezence
show_deaths_volume_by_species(
    outputs,
    scale="absolute",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_deaths_volume_by_species(
    outputs,
    scale="relative",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_deaths_volume_by_species(
    outputs,
    scale="relative_toForest",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)

# Smrti objemu biomasy v jednotlivych letech dle druhu na zaklade jejich prezence
show_deaths_volume_by_year_species(
    outputs,
    scale="absolute",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_deaths_volume_by_year_species(
    outputs,
    scale="relative",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_deaths_volume_by_year_species(
    outputs,
    scale="relative_toForest",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)

# Prezence stromu v jednotlivych letech
show_trees_by_species_presence(
    outputs,
    scale="absolute",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_trees_by_species_presence(
    outputs,
    scale="relative",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)

# Living volume v jednotlivych letech
show_trees_by_species_vol(
    outputs,
    scale="absolute",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)
show_trees_by_species_vol(
    outputs,
    scale="relative",  # nebo "absolute" / "relative"
    preview=True,
    save=False                   # můžeš dát True, když chceš uložit JPG
)

