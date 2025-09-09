import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

def write_iland_minimal_project_xml(
    site_name: str,
    lat: float,
    initial_forest: str,
    soil_texture: pd.DataFrame,
    soil_chemistry: pd.DataFrame
) -> Path:

    # --- build XML ---
    iland = ET.Element("iland")

    # --- system ---
    system = ET.SubElement(iland, "system")

    # path
    path = ET.SubElement(system, "path")
    ET.SubElement(path, "home").text = ""
    ET.SubElement(path, "database").text = "database"
    ET.SubElement(path, "lip").text = "lip"
    ET.SubElement(path, "temp").text = "temp"
    ET.SubElement(path, "script").text = "scripts"
    ET.SubElement(path, "init").text = "init"
    ET.SubElement(path, "output").text = "output"

    # database
    database = ET.SubElement(system, "database")
    ET.SubElement(database, "in").text = "all_species_database.sqlite"
    ET.SubElement(database, "climate").text = f"climate_file_{site_name}.sqlite"
    ET.SubElement(database, "out").text = "output.sqlite"

    # logging (abychom nÏco vidÏli p¯ed segfaultem)
    logging = ET.SubElement(system, "logging")
    ET.SubElement(logging, "logTarget").text = "file"
    ET.SubElement(logging, "logFile").text = "{site_name}_log.txt"
    ET.SubElement(logging, "flush").text = "false"

    # --- model ---
    model = ET.SubElement(iland, "model")

    # world
    world = ET.SubElement(model, "world")
    ET.SubElement(world, "cellSize").text = str(2)
    ET.SubElement(world, "width").text = str(100)
    ET.SubElement(world, "height").text = str(100)
    ET.SubElement(world, "buffer").text = str(100)
    ET.SubElement(world, "latitude").text = str(lat)
    ET.SubElement(world, "resourceUnitsAsGrid").text = "true"
    
    # site
    site = ET.SubElement(model, "site")
    ET.SubElement(site, "soilDepth").text = str(soil_texture.at[0, "soilDepth"])
    ET.SubElement(site, "pctSand").text = str(soil_texture.at[0, "pctSand"])
    ET.SubElement(site, "pctSilt").text = str(soil_texture.at[0, "pctSilt"])
    ET.SubElement(site, "pctClay").text = str(soil_texture.at[0, "pctClay"])
    ET.SubElement(site, "somC").text = str(soil_chemistry.at[0, "C"])
    ET.SubElement(site, "somN").text = str(soil_chemistry.at[0, "N"])
    
    # modules

    # settings
    settings = ET.SubElement(model, "settings")
    ET.SubElement(settings, "growthEnabled").text = "true"
    ET.SubElement(settings, "mortalityEnabled").text = "true"
    ET.SubElement(settings, "regenerationEnabled").text = "true"

    # species
        # iLand potrebuje tuto sekci pro to, aby bezel
        # Definovany jsou implicitni hodnoty dle iLand dokumentace
    species = ET.SubElement(model, "species")
    ET.SubElement(species, "source").text = "species"  # n·zev tabulky v all_species_database.sqlite

    # ... species - NitrogenResponseClasses
        # nazvy class_1_a se mohou lisit dle verze iLandu
    nrc = ET.SubElement(species, "nitrogenResponseClasses")
    ET.SubElement(nrc, "class_1_a").text = "-0.045"
    ET.SubElement(nrc, "class_1_b").text = "10"
    ET.SubElement(nrc, "class_2_a").text = "-0.055"
    ET.SubElement(nrc, "class_2_b").text = "25"
    ET.SubElement(nrc, "class_3_a").text = "-0.065"
    ET.SubElement(nrc, "class_3_b").text = "40"

    # ... species - CO2 response
    co2 = ET.SubElement(species, "CO2Response")
    ET.SubElement(co2, "p0").text = "1"
    ET.SubElement(co2, "baseConcentration").text = "340"
    ET.SubElement(co2, "compensationPoint").text = "80"
    ET.SubElement(co2, "beta0").text = "0.6"

    # ... species - Light response
    light = ET.SubElement(species, "lightResponse")
    ET.SubElement(light, "shadeTolerant").text = "1-exp(-6*(lri-0.05))"     # class 1
    ET.SubElement(light, "shadeIntolerant").text = "1-exp(-3*(lri-0.01))"  # class 5
    ET.SubElement(light, "LRImodifier").text = "exp(ln(lri)/0.5*(1-0.5*relH))" # pravdepodobne neni nutne

    # ... species - Phenology 
        # pouze implicitni hodnoty pro druhy nedefinovane v databazi
    phen = ET.SubElement(species, "phenology")

    # ... deciduous broadleaved
    t1 = ET.SubElement(phen, "type", {"id": "1"})
    ET.SubElement(t1, "vpdMin").text = "0.9"
    ET.SubElement(t1, "vpdMax").text = "4.1"
    ET.SubElement(t1, "dayLengthMin").text = "10"
    ET.SubElement(t1, "dayLengthMax").text = "11"
    ET.SubElement(t1, "tempMin").text = "2"
    ET.SubElement(t1, "tempMax").text = "9"

    # ... deciduous coniferous
    t2 = ET.SubElement(phen, "type", {"id": "2"})
    ET.SubElement(t2, "vpdMin").text = "1"
    ET.SubElement(t2, "vpdMax").text = "4.1"
    ET.SubElement(t2, "dayLengthMin").text = "10"
    ET.SubElement(t2, "dayLengthMax").text = "11"
    ET.SubElement(t2, "tempMin").text = "0"
    ET.SubElement(t2, "tempMax").text = "7"

    # initialization - forect initial state
    init = ET.SubElement(model, "initialization")
    ET.SubElement(init, "mode").text = "single"   # naËti 1 soubor pro celou krajinu
    ET.SubElement(init, "type").text = "single"   # form·t = jednotlivÈ stromy (ne DBH distribuce)
    ET.SubElement(init, "file").text = str(initial_forest)

    # outputs
    output = ET.SubElement(model, "output")

    landscape = ET.SubElement(output, "landscape")
    ET.SubElement(landscape, "enabled").text = "true"
    
    stand = ET.SubElement(output, "stand")
    ET.SubElement(stand, "enabled").text = "true"

    tree = ET.SubElement(output, "tree")
    ET.SubElement(tree, "enabled").text = "true"
    ET.SubElement(tree, "interval").text = "1"
    ET.SubElement(tree, "outputType").text = "sqlite"  # aù to pad· do SQLite
    
    # climate
    climate = ET.SubElement(model, "climate")
    ET.SubElement(climate, "tableName").text = str(site_name)
    ET.SubElement(climate, "co2concentration").text = str(380)

    # --- output ---
    output_main = ET.SubElement(iland, "output")

    # landscape including removed
    output_landscape_main = ET.SubElement(output_main, "landscape")
    ET.SubElement(output_landscape_main, "enabled").text = "true"
    output_landscape_death_main = ET.SubElement(output_main, "landscape_removed")
    ET.SubElement(output_landscape_death_main, "enabled").text = "true"

    # stand
    output_stand_main = ET.SubElement(output_main, "stand")
    ET.SubElement(output_stand_main, "enabled").text = "true"
    output_stand_death_main = ET.SubElement(output_main, "standdead")
    ET.SubElement(output_stand_death_main, "enabled").text = "true"

    # tree
    output_tree_main = ET.SubElement(output_main, "tree")
    ET.SubElement(output_tree_main, "enabled").text = "true"
    output_tree_death_main = ET.SubElement(output_main, "treeremoved")
    ET.SubElement(output_tree_death_main, "enabled").text = "true"

    # nutno definovat modules - specificky vitr

    rough = ET.tostring(iland, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    pretty_xml = reparsed.toprettyxml(indent="  ", encoding="utf-8")

    xml_declaration: bool = True

    out_path = Path(f"iLand/{site_name}.xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(pretty_xml)

    return out_path
