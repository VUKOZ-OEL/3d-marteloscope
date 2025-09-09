import os
import pathlib
import subprocess
from typing import Dict, Any

from climate_utils import extract_site_climate
from soil_utils import extract_soil_texture_data
from soil_utils import extract_soil_chemistry_data
# from data_preparation import write_sql_file
from data_preparation import write_sqlite_climate_hard
from xml_file_utils import write_iland_minimal_project_xml

def prepare_iland_files(
    site_name : str,
    lat = float,
    lon = float,
    initial_date = int,
    climate_scenario = str,
    initial_forest_file = str
):
    print("Preparing climate dataset")
    site_climate_data = extract_site_climate(lat, lon, climate_scenario)

    print("Extracting soil data")
    site_soil_texture_data = extract_soil_texture_data(lat, lon, site_name)
    site_soil_chemistry_data = extract_soil_chemistry_data(lat, lon, site_name)

    print("Preparing files for iLand runs")
    print("...writing climate dataset")
    # ok = write_sql_file(
    #      climate_df = site_climate_data,
    #      output_folder = "iLand/database",
    #      output_file = f"climate_file_{site_name}.sqlite",
    #      site_name = site_name,
    #      since = 2025
    # )

    write_sqlite_climate_hard(
        climate_df = site_climate_data,
        site_name = site_name,
        since=initial_date
    )

    print("...creating project file")
    write_iland_minimal_project_xml(
        site_name = site_name,
        initial_forest = initial_forest_file,
        lat = lat,
        soil_texture = site_soil_texture_data,
        soil_chemistry = site_soil_chemistry_data
    )

def run_iland(
    project_file: str,
    base_out: str,
    rep: int,
    years: int = 30,
) -> Dict[str, Any]:
    # absolutní cesty + forward slashe
    iland_path   = pathlib.Path("iLand/ilandc.exe").expanduser().resolve()
    project_file = pathlib.Path(project_file).expanduser().resolve()
    base_out     = pathlib.Path(base_out).expanduser().resolve()

    if not iland_path.is_file():
        raise FileNotFoundError(f"ilandc.exe nenalezen: {iland_path}")
    if not project_file.is_file():
        raise FileNotFoundError(f"Projekt XML nenalezen: {project_file}")

    # složky
    base_out.mkdir(parents=True, exist_ok=True)
    outputs_dir = base_out / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = base_out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # názvy souborů (absolutní cesty -> forward slashe; db pouze název)
    db_name   = f"output_{rep}.sqlite"
    db_file   = (outputs_dir / db_name).resolve()
    iland_log = (logs_dir / f"iland_internal_{rep}.log").resolve()
    r_log     = (base_out / "run_stdout_stderr.log").resolve()

    # helper pro citování a forward slashe (jako R::shQuote + winslash="/")
    def qq(p: pathlib.Path) -> str:
        return f'"{p.as_posix()}"'

    project_cwd = project_file.parent

    # iLand argumenty (pozice: projekt, roky, klíče)
    args = [
        str(iland_path),
        project_file.as_posix(),           # projekt (XML)
        str(years),                        # roky
        f'system.path.home="{project_cwd.as_posix()}"',   # KOŘEN PRO REL. CESTY
        # !!! NEPŘEPISOVAT system.path.database NA outputs !!!
        f'system.database.out="{db_file.as_posix()}"',    # ABSOLUTNÍ CESTA VÝSTUPU
        f'system.logging.logFile="{iland_log.as_posix()}"'
    ]

    old_cwd = os.getcwd()
    try:
        os.chdir(project_cwd)
        with open(r_log, "a", encoding="utf-8", errors="replace") as logf:
            logf.write(f"\n=== RUN rep={rep}, years={years} ===\n")
            logf.write(f"CMD: {' '.join(args)}\n")
            proc = subprocess.run(args, stdout=logf, stderr=logf, check=False)
            status = proc.returncode
            logf.write(f"STATUS: {status}\n")
    finally:
        os.chdir(old_cwd)

    return {
        "rep": rep,
        "status": status,
        "db": str(db_file),
        "iland_log": str(iland_log),
        "stdout_stderr_log": str(r_log),
    }

def run_multiple_iland(
    project_file: str,
    base_out: str,
    years: int,
    n_reps: int = 100
):
    step = 10  # velikost bloku
    for start in range(1, n_reps + 1, step):
        end = min(start + step - 1, n_reps)
        print(f"...running iLand simulation {start}-{end} from {n_reps}", flush=True)
        for i in range(start, end + 1):
            run_iland(
                project_file=project_file,
                base_out=base_out,
                rep=i,
                years=years
            )