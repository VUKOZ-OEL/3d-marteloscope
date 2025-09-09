import os, math, sqlite3
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy import types as satypes  # ← DŮLEŽITÉ

def write_sql_file_old(climate_df: pd.DataFrame,
                   output_folder: str,
                   output_file: str,
                   site_name: str,
                   since: float = -math.inf) -> bool:
    """
    Zapíše iLand-klima tabulku do SQLite.
    - climate_df musí mít sloupce: year, month, day, min_temp, max_temp, prec, rad, vpd
    - tabulka se jmenuje podle `site_name`
    - existující tabulka bude přepsána (if_exists='replace')
    """
    required = ["year", "month", "day", "min_temp", "max_temp", "prec", "rad", "vpd"]
    missing = [c for c in required if c not in climate_df.columns]
    if missing:
        raise ValueError(f"Vstupní data postrádají sloupce: {', '.join(missing)}")

    # jen požadované sloupce + filtr od roku `since`
    df = climate_df.loc[:, required].copy()
    df = df[df["year"] >= since]

    # přetypování (co nejblíže R verzi)
    df["year"]     = df["year"].astype("int64", copy=False, errors="ignore")
    df["month"]    = df["month"].astype("int64", copy=False, errors="ignore")
    df["day"]      = df["day"].astype("int64", copy=False, errors="ignore")
    df["min_temp"] = pd.to_numeric(df["min_temp"], errors="coerce")
    df["max_temp"] = pd.to_numeric(df["max_temp"], errors="coerce")
    df["prec"]     = pd.to_numeric(df["prec"], errors="coerce")
    df["rad"]      = pd.to_numeric(df["rad"], errors="coerce")
    df["vpd"]      = pd.to_numeric(df["vpd"], errors="coerce")

    # vytvoření cílové složky
    os.makedirs(output_folder, exist_ok=True)

    # zápis do SQLite (overwrite)
    db_path = os.path.join(output_folder, output_file)
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:  # automatický commit/rollback
        df.to_sql(site_name, conn, if_exists="replace", index=False)

    return True

def write_sqlite_climate_hard(climate_df: pd.DataFrame, 
                              site_name: str, 
                              since: float = -math.inf):
    """
    Zapíše klimatická data pro iLand do SQLite.
    - Tabulka se jmenuje přesně `site_name`
    - Soubor se uloží do 'iLand/database/climate_file_{site_name}.sqlite'
    - Vstupní df musí mít sloupce: year, month, day, min_temp, max_temp, prec, rad, vpd
    """

    # cílový soubor dle konvence
    output_route = Path("iLand") / "database" / f"climate_file_{site_name}.sqlite"

    # --- validace vstupu ---
    required = ["year","month","day","min_temp","max_temp","prec","rad","vpd"]
    df = climate_df.copy()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    # jen požadované sloupce + filtr 'since'
    df = df.loc[df["year"] >= since, required].copy()

    # tvrdé typování + základní kontroly
    for c in ["year","month","day"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
    for c in ["min_temp","max_temp","prec","rad","vpd"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)

    if df.empty:
        raise ValueError("Po filtru `since` je dataset prázdný.")
    if not df["month"].between(1,12).all() or not df["day"].between(1,31).all():
        raise ValueError("Mimo rozsah: month (1–12) / day (1–31).")

    # připrav cílovou složku
    out = output_route.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # schéma tabulky = přesně jak iLand očekává
    schema = f"""
    DROP TABLE IF EXISTS "{site_name}";
    CREATE TABLE "{site_name}" (
        year     INTEGER NOT NULL,
        month    INTEGER NOT NULL,
        day      INTEGER NOT NULL,
        min_temp REAL    NOT NULL,
        max_temp REAL    NOT NULL,
        prec     REAL    NOT NULL,
        rad      REAL    NOT NULL,
        vpd      REAL    NOT NULL,
        PRIMARY KEY (year, month, day)
    );
    """

    # zápis + rychlá verifikace
    with sqlite3.connect(str(out)) as con:
        con.executescript(schema)
        con.executemany(
            f'INSERT INTO "{site_name}" (year,month,day,min_temp,max_temp,prec,rad,vpd) VALUES (?,?,?,?,?,?,?,?)',
            df.itertuples(index=False, name=None)
        )
        con.commit()

        tabs  = con.execute("SELECT type,name FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;").fetchall()
        info  = con.execute(f'PRAGMA table_info("{site_name}")').fetchall()
        nrows = con.execute(f'SELECT COUNT(*) FROM "{site_name}"').fetchone()[0]
        head  = con.execute(f'SELECT * FROM "{site_name}" ORDER BY year,month,day LIMIT 3').fetchall()
        tail  = con.execute(f'SELECT * FROM "{site_name}" ORDER BY year DESC,month DESC,day DESC LIMIT 3').fetchall()

    # testovani spravnosti outputu - neni normalne potreba
    # print("=== iLand climate DB sanity check ===")
    # print("DB path:", str(out))                  # musí sedět s logem iLandu
    # print("Tables :", tabs)                      # ('table','<site_name>')
    # print(f"Schema {site_name}:", info)         # year..vpd
    # print("Rows   :", nrows)                     # >0
    # print("Head   :", head)
    # print("Tail   :", tail)

def write_sql_file(climate_df: pd.DataFrame,
                   output_folder: str,
                   output_file: str,
                   site_name: str,
                   since: float = -math.inf,
                   verify: bool = True) -> bool:
    """
    Zapíše iLand-klima tabulku do SQLite tak, jak ji iLand očekává:
    year, month, day, min_temp, max_temp, prec, rad, vpd
    """
    df = climate_df.copy()
    required = ["year","month","day","min_temp","max_temp","prec","rad","vpd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Chybí sloupce: {missing}")

    # filtr a pořadí
    df = df.loc[df["year"] >= since, required].copy()

    # tvrdé typování
    for c in ["year","month","day"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    for c in ["min_temp","max_temp","prec","rad","vpd"]:
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)

    if df.empty:
        raise ValueError("Po filtru `since` je dataset prázdný – iLand pak klima přeskočí.")
    if df.isna().any().any():
        raise ValueError("NaN/NULL ve vstupu – oprav data před zápisem.")

    # explicitní SQLAlchemy typy
    dtype_map = {
        "year":     satypes.Integer(),
        "month":    satypes.Integer(),
        "day":      satypes.Integer(),
        "min_temp": satypes.Float(),
        "max_temp": satypes.Float(),
        "prec":     satypes.Float(),
        "rad":      satypes.Float(),
        "vpd":      satypes.Float(),
    }

    os.makedirs(output_folder, exist_ok=True)
    db_path = os.path.abspath(os.path.join(output_folder, output_file))

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        df.to_sql(site_name, conn, if_exists="replace", index=False, dtype=dtype_map)

    # rychlá verifikace té SAMÉ DB a tabulky
    if verify:
        with sqlite3.connect(db_path) as con:
            tabs = con.execute("SELECT type,name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
            info = con.execute(f'PRAGMA table_info("{site_name}")').fetchall()
            n    = con.execute(f'SELECT COUNT(*) FROM "{site_name}"').fetchone()[0]
        print("DB:", db_path)
        print("Tables:", tabs)
        print(f"Schema {site_name}:", info)
        print("Rows:", n)
        if n == 0:
            raise RuntimeError("Tabulka existuje, ale je prázdná.")

    return True