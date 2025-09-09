import sqlite3
from pathlib import Path
import textwrap

# ⬇️ UPRAV SEM cestu k souboru s klimatem
DB_PATH = Path(r"C:\Users\kaspar\source\repos\iLand_for_3DForest\iLand_for_3DForest\iLand\database\climate_file_Krivoklat.sqlite")

assert DB_PATH.exists(), f"Soubor neexistuje: {DB_PATH}"

with sqlite3.connect(DB_PATH) as con:
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1) seznam tabulek a pohledů
    cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    objects = cur.fetchall()
    if not objects:
        print("V databázi nejsou žádné tabulky/pohledy.")
    else:
        print("Nalezené tabulky/pohledy:")
        for obj in objects:
            print(f"  - {obj['name']} ({obj['type']})")

        print("\nDetaily:")
        for obj in objects:
            name = obj["name"]
            # 2) počet řádků
            try:
                cur.execute(f"SELECT COUNT(*) AS n FROM '{name}';")
                n = cur.fetchone()["n"]
            except sqlite3.DatabaseError as e:
                n = f"chyba: {e}"

            # 3) ukázkové řádky
            sample_txt = ""
            try:
                cur.execute(f"SELECT * FROM '{name}' LIMIT 3;")
                rows = cur.fetchall()
                if rows:
                    cols = rows[0].keys()
                    header = " | ".join(cols)
                    lines = []
                    for r in rows:
                        lines.append(" | ".join(str(r[c]) for c in cols))
                    sample_txt = header + "\n" + "\n".join(lines)
                else:
                    sample_txt = "(prázdná tabulka)"
            except sqlite3.DatabaseError as e:
                sample_txt = f"(nelze načíst: {e})"

            print(f"\n— {name} — řádků: {n}")
            print(textwrap.indent(sample_txt, "  "))
