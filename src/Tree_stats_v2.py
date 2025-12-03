def load_project_json(file_path: str, exclude_from_sql_update: List[str] = None) -> pd.DataFrame:
    """
    Načte JSON, zpracuje atributy a barvy, a synchronizuje s SQLite.
    Hodnoty z JSONu (species, barvy) mají přednost a nejsou přepisovány z SQL.
    """
    if exclude_from_sql_update is None:
        exclude_from_sql_update = []

    # Tyto sloupce jsou "svaté" - pochází z logiky JSONu a SQL je nesmí nikdy přepsat.
    protected_cols = [
        "species", 
        "speciesColorHex", 
        "management_status", 
        "managementColorHex",
        "label" # Label často chceme taky zachovat z JSONu, pokud to není v zadání jinak
    ]
    
    # Rozšíříme seznam výjimek
    exclude_final = list(set(exclude_from_sql_update + protected_cols))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Soubor {file_path} nebyl nalezen.")

    # --- 1. Načtení a parsování JSON ---
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # Lookupy s vynuceným int klíčem
    species_list = data.get("species", [])
    sp_id_map = {}
    for item in species_list:
        try: sp_id_map[int(item.get("id"))] = {"latin": item.get("latin", "Unknown"), "color": _to_hex(item.get("color"))}
        except: pass

    mgmt_list = data.get("managementStatus", [])
    mg_id_map = {}
    for item in mgmt_list:
        try: mg_id_map[int(item.get("id"))] = {"label": item.get("label", "Unknown"), "color": _to_hex(item.get("color"))}
        except: pass

    # Zpracování segmentů
    segments = data.get("segments", [])
    rows = []
    for seg in segments:
        base = {}
        base["id"] = seg.get("id")
        base["label"] = seg.get("label", "")

        # IDs
        try: s_id = int(seg.get("speciesId", -1))
        except: s_id = -1
        try: m_id = int(seg.get("managementStatusId", -1))
        except: m_id = -1

        # Data z lookupů
        sp = sp_id_map.get(s_id, {})
        mg = mg_id_map.get(m_id, {})

        base["species"] = sp.get("latin", "Unknown")
        base["speciesColorHex"] = sp.get("color") # Tady musí být HEX nebo None
        
        base["management_status"] = mg.get("label", "Unknown")
        base["managementColorHex"] = mg.get("color")

        SKIP_ATTRS = {"speciesColorHex", "managementColorHex", "species", "management_status"}
        # Atributy
        attrs = seg.get("treeAttributes", {}) or {}
        print(attrs.items())
        for k, v in attrs.items():
            if k == "position":
                continue
            if k in SKIP_ATTRS:
                continue    # ← TADY ZABRÁNÍME PŘEPISU
            base[k] = v

        # Pozice
        pos = attrs.get("position")
        if isinstance(pos, list) and len(pos) >= 2:
            base["x"], base["y"] = float(pos[0]), float(pos[1])
            if len(pos) > 2: base["z"] = float(pos[2])
        else:
            base["x"], base["y"] = 0.0, 0.0
        rows.append(base)

    print("ROWS:")
    #print(rows)
    # Vytvoření DataFrame
    df_json = pd.DataFrame(rows)
    
    if df_json.empty:
        return df_json

    # Nastavení indexu a typů
    df_json['id'] = df_json['id'].astype(int)
    df_json.set_index('id', inplace=True, drop=False)

    # Filtrace (Ground / Unsegmented)
    if "label" in df_json.columns:
        mask = df_json["label"].astype(str).str.contains("ground|unsegmented", case=False, na=False)
        df_json = df_json.loc[~mask]

    # --- 2. SQLite Logika ---
    sqlite_path = os.path.splitext(file_path)[0] + ".sqlite"
    table_name = "tree"

    # A) Pokud neexistuje -> Vytvořit
    if not os.path.exists(sqlite_path):
        try:
            conn = sqlite3.connect(sqlite_path)
            # Uložíme jen základní sloupce
            cols_to_save = [c for c in ["id", "label", "species"] if c in df_json.columns]
            df_json[cols_to_save].to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"SQLite vytvořeno: {sqlite_path}")
            conn.close()
        except Exception as e:
            print(f"Chyba při tvorbě SQLite: {e}")
    
    # B) Pokud existuje -> Merge (Update)
    else:
        try:
            conn = sqlite3.connect(sqlite_path)
            # Ověření tabulky
            if pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';", conn).empty:
                conn.close()
                return df_json.reset_index(drop=True)

            # Načtení SQL dat
            df_sql = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()

            if not df_sql.empty:
                # Sjednotit index
                df_sql['id'] = df_sql['id'].astype(int)
                df_sql.set_index('id', inplace=True, drop=False)

                # 1. Identifikace sloupců pro update
                # Vezmeme sloupce z SQL, ale VYŘADÍME ty, které jsou v exclude_final (barvy, species, atd.)
                # Tím zajistíme, že i kdyby v SQL sloupec 'speciesColorHex' byl (a byl prázdný), nepoužije se.
                update_cols = [c for c in df_sql.columns if c in df_json.columns and c not in exclude_final]
                
                if update_cols:
                    # Update přepíše hodnoty v df_json hodnotami z df_sql (tam kde se shoduje ID)
                    df_json.update(df_sql[update_cols])

                # 2. Identifikace nových sloupců (které jsou v SQL, ale ne v JSON)
                # Např. uživatelské poznámky přidané v aplikaci
                new_cols = [c for c in df_sql.columns if c not in df_json.columns]
                
                if new_cols:
                    df_json = df_json.join(df_sql[new_cols])

        except Exception as e:
            print(f"Chyba při merge SQLite: {e}")

    return df_json.reset_index(drop=True)
