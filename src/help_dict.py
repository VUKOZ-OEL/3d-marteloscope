# help_dict.py
from typing import Literal


Lang = Literal["cs", "en"]

HELP_I18N: dict[str, dict[Lang, str]] = {
    # help_dict.py

    "dashboard_help": {
        "cs": """
        ### Přehled a zásahy (Dashboard)

        Tato stránka slouží k **práci se zásahy (managementem)** nad aktuální plochou
        a k **exportu přehledového reportu**.

        ---

        ## 🧭 Základní přehled lokality
        V levé části stránky vidíte souhrnné informace o ploše:
        - typ lesa
        - počet stromů
        - objem dřeva
        - rozlohu
        - klimatické a lokalizační údaje

        Tyto hodnoty se **nemění se zásahy** – slouží jako kontext.

        ---

        ## 🌲 Zásahy (management)

        Zásah určuje, **jaký pěstební režim má každý strom** (např. cílový strom, bez zásahu).

        ### Typy zásahů:
        - **Uživatelský zásah** – aktuální, zatím neuložený stav
        - **Uložené zásahy** – dříve uložené scénáře v databázi projektu

        Pomocí rozbalovacího seznamu lze mezi zásahy **přepínat**.

        ---

        ## 🔁 Přepínání zásahů
        - Tlačítko **Načíst příklad** aplikuje vybraný zásah na stromy
        - Při návratu k uživatelskému zásahu se obnoví poslední lokální změny

        💡 Uživatelský zásah je automaticky cacheován.

        ---

        ## 💾 Uložení zásahu
        Aktuální uživatelský zásah lze uložit:
        1. Zadejte název zásahu
        2. Klikněte na **Uložit zásah do projektu**

        Uložený zásah:
        - je uložen v SQLite databázi projektu
        - lze jej znovu načíst nebo použít v simulacích

        ---

        ## 🗑️ Mazání zásahu
        - Uložený zásah lze **trvale smazat**
        - Uživatelský zásah lze **vymazat** a vrátit se k čistému stavu

        ⚠️ Smazání uloženého zásahu je nevratné.

        ---

        ## 📄 Export reportu
        Tlačítko **Exportovat výsledky**:
        - vygeneruje PDF report
        - obsahuje souhrn plochy, zásahu a grafy
        - jazyk reportu odpovídá zvolenému jazyku aplikace

        Po vygenerování se zobrazí tlačítko ke stažení.

        ---

        ## 📊 Interpretace výsledků
        Report i přehledy:
        - pracují **na hektar**
        - reflektují **aktuální zásah**
        - slouží k porovnání scénářů

        ---

        ## 💡 Tipy
        - Zásahy pojmenovávejte jednoznačně (např. „Probírka 2025“)
        - Před exportem reportu zkontrolujte aktivní zásah
        - Uživatel může mít vždy právě jeden „živý“ uživatelský zásah
        """,

                "en": """
        ### Dashboard & Management

        This page is used to **manage interventions (management scenarios)**
        and to **export summary reports** for the current plot.

        ---

        ## 🧭 Site overview
        The left panel shows static information about the plot:
        - forest type
        - number of trees
        - wood volume
        - area
        - climate and location

        These values provide context and **do not change with interventions**.

        ---

        ## 🌲 Management interventions
        An intervention defines **how each tree is treated**
        (e.g. target tree, untouched).

        ### Types of interventions:
        - **User intervention** – current, not yet saved
        - **Saved interventions** – scenarios stored in the project database

        You can switch between them using the dropdown.

        ---

        ## 🔁 Switching interventions
        - **Load example** applies the selected intervention
        - Returning to the user intervention restores the last local edits

        💡 The user intervention is cached automatically.

        ---

        ## 💾 Saving an intervention
        To save the current user intervention:
        1. Enter a name
        2. Click **Save intervention to project**

        Saved interventions:
        - are stored in the project SQLite database
        - can be reused or simulated later

        ---

        ## 🗑️ Deleting interventions
        - Saved interventions can be permanently deleted
        - The user intervention can be cleared to reset the state

        ⚠️ Deleting a saved intervention is irreversible.

        ---

        ## 📄 Export report
        **Export results** generates a PDF report containing:
        - plot overview
        - active intervention
        - summary figures

        The report language matches the app language.

        ---

        ## 📊 Interpreting results
        Reports and summaries:
        - are normalized per hectare
        - reflect the active intervention
        - are intended for scenario comparison

        ---

        ## 💡 Tips
        - Use clear intervention names (e.g. “Thinning 2025”)
        - Always check which intervention is active before exporting
        - Only one user intervention exists at a time
        """
    },

    "summary_help": {
        "cs": """
            #### Souhrn porostu – přehled grafů
            Tato stránka poskytuje přehled struktury porostu a umožňuje porovnání stavu před zásahem, po zásahu a vytěžených stromů. Všechny hodnoty jsou přepočteny na hektar.

            #### Ovládání
            ##### Zobrazit data pro
            Volba určuje, které stromy jsou zahrnuty:
            - Před – všechny stromy před zásahem
            - Po – ponechané stromy (Target tree + Untouched)
            - Vytěženo – odstraněné stromy

            ##### Součty podle
            Určuje zobrazovanou veličinu:
            - Počet stromů
            - Objem dřeva (m³)
            - Výčetní kruhová základna (m²)
            - Zápoj porostu (%)
            - Zakmenění (%), pokud je k dispozici referenční tabulka

            ##### Barvy podle
            Určuje rozdělení hodnot:
            - Dřevina – podle druhů stromů
            - Zásah – podle kategorií managementu  
            Při zobrazení zakmenění je tato volba nedostupná.

            #### Grafy
            ##### Koláčový graf
            Zobrazuje celkové složení porostu podle zvolené metriky.
            - Střed koláče obsahuje součet (Σ)
            - U zápoje a zakmenění je dopočten prázdný podíl do 100 %
            - Procenta vyjadřují relativní zastoupení

            ##### Sloupcové grafy
            Tloušťkové (DBH) a výškové třídy zobrazují rozdělení stromů do tříd.
            - Sloupce jsou skládané
            - Výška sloupce odpovídá součtu hodnot ve třídě
            - Barvy odpovídají zvolenému režimu

            #### Interaktivita a interpretace
            Najetím myši se zobrazí hodnota kategorie a součet celé třídy. Pomocí legendy lze dočasně skrývat jednotlivé kategorie. Porovnání Před × Po ukazuje efekt zásahu, Po × Vytěženo jeho intenzitu a selektivnost.
            """,

                "en": """
            #### Stand summary – overview charts
            This page provides an overview of stand structure and allows comparison of conditions before intervention, after intervention, and removed trees. All values are normalized per hectare.

            #### Controls
            ##### Show data for
            Select which trees are included:
            - Before – all trees before intervention
            - After – remaining trees (Target tree + Untouched)
            - Removed – harvested trees

            ##### Sum values by
            Defines the displayed metric:
            - Tree count
            - Wood volume (m³)
            - Basal area (m²)
            - Canopy cover (%)
            - Stocking (%), if a reference table is available

            ##### Color by
            Defines how values are grouped:
            - Species – by tree species
            - Management – by management categories  
            This option is disabled when stocking is selected.

            #### Charts
            ##### Pie chart
            Shows overall stand composition for the selected metric.
            - The center displays the total (Σ)
            - For canopy cover and stocking, the missing share to 100 % is added
            - Percentages express relative contribution

            ##### Bar charts
            Diameter (DBH) and height classes show distribution across classes.
            - Bars are stacked
            - Bar height equals the sum within the class
            - Colors follow the selected grouping

            #### Interactivity and interpretation
            Hovering displays category values and total class sums. The legend allows temporary hiding of categories. Comparing Before × After shows intervention effects, while After × Removed indicates intensity and selectivity.
            """
            },

    "intensity_help": {
        "cs": """
        ### Intenzita pěstebních zásahů

        Tato stránka slouží k hodnocení **rozsahu a struktury těžby** vzhledem
        k celkovému porostu i jednotlivým skupinám stromů.

        Zobrazené hodnoty jsou vždy vyjádřeny **v procentech (%)**.

        ---

        ## Ovládání

        **Intenzita podle**  
        Určuje metriku, vůči které se intenzita zásahu počítá:
        - počet stromů
        - objem dřeva
        - výčetní kruhová základna
        - objem koruny

        **Vykreslit podle**  
        Určuje hlavní skupiny na ose Y:
        - dřeviny
        - zásah (management)

        **Filtry DBH a výšky**  
        Omezují vstupní soubor stromů, ze kterého se intenzita počítá.

        ---

        ## Grafy

        ### Odstranění z celku
        Levý graf ukazuje, **jaký podíl celkového porostu byl odstraněn**.
        - Každý řádek představuje skupinu (dřevinu nebo zásah)
        - Skládané barvy ukazují strukturu odstranění
        - Řádek „Suma“ shrnuje celkovou intenzitu zásahu

        ---

        ### Intenzita uvnitř skupiny
        Pravý graf zobrazuje, **jak velká část každé skupiny byla odstraněna**.
        - Hodnoty jsou vztaženy k velikosti dané skupiny
        - Graf umožňuje porovnat selektivitu zásahu

        ---

        ## Interpretace

        - Vysoké hodnoty v levém grafu znamenají **silný vliv na celý porost**
        - Vysoké hodnoty v pravém grafu znamenají **intenzivní zásah do konkrétní skupiny**
        - Rozdíl mezi grafy ukazuje, zda byl zásah plošný nebo cílený

        ---

        ## Poznámka
        Pokud nejsou po aplikaci filtrů k dispozici žádná relevantní data,
        grafy se zobrazí prázdné a aplikace na to upozorní.
        """,

            "en": """
        ### Intensity of silvicultural intervention

        This page evaluates the **extent and structure of tree removal**
        relative to the entire stand and individual groups.

        All values are expressed **in percent (%)**.

        ---

        ## Controls

        **Intensity based on**  
        Defines the metric used to compute intervention intensity:
        - tree count
        - wood volume
        - basal area
        - crown volume

        **Plot by**  
        Defines the main grouping on the Y-axis:
        - species
        - management

        **DBH and height filters**  
        Limit the set of trees used for the calculation.

        ---

        ## Charts

        ### Removal from total
        The left chart shows **the share of the total stand that was removed**.
        - Each row represents a group (species or management)
        - Stacked bars show the composition of removal
        - The “Sum” row summarizes overall intervention intensity

        ---

        ### Intensity within group
        The right chart shows **the proportion removed within each group**.
        - Values are relative to the size of the group itself
        - Useful for assessing selectivity of the intervention

        ---

        ## Interpretation

        - High values in the left chart indicate **strong impact on the stand**
        - High values in the right chart indicate **intensive removal within a group**
        - Differences between charts reveal whether the intervention was uniform or targeted

        ---

        ## Note
        If no meaningful data remain after filtering,
        empty charts are shown and a warning is displayed.
        """
        },

    "tree_stats_help": {
        "cs": """
        ### Statistiky stromů

        Tato stránka umožňuje detailně analyzovat rozdělení vlastností stromů
        před zásahem, po zásahu a u odstraněných stromů.

        Zobrazení se automaticky přizpůsobuje zvolené metrice.

        ---

        ## Ovládání

        **Zobrazované hodnoty**  
        Určuje proměnnou, která se bude analyzovat (např. DBH, výška, objem, zápoj).
        Volba *Počet stromů* aktivuje sloupcové grafy, ostatní metriky houslové grafy.

        **Vykreslit podle**  
        Určuje způsob rozdělení dat:
        - podle kategorií (dřevina / zásah)
        - podle tloušťkových tříd (DBH)
        - podle výškových tříd  
        (Třídění je dostupné pouze pro počet stromů.)

        **Barvy podle**  
        Určuje, zda jsou grafy barevně rozlišeny podle dřevin nebo zásahu.

        **Filtry DBH a výšky**  
        Omezují množinu stromů vstupujících do výpočtu statistik.

        ---

        ## Typy grafů

        ### Sloupcové grafy (Počet stromů)
        Zobrazeny jsou tři panely:
        - stav před zásahem
        - stav po zásahu
        - odstraněné stromy  

        Sloupce mohou být seskupené nebo skládané.
        Osa X odpovídá kategoriím nebo třídám, osa Y počtu stromů.

        ---

        ### Houslové grafy (spojité metriky)
        Používají se pro všechny ostatní proměnné.
        Každý panel zobrazuje rozdělení hodnot:
        - šířka odpovídá hustotě dat
        - box ukazuje medián a kvartily
        - čára znázorňuje průměr

        Grafy umožňují porovnat nejen střední hodnoty,
        ale i rozptyl a tvar rozdělení.

        ---

        ### Projection Exposure (zápoj stromu)
        Tato metrika má speciální zobrazení:
        - před zásahem: původní zápoj všech stromů
        - po zásahu: zápoj ponechaných stromů
        - odstraněné stromy: původní zápoj odstraněných jedinců  

        Umožňuje posoudit, jak zásah ovlivnil světelné podmínky porostu.

        ---

        ## Interpretace

        - Rozdíly mezi panely ukazují efekt zásahu
        - Změna tvaru rozdělení signalizuje selektivitu zásahu
        - Posun mediánu ukazuje změnu typických hodnot
        - Šířka houslí odhaluje heterogenitu porostu

        ---

        ## Poznámka
        Pokud po aplikaci filtrů nejsou k dispozici žádná platná data,
        grafy se nezobrazí a aplikace na to upozorní.
        """,

            "en": """
        ### Tree statistics

        This page provides a detailed analysis of tree attributes
        before management, after management, and for removed trees.

        The visualization adapts automatically to the selected metric.

        ---

        ## Controls

        **Values to plot**  
        Selects the variable to analyze (e.g. DBH, height, volume, projection exposure).
        *Tree count* activates bar charts; other metrics use violin plots.

        **Plot by**  
        Defines how data are grouped:
        - by category (species / management)
        - by DBH classes
        - by height classes  
        (Class-based grouping is available only for tree count.)

        **Color by**  
        Controls whether colors represent species or management categories.

        **DBH and height filters**  
        Limit the set of trees included in the analysis.

        ---

        ## Chart types

        ### Bar charts (Tree count)
        Three panels are shown:
        - before management
        - after management
        - removed trees  

        Bars can be grouped or stacked.
        The X-axis shows categories or classes, the Y-axis tree counts.

        ---

        ### Violin plots (continuous metrics)
        Used for all other variables.
        Each panel displays the distribution of values:
        - width represents data density
        - box shows median and quartiles
        - line indicates the mean

        These plots allow comparison of both central tendency and variability.

        ---

        ### Projection Exposure
        This metric uses a dedicated layout:
        - before: original exposure of all trees
        - after: exposure of retained trees
        - removed: original exposure of removed trees  

        It helps assess changes in canopy light conditions caused by management.

        ---

        ## Interpretation

        - Differences between panels indicate management effects
        - Changes in distribution shape reveal selectivity
        - Median shifts show changes in typical values
        - Violin width reflects stand heterogeneity

        ---

        ## Note
        If no valid data remain after applying filters,
        the charts are hidden and a warning is displayed.
        """
        },

    "map_help": {
        "cs": """
            ### Mapa stromů – prostorové zobrazení porostu

            Tato stránka zobrazuje **polohu jednotlivých stromů v ploše**
            a umožňuje vizuálně analyzovat strukturu porostu,
            zásah a prostorové vztahy mezi stromy.

            Každý bod odpovídá jednomu stromu v reálné poloze.

            ---

            ## Horní ovládací prvky

            **Zobrazit data pro**  
            Určuje, které stromy se zobrazí:
            - před zásahem
            - po zásahu
            - odstraněné stromy

            **Barvit podle**  
            Určuje barevné rozlišení bodů:
            - dřevina
            - zásah (management)

            **Velikost bodu podle**  
            Určuje, která proměnná ovlivňuje velikost bodů
            (např. DBH, výška, objem, zápoj, vlastnosti koruny).

            ---

            ## Levý panel – filtry a zobrazení

            **Zobrazit popisky**  
            Zapíná textové popisky stromů (ID / label).

            **Projekce korun**  
            Zobrazuje plošné projekce korun stromů jako polygony.

            **Invertovat barvy korun**  
            Přepíná barvení korun mezi dřevinou a managementem
            nezávisle na barvě bodů.

            **Filtry DBH a výšky**  
            Omezují zobrazené stromy podle tloušťky a výšky.

            **Min / max velikost bodu**  
            Určuje rozsah velikostí bodů ve vizualizaci.

            ---

            ## Mapa

            - Body jsou vykresleny v měřítku 1 : 1
            - Zobrazená mřížka má krok 10 metrů
            - Poměr os je fixní (nedochází ke zkreslení vzdáleností)
            - Mapu lze posouvat tažením myši

            ---

            ## Interaktivita

            - Najetím myši na bod se zobrazí detail stromu
            - Legenda umožňuje skrývat / zobrazovat jednotlivé skupiny
            - Projekce korun slouží pouze jako kontext (bez tooltipů)

            ---

            ## Interpretace

            - Velikost bodů odráží zvolenou proměnnou
            - Barvy umožňují sledovat prostorové rozložení dřevin nebo zásahů
            - Projekce korun pomáhají posoudit překryv a zapojení porostu
            - Filtry umožňují cíleně analyzovat konkrétní část stromové populace

            ---

            ## Poznámka
            Pokud filtry vyloučí všechny stromy,
            mapa se nezobrazí a aplikace na to upozorní.
            """,

            "en": """
            ### Tree map – spatial stand visualization

            This page displays the **spatial position of individual trees**
            and allows visual analysis of stand structure,
            management intervention, and spatial relationships.

            Each point represents one tree at its real location.

            ---

            ## Top controls

            **Show data for**  
            Selects which trees are displayed:
            - before intervention
            - after intervention
            - removed trees

            **Color by**  
            Defines how points are colored:
            - species
            - management category

            **Scale point size by**  
            Selects the variable controlling point size
            (e.g. DBH, height, volume, canopy metrics).

            ---

            ## Left panel – filters and display

            **Show labels**  
            Displays tree labels (ID / name).

            **Crown projections**  
            Displays planar crown projections as polygons.

            **Invert crown colors**  
            Switches crown coloring between species and management
            independently of point colors.

            **DBH and height filters**  
            Restrict displayed trees by diameter and height.

            **Min / max point size**  
            Defines the visual size range of points.

            ---

            ## Map view

            - Points are shown at true spatial scale
            - Grid spacing is 10 meters
            - Axes are locked to preserve distances
            - The map can be panned by dragging

            ---

            ## Interactivity

            - Hovering over a point shows tree details
            - The legend allows hiding/showing groups
            - Crown polygons are contextual only (no tooltips)

            ---

            ## Interpretation

            - Point size reflects the selected variable
            - Colors reveal spatial patterns of species or management
            - Crown projections help assess overlap and canopy closure
            - Filters enable focused analysis of selected tree subsets

            ---

            ## Note
            If all trees are filtered out,
            the map is hidden and a warning is shown.
            """
        },

    "heatmap_help": {
        "cs": """
            ### Prostorové heatmapy – hustota a rozložení atributů

            Tato stránka zobrazuje **prostorové rozložení stromů nebo jejich vlastností**
            pomocí XY heatmap ve třech panelech:
            - před zásahem
            - po zásahu
            - odstraněné stromy

            Heatmapy jsou založeny na pravidelné mřížce a zobrazují
            lokální koncentraci hodnot v ploše.

            ---

            ## Horní ovládání

            **Zobrazovaná veličina**  
            Určuje, co heatmapa vyjadřuje:
            - počet stromů
            - průměrné nebo součtové charakteristiky stromů
            (např. DBH, výška, objem, vlastnosti koruny)

            **Filtry DBH a výšky**  
            Omezují množinu stromů vstupujících do výpočtu heatmap.

            **Barvit podle**  
            Určuje barvy překryvných bodů stromů:
            - dřevina
            - zásah (management)

            **Zobrazit polohu stromů**  
            Překryje heatmapy body skutečných poloh stromů
            pro lepší prostorovou orientaci.

            ---

            ## Filtry specifické pro heatmapy

            **Výběr dřevin**  
            Omezuje výpočet heatmap pouze na vybrané druhy stromů.

            **Výběr zásahů**  
            Omezuje výpočet na vybrané kategorie managementu.

            Tyto filtry se týkají **pouze heatmap**,
            nikoli překryvných bodů stromů.

            ---

            ## Struktura grafů

            - Každý panel představuje stejnou plochu
            - Osy X a Y mají shodné měřítko
            - Barevná škála je společná pro všechny tři panely
            - Intenzita barvy odpovídá velikosti zvolené veličiny

            Hodnoty jsou počítány v buňkách pravidelné mřížky
            a jemně vyhlazeny pro lepší čitelnost.

            ---

            ## Interaktivita

            - Najetím myši se zobrazí hodnota v dané buňce
            - Legenda umožňuje skrývat / zobrazovat skupiny bodů
            - Překryvné body slouží jako referenční kontext

            ---

            ## Interpretace

            - Světlé oblasti znamenají nízké hodnoty nebo řídký výskyt
            - Tmavé oblasti ukazují koncentraci stromů nebo vysoké hodnoty
            - Rozdíly mezi panely ukazují prostorový efekt zásahu
            - Shodný rozsah barev umožňuje přímé porovnání stavů

            ---

            ## Poznámka
            Pokud po aplikaci filtrů nezůstanou žádná platná data,
            heatmapy se zobrazí prázdné a aplikace na to upozorní.
            """,

            "en": """
            ### Spatial heatmaps – density and attribute distribution

            This page visualizes the **spatial distribution of trees or their attributes**
            using XY heatmaps shown in three panels:
            - before intervention
            - after intervention
            - removed trees

            Heatmaps are computed on a regular grid and represent
            local concentrations of values across the plot.

            ---

            ## Top controls

            **Displayed value**  
            Selects what the heatmap represents:
            - tree count
            - aggregated tree attributes
            (e.g. DBH, height, volume, crown properties)

            **DBH and height filters**  
            Restrict the set of trees used to compute the heatmaps.

            **Color by**  
            Controls the colors of overlaid tree positions:
            - species
            - management category

            **Show tree positions**  
            Overlays actual tree locations on top of the heatmaps
            for spatial reference.

            ---

            ## Heatmap-specific filters

            **Species selection**  
            Limits heatmap calculation to selected species.

            **Management selection**  
            Limits heatmap calculation to selected management categories.

            These filters affect **only the heatmaps**,
            not the overlaid tree points.

            ---

            ## Chart structure

            - Each panel represents the same spatial extent
            - X and Y axes use identical scales
            - A shared color scale is used for all panels
            - Color intensity reflects the magnitude of the selected value

            Values are aggregated within grid cells
            and slightly smoothed for readability.

            ---

            ## Interactivity

            - Hovering shows the value of an individual grid cell
            - The legend allows hiding/showing point groups
            - Overlaid points serve as spatial context

            ---

            ## Interpretation

            - Light areas indicate low values or sparse occurrence
            - Dark areas highlight concentrations or high values
            - Differences between panels reveal spatial effects of management
            - A shared color range enables direct comparison

            ---

            ## Note
            If no valid data remain after filtering,
            the heatmaps appear empty and a warning is shown.
            """
        },

    "canopy_help": {
        "cs": """
            ### Vertikální profily objemu korun

            Tato stránka zobrazuje **rozložení objemu korun stromů po výškových vrstvách**
            a umožňuje porovnat strukturu korun:
            - před zásahem
            - po zásahu
            - u odstraněných stromů

            Hodnoty jsou agregovány po výškových metrech
            a přepočteny **na hektar**.

            ---

            ## Ovládání

            **Zobrazit hodnoty podle**  
            Určuje, podle jakých skupin se profily vykreslí:
            - dřeviny
            - zásah (management)

            Je možné zobrazit:
            - jednu skupinu (samostatné křivky),
            - nebo kombinaci dvou skupin jako překryv.

            První zvolená skupina tvoří hlavní profily,
            druhá (pokud je zvolena) je zobrazena jako doplňkový překryv.

            ---

            ## Struktura grafů

            Každý ze tří panelů zobrazuje stejný typ profilu:
            - levý panel: stav před zásahem
            - prostřední panel: stav po zásahu
            - pravý panel: odstraněné stromy

            Osa Y:
            - výška nad zemí (v metrech)

            Osa X:
            - objem korun ve výškové vrstvě (m³/ha)

            ---

            ## Křivky v grafech

            - Barevné křivky představují jednotlivé skupiny
            (dřeviny nebo management)
            - Tloušťka a styl čar odlišují hlavní a překryvné vrstvy
            - Tečkovaná šedá křivka „Suma“ ukazuje
            celkový objem korun ve výšce bez rozlišení skupin

            Titulek každého panelu obsahuje
            **celkový objem korun na hektar** pro daný stav.

            ---

            ## Interaktivita

            - Pohybem myši se zobrazí:
            - výška vrstvy
            - objem korun dané skupiny
            - Režim sjednoceného kurzoru umožňuje
            snadné porovnání hodnot mezi křivkami

            ---

            ## Interpretace

            - Poloha maxima křivky ukazuje,
            ve které výšce je soustředěna většina korun
            - Rozdíly mezi panely ukazují,
            jak zásah změnil vertikální strukturu porostu
            - Změna tvaru křivek indikuje selektivitu zásahu
            - Porovnání skupin odhaluje rozdíly
            mezi dřevinami nebo typy managementu

            ---

            ## Poznámka
            Pokud nejsou k dispozici potřebná data o korunách,
            grafy se nezobrazí a aplikace na to upozorní.
            """,

            "en": """
            ### Vertical crown volume profiles

            This page visualizes the **vertical distribution of tree crown volume**
            and allows comparison of crown structure:
            - before intervention
            - after intervention
            - removed trees

            Values are aggregated by height meters
            and normalized **per hectare**.

            ---

            ## Controls

            **Show values by**  
            Defines the grouping used to draw the profiles:
            - species
            - management category

            You can display:
            - a single grouping,
            - or a combination of two groupings as an overlay.

            The first selected grouping forms the primary profiles,
            the second (if selected) is shown as an overlay.

            ---

            ## Chart structure

            Each of the three panels shows the same type of profile:
            - left panel: before intervention
            - middle panel: after intervention
            - right panel: removed trees

            Y-axis:
            - height above ground (meters)

            X-axis:
            - crown volume within the height layer (m³/ha)

            ---

            ## Curves in the charts

            - Colored curves represent individual groups
            (species or management)
            - Line width and style distinguish primary and overlay layers
            - The dotted gray “Sum” curve shows
            total crown volume at each height without grouping

            The title of each panel includes
            the **total crown volume per hectare** for that state.

            ---

            ## Interactivity

            - Hovering shows:
            - height of the layer
            - crown volume of the selected group
            - Unified hover mode allows
            easy comparison across curves

            ---

            ## Interpretation

            - The position of the peak indicates
            where most crown volume is concentrated vertically
            - Differences between panels reveal
            how management altered vertical structure
            - Changes in curve shape indicate selectivity
            - Group comparison highlights differences
            between species or management types

            ---

            ## Note
            If required crown data are missing,
            the charts are hidden and a warning is displayed.
            """
        },

    "space_comp_help": {
        "cs": """
            ### Prostorová konkurence korun

            Tato stránka hodnotí **konkurenci stromů o prostor v korunovém patře**
            na základě **sdílených objemů korun** (překryv korunových voxelů).

            Analýza ukazuje:
            - kolik korunového prostoru je sdíleno mezi stromy,
            - jak se sdílený prostor rozděluje mezi dřeviny a typy zásahu,
            - jak se konkurence mění po zásahu.

            ---

            ## Horní ovládání

            **Stav porostu**  
            Určuje, pro jaký stav se konkurence vyhodnocuje:
            - před zásahem
            - po zásahu  

            Po zásahu se sdílený objem automaticky upravuje
            s ohledem na odstraněné stromy.

            **Zobrazit hodnoty jako**  
            Přepíná způsob vyjádření sdíleného prostoru:
            - procenta z celkového sdíleného objemu
            - absolutní objem (m³)

            **Filtry DBH a výšky**  
            Omezují množinu hodnocených stromů podle velikosti.

            ---

            ## Co vstupuje do výpočtu

            - Vybrané stromy tvoří **fokální množinu**
            - Do výpočtu jsou automaticky zahrnuti i jejich
            bezprostřední sousedé, se kterými sdílejí koruny
            - Filtry druhů a managementu určují,
            které stromy jsou považovány za fokální

            To zajišťuje, že konkurence je vyhodnocena korektně
            i na hranách výběru.

            ---

            ## Struktura grafů

            ### Sdílený vs. celkový objem korun
            Levý diagram porovnává:
            - celkový objem korun fokálních stromů
            - objem korunového prostoru sdíleného s okolními stromy  

            V procentním režimu vyjadřuje,
            jak velká část korun je v konkurenci.

            ---

            ### Sdílený prostor podle dřevin
            Prostřední sloupcový graf ukazuje,
            jaký podíl sdíleného prostoru připadá na jednotlivé dřeviny.

            ---

            ### Sdílený prostor podle zásahu
            Pravý sloupcový graf zobrazuje rozdělení
            sdíleného prostoru podle kategorií managementu.

            ---

            ## Interpretace

            - Vysoký podíl sdíleného prostoru znamená
            silnou konkurenci o prostor v korunách
            - Pokles sdílení po zásahu ukazuje,
            že zásah uvolnil korunový prostor
            - Rozdělení podle dřevin nebo zásahu odhaluje,
            které skupiny se na konkurenci podílejí nejvíce

            ---

            ## Poznámka
            Pokud nejsou k dispozici údaje o sdílení korun
            nebo filtry vyloučí všechny relevantní stromy,
            grafy se nezobrazí a aplikace na to upozorní.
            """,

            "en": """
            ### Crown space competition

            This page evaluates **competition between trees for canopy space**
            based on **shared crown volumes** (overlapping crown voxels).

            The analysis shows:
            - how much crown space is shared between trees,
            - how shared space is distributed across species and management,
            - how competition changes after intervention.

            ---

            ## Top controls

            **Stand state**  
            Selects the stand state for evaluation:
            - before intervention
            - after intervention  

            After intervention, shared volumes are automatically adjusted
            to account for removed trees.

            **Show values as**  
            Switches how shared space is expressed:
            - percentage of total shared volume
            - absolute volume (m³)

            **DBH and height filters**  
            Restrict the set of evaluated trees by size.

            ---

            ## What enters the calculation

            - Selected trees form the **focal set**
            - Immediate neighbors sharing crown space
            are automatically included in the calculation
            - Species and management filters define
            which trees are considered focal

            This ensures that crown competition is evaluated correctly,
            even at the edges of the selection.

            ---

            ## Chart structure

            ### Shared vs total crown volume
            The left diagram compares:
            - total crown volume of focal trees
            - crown volume shared with neighboring trees  

            In percentage mode, it expresses
            the proportion of crown space under competition.

            ---

            ### Shared space by species
            The middle bar chart shows
            how shared crown space is distributed among species.

            ---

            ### Shared space by management
            The right bar chart shows
            how shared crown space is distributed across management categories.

            ---

            ## Interpretation

            - A high share of shared space indicates
            strong competition in the canopy
            - A decrease after intervention indicates
            that management released crown space
            - Group-wise breakdown reveals
            which species or management types dominate competition

            ---

            ## Note
            If crown sharing data are missing
            or filters exclude all relevant trees,
            the charts are hidden and a warning is shown.
            """
        },

    "light_comp_help": {
        "cs": """
            ### Konkurence o světlo

            Tato stránka hodnotí **světelné podmínky stromů**
            a **konkurenci o světlo způsobenou okolními stromy**.

            Analýza pracuje s:
            - dostupným světlem jednotlivých stromů (sky view, %),
            - příspěvky sousedních stromů ke stínění.

            Všechny hodnoty jsou vyjádřeny **v procentech (%)**.

            ---

            ## Horní ovládání

            **Stav porostu**  
            Určuje, pro jaký stav se konkurence vyhodnocuje:
            - před zásahem
            - po zásahu  

            Po zásahu se světelné podmínky automaticky přepočítají
            s ohledem na odstraněné stromy.

            **Režim zobrazení**  
            Přepíná mezi dvěma pohledy:
            - **Kdo konkuruje** – kolik světla jednotlivé skupiny odebírají
            - **Sky view** – rozdělení dostupného světla stromů

            **Filtry DBH a výšky**  
            Omezují množinu hodnocených stromů podle velikosti.

            ---

            ## Struktura grafů

            ### Průměrné dostupné světlo
            Levý kruhový diagram znázorňuje:
            - 100 % potenciálního světla (vnější kruh),
            - průměrné dostupné světlo stromů (vnitřní plocha).

            Čím větší vnitřní plocha, tím lepší světelné podmínky.

            ---

            ## Režim „Kdo konkuruje“

            ### Stínění podle dřevin
            Prostřední sloupcový graf ukazuje,
            jaký podíl stínění způsobují jednotlivé dřeviny.

            ### Stínění podle zásahu
            Pravý sloupcový graf zobrazuje,
            jaký podíl stínění pochází z jednotlivých kategorií managementu.

            Hodnoty představují **součet procent stínění**
            přes všechny hodnocené stromy.

            ---

            ## Režim „Sky view“

            V tomto režimu jsou zobrazeny:
            - houslové grafy dostupného světla stromů
            - rozdělené podle dřevin nebo zásahu

            Grafy ukazují:
            - typické světelné podmínky,
            - variabilitu mezi stromy,
            - extrémně zastíněné i velmi osvětlené jedince.

            ---

            ## Filtry dole

            **Výběr dřevin a zásahů**  
            Určuje, které stromy tvoří hodnocenou množinu.

            Tyto filtry ovlivňují:
            - průměrné dostupné světlo,
            - rozdělení konkurence,
            - sky view statistiky.

            ---

            ## Interpretace

            - Nízké průměrné světlo znamená silnou konkurenci
            - Pokles konkurence po zásahu indikuje uvolnění korun
            - Rozdělení stínění ukazuje,
            které skupiny nejvíce omezují světelné podmínky
            - Sky view režim odhaluje heterogenitu porostu

            ---

            ## Poznámka
            Pokud nejsou k dispozici údaje o světelných poměrech
            nebo filtry vyloučí všechny relevantní stromy,
            grafy se nezobrazí a aplikace na to upozorní.
            """,

            "en": """
            ### Competition for light

            This page evaluates **light conditions of trees**
            and **competition for light caused by neighboring trees**.

            The analysis is based on:
            - available light per tree (sky view, %),
            - shading contributions from surrounding trees.

            All values are expressed **in percent (%)**.

            ---

            ## Top controls

            **Stand state**  
            Selects the stand state for evaluation:
            - before intervention
            - after intervention  

            After intervention, light conditions are automatically adjusted
            to account for removed trees.

            **Display mode**  
            Switches between two perspectives:
            - **Who competes** – which groups cause shading
            - **Sky view** – distribution of available light among trees

            **DBH and height filters**  
            Restrict the set of evaluated trees by size.

            ---

            ## Chart structure

            ### Average available light
            The left circular diagram shows:
            - 100 % potential light (outer circle),
            - average available light of trees (inner area).

            Larger inner area indicates better light conditions.

            ---

            ## “Who competes” mode

            ### Shading by species
            The middle bar chart shows
            how much shading is caused by individual species.

            ### Shading by management
            The right bar chart shows
            how much shading originates from management categories.

            Values represent the **sum of shading percentages**
            across all evaluated trees.

            ---

            ## “Sky view” mode

            This mode displays:
            - violin plots of available light
            - grouped by species or management

            The plots show:
            - typical light conditions,
            - variability between trees,
            - strongly shaded and well-lit individuals.

            ---

            ## Bottom filters

            **Species and management selection**  
            Define which trees form the evaluated set.

            These filters affect:
            - average available light,
            - competition breakdown,
            - sky view statistics.

            ---

            ## Interpretation

            - Low average light indicates strong competition
            - Reduced competition after intervention indicates canopy release
            - Shading breakdown reveals dominant competitors
            - Sky view mode highlights stand heterogeneity

            ---

            ## Note
            If light data are missing
            or filters exclude all relevant trees,
            the charts are hidden and a warning is shown.
            """
        },

    "add_att_help": {
  "cs": """
### Import a správa uživatelských atributů stromů

Tato stránka slouží k **importu uživatelských atributů stromů**
ze souboru CSV a k jejich **uložení do projektu**.

Atributy jsou ukládány **odděleně od základních dat stromů**
do interní databáze projektu a nemění původní projektová data.

---

## Co tato stránka umožňuje

- importovat nové uživatelské atributy stromů z CSV,
- **aktualizovat (přepsat)** již existující uživatelské atributy,
- **mazat vybrané uživatelské atributy** z projektu.

---

## Postup importu

1. Nahrajte CSV soubor pomocí pole pro nahrání souboru  
2. Aplikace provede kontrolu dat a zobrazí případná upozornění  
3. Pokud je vše v pořádku, klikněte na **Import attributes**

Import se provede **až po potvrzení tlačítkem**.

---

## Požadavky na CSV soubor

CSV soubor musí splňovat **všechny následující podmínky**:

- musí obsahovat sloupec **`id`** (identifikátor stromu),
- hodnoty `id` musí být **číselné** a bez chybějících hodnot,
- `id` v CSV musí **přesně odpovídat** ID stromů v projektu  
  (žádné ID navíc ani žádné chybějící),
- všechny ostatní sloupce jsou považovány za atributy,
- hodnoty atributů musí být **číselné (float)**,
- v datech nesmí být žádné prázdné nebo nečíselné hodnoty.

Pokud některá z podmínek není splněna, **import není povolen**.

---

## Kontrola dat před importem

Po nahrání CSV aplikace automaticky:

- ověří shodu ID stromů s projektem,
- zkontroluje datové typy hodnot,
- zjistí, zda některé atributy již v projektu existují.

### Přepis existujících atributů
Pokud CSV obsahuje atributy, které již v projektu existují,
zobrazí se **upozornění** a import je možný
**až po explicitním potvrzení přepisu uživatelem**.

---

## Uložení atributů

Po potvrzení importu:

- atributy jsou uloženy do databáze projektu,
- existující uživatelské atributy mohou být přepsány,
- základní data stromů (**projektová data**) zůstávají beze změny.

Importované atributy jsou dostupné v aplikaci
jako **uživatelská data projektu**.

---

## Mazání uživatelských atributů

V dolní části stránky lze:

- vybrat jeden nebo více uživatelských atributů,
- kliknutím na **Remove selected attributes** je trvale odstranit.

Mazání:
- se projeví okamžitě,
- neovlivňuje základní data stromů,
- nelze jej vrátit zpět.

---

## Doporučení

- Před importem vždy zkontrolujte shodu ID stromů mezi CSV a projektem
- Při přepisu existujících atributů pracujte s kopií projektu
- Atributy ukládejte s jednoznačnými a srozumitelnými názvy
- Datový typ a význam hodnot by měl být konzistentní v celém projektu

---

## Poznámka

Tato stránka **nemění původní projektová data stromů**.
Uživatelské atributy jsou spravovány samostatně
a mohou být kdykoliv z projektu odstraněny.
""",

  "en": """
### Import and management of user-defined tree attributes

This page is used to **import user-defined tree attributes**
from a CSV file and store them in the project.

User attributes are stored **separately from the core tree data**
in the project database and do not modify the original project data.

---

## What this page allows

- importing new user-defined tree attributes from CSV,
- **updating (overwriting)** existing user-defined attributes,
- **removing selected user-defined attributes** from the project.

---

## Import workflow

1. Upload a CSV file using the file uploader  
2. The application validates the data and shows any warnings  
3. If everything is valid, click **Import attributes**

The import is executed **only after confirmation**.

---

## CSV file requirements

The CSV file must meet **all of the following conditions**:

- it must contain a column named **`id`** (tree identifier),
- `id` values must be **numeric** and without missing values,
- CSV `id` values must **exactly match** tree IDs in the project  
  (no extra or missing IDs),
- all other columns are treated as attributes,
- attribute values must be **numeric (float)**,
- no missing or non-numeric values are allowed.

If any condition is not met, **the import is disabled**.

---

## Data validation before import

After uploading the CSV, the application automatically:

- verifies ID consistency with the project,
- checks value data types,
- detects attributes that already exist in the project.

### Overwriting existing attributes
If the CSV contains attributes that already exist,
a **warning is shown** and the import is allowed
**only after explicit user confirmation**.

---

## Attribute storage

After confirming the import:

- attributes are stored in the project database,
- existing user-defined attributes may be overwritten,
- core project tree data remain unchanged.

Imported attributes are available in the application
as **user-defined project data**.

---

## Removing user-defined attributes

In the lower section of the page, you can:

- select one or more user-defined attributes,
- permanently remove them using **Remove selected attributes**.

Removal:
- takes effect immediately,
- does not affect core tree data,
- cannot be undone.

---

## Recommendations

- Always verify tree ID consistency between the CSV and the project
- Use a project copy when overwriting existing attributes
- Use clear and consistent attribute names
- Ensure consistent meaning and units of values across the project

---

## Note

This page **does not modify the original project tree data**.
User-defined attributes are managed separately
and can be removed from the project at any time.
"""
},

    "simulation_help": {
    "cs": """
        ### Simulace růstu lesního porostu (iLand)

        #### 1️⃣ Ovládání simulace
        Tato stránka umožňuje spustit **simulaci vývoje lesního porostu** pomocí modelu *iLand*.

        Postup práce:
        - Pomocí tlačítka **Znovu spustit simulaci** se spustí sada opakovaných běhů modelu (*replikací*).
        - Posuvníkem **Simulované období** zvolíte délku simulace v letech.
        - Přepínače **Mortalita** a **Obnova** zapínají či vypínají náhodné (stochastické) procesy v modelu.

        Každá replikace:
        - začíná ze stejného počátečního stavu,
        - ale může se vyvíjet odlišně v důsledku náhodných procesů (pokud jsou zapnuté).

        ---

        #### 2️⃣ Interpretace výsledných grafů
        Výsledky jsou zobrazeny pomocí tzv. **fan chartů (vějířových grafů)**.

        Co grafy zobrazují:
        - **Středová čára** představuje medián (typický průběh).
        - **Barevné pásy** kolem čáry vyjadřují nejistotu výsledků:
        - užší pásy = 25–75 % kvantil
        - širší pásy = 5–95 % kvantil
        - Čím širší vějíř, tím větší rozdíly mezi replikacemi.

        Zobrazené grafy:
        - **Objem podle dřevin** – vývoj objemu jednotlivých druhů a jejich součet.
        - **Objem podle managementu** – cílové stromy vs. ostatní stromy + celkový objem.

        💡 Pokud grafy vypadají jako jediná čára bez vějíře, znamená to, že všechny replikace jsou shodné.

        ---

        #### 3️⃣ Stochastičnost a význam přepínačů
        - **Mortalita**: zapíná náhodný úhyn stromů v čase.
        - **Obnova**: zapíná vznik nových stromů (zmlazení).

        Pokud jsou **oba přepínače vypnuté**:
        - simulace je **deterministická**,
        - všechny replikace mají identický průběh,
        - fan chart se zhroutí na jednu čáru.

        Pro smysluplnou analýzu nejistoty je doporučeno zapnout alespoň **mortalitu**.

        ---

        #### 4️⃣ Model iLand – stručný popis
        *iLand* (*individual-based forest landscape model*) je procesně orientovaný model, který simuluje:

        - růst jednotlivých stromů,
        - konkurenci o světlo, vodu a živiny,
        - mortalitu a obnovu,
        - reakci porostu na klima a stanoviště.

        Model pracuje na úrovni **jednotlivých stromů**, ale výsledky jsou agregovány
        na úroveň porostu nebo plochy.

        Simulace zde slouží k:
        - odhadu budoucího vývoje porostu,
        - porovnání scénářů,
        - pochopení nejistoty predikcí.

        📘 Více informací o modelu iLand:
        https://iland-model.org
        """,

            "en": """
        ### Forest Growth Simulation (iLand)

        #### 1️⃣ Simulation controls
        This page allows you to run **forest stand development simulations** using the *iLand* model.

        Workflow:
        - Click **Restart simulation** to start a set of repeated model runs (*replications*).
        - Use the **Simulation period** slider to select the number of simulated years.
        - The **Mortality** and **Regeneration** toggles enable or disable stochastic processes.

        Each replication:
        - starts from the same initial conditions,
        - but may evolve differently due to random processes (if enabled).

        ---

        #### 2️⃣ Interpreting the results
        Results are visualized using **fan charts**.

        What the charts show:
        - The **central line** represents the median (typical trajectory).
        - **Colored bands** indicate uncertainty across replications:
        - inner band: 25–75 % quantile
        - outer band: 5–95 % quantile
        - Wider fans indicate greater variability between model runs.

        Displayed charts:
        - **Volume by species** – development of individual species and total volume.
        - **Volume by management** – target trees vs. other trees plus total volume.

        💡 If the chart collapses to a single line, all replications are identical.

        ---

        #### 3️⃣ Stochasticity and toggles
        - **Mortality** enables random tree death over time.
        - **Regeneration** enables the establishment of new trees.

        If **both toggles are off**:
        - the simulation is **deterministic**,
        - all replications produce identical results,
        - fan charts show no spread.

        For meaningful uncertainty analysis, enabling at least **mortality** is recommended.

        ---

        #### 4️⃣ About the iLand model
        *iLand* (*individual-based forest landscape model*) is a process-based model simulating:

        - individual tree growth,
        - competition for light, water, and nutrients,
        - mortality and regeneration,
        - climate and site effects.

        The model operates at the **individual tree level**, while outputs are aggregated
        to stand or plot level for analysis.

        This simulation supports:
        - exploration of future stand development,
        - scenario comparison,
        - understanding prediction uncertainty.

        📘 More information about iLand:
        https://iland-model.org
        """,
    },


}
