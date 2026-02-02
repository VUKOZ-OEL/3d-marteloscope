# help_dict.py
from typing import Literal

Lang = Literal["cs", "en"]

HELP_I18N: dict[str, dict[Lang, str]] = {

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
}


}
