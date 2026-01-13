# sun_access_cone_raytracing_multi_las.py
# -*- coding: utf-8 -*-

import json
import math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ------------------------
# Parametry
# ------------------------
APEX_ANGLE_DEG = 60.0        # vrcholový úhel kužele (celkový), poloviční úhel = 30°
VOXEL_SIZE = 0.25            # [m] velikost voxelu
CONE_HEIGHT = 60.0           # [m] max délka paprsku / výška kužele
NEIGHBOR_RADIUS = 40.0       # [m] horizontální okno (vybere okolní stromy podle feather)
Z_MARGIN_UP = 10.0           # [m] rezerva nad Z cílového bodu
Z_MIN = 0.0                  # [m] spodní mez
N_RAY_SAMPLES = 5000         # počet paprsků v kuželu (vyšší= přesnější, pomalejší)

# start paprsku: malé odsazení od apexu, aby paprsek hned nenarazil do "stejného voxelu"
# (užitečné, když se v datech vyskytují body extrémně blízko focal pointu)
RAY_T0 = 2.0 * VOXEL_SIZE

INPUT_FEATHER = r"C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/out_poses.feather"
OUTPUT_CSV = r"C:/Users/krucek/OneDrive - vukoz.cz/DATA/_GS-LCR/SLP_Pokojna/crown_poses_light.csv"

# ------------------------
# Pomocné funkce
# ------------------------
def unit_vector(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def sample_directions_in_cone(axis, half_angle_rad, n_samples):
    """
    Náhodné přibližně rovnoměrné směry uvnitř kužele o polovičním úhlu half_angle_rad
    kolem osy 'axis' (normovaný vektor).
    """
    axis = unit_vector(axis)

    # báze: axis => z, zvolíme libovolnou ortonormální soustavu
    if abs(axis[2]) < 0.999:
        x_axis = unit_vector(np.cross(axis, np.array([0, 0, 1.0])))
    else:
        x_axis = unit_vector(np.cross(axis, np.array([0, 1.0, 0])))
    y_axis = np.cross(axis, x_axis)

    cos_min = math.cos(half_angle_rad)
    u = np.random.rand(n_samples)
    cos_theta = 1 - u * (1 - cos_min)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.rand(n_samples)

    dirs_local = (
        (sin_theta * np.cos(phi))[:, None] * x_axis[None, :] +
        (sin_theta * np.sin(phi))[:, None] * y_axis[None, :] +
        (cos_theta)[:, None] * axis[None, :]
    )
    return dirs_local

def points_in_inverted_cone(apex, axis, half_angle_rad, height, pts):
    """
    Vrátí masku bodů v "obráceném kuželu" s vrcholem v apex a osou 'axis' (normováno),
    poloviční úhel = half_angle_rad, délka = height.
    """
    v = pts - apex
    proj_len = np.dot(v, axis)
    inside_height = (proj_len >= 0) & (proj_len <= height)

    v_norm = np.linalg.norm(v, axis=1)
    safe = v_norm > 0
    cosang = np.zeros_like(v_norm)
    cosang[safe] = (np.dot(v[safe], axis)) / v_norm[safe]

    inside_angle = np.zeros_like(inside_height, dtype=bool)
    inside_angle[safe] = cosang[safe] >= math.cos(half_angle_rad)

    return inside_height & inside_angle

def voxel_hash(points, voxel_size, extra_payload=None, reducer="first"):
    """
    Založí obsazenost voxelů. Vrací:
      - set klíčů voxelů (i,j,k)
      - map voxel -> payload (např. id stromu)
    """
    ijk = np.floor(points / voxel_size).astype(np.int64)
    keys = [tuple(x) for x in ijk]
    occ = {}

    if extra_payload is None:
        for k in keys:
            if k not in occ:
                occ[k] = None
        return set(occ.keys()), occ

    if reducer == "first":
        for k, p in zip(keys, extra_payload):
            if k not in occ:
                occ[k] = p
    elif reducer == "mode":
        bags = defaultdict(list)
        for k, p in zip(keys, extra_payload):
            bags[k].append(p)
        for k, bag in bags.items():
            occ[k] = Counter(bag).most_common(1)[0][0] if bag else None
    else:
        for k in keys:
            if k not in occ:
                occ[k] = None

    return set(occ.keys()), occ

def raytrace_voxels(apex, dirs, voxel_size, height, occ_set, occ_payload, t0=0.0):
    """
    Jednoduchý voxel stepping (krok ~ voxel_size).
    Pro každý směr vrací tuple (hit_bool, hit_payload_or_None).
    """
    results = []
    for d in dirs:
        d = unit_vector(d)
        t = float(t0)
        hit_payload = None
        hit = False
        while t <= height:
            p = apex + d * t
            k = tuple(np.floor(p / voxel_size).astype(np.int64))
            if k in occ_set:
                hit = True
                hit_payload = occ_payload.get(k, None)
                break
            t += voxel_size
        results.append((hit, hit_payload))
    return results

def laspy_read_points_window(las_path, x, y, z_min, z_max, radius):
    """
    Načte LAS a vybere body v okně kolem (x,y) s daným radius a ve výškovém intervalu.
    (Fallback v RAM – bez PDAL indexu.)
    """
    import laspy

    with laspy.open(las_path) as f:
        las = f.read()

    X = las.X * las.header.scale[0] + las.header.offsets[0]
    Y = las.Y * las.header.scale[1] + las.header.offsets[1]
    Z = las.Z * las.header.scale[2] + las.header.offsets[2]

    dx = X - x
    dy = Y - y
    r2 = dx * dx + dy * dy

    mask = (r2 <= radius * radius) & (Z >= z_min) & (Z <= z_max)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    pts = np.vstack([X[mask], Y[mask], Z[mask]]).T.astype(np.float32)
    return pts

# ------------------------
# Hlavní běh
# ------------------------
def main():
    df = pd.read_feather(INPUT_FEATHER)

    # Očekáváme sloupce: x, y, z, file, id
    required = {"x", "y", "z", "file", "id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Chybí sloupce ve featheru: {missing}. Nalezeno: {list(df.columns)}")

    # Pro rychlé vyhledání sousedů si vytáhneme pole
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)
    zs = df["z"].to_numpy(dtype=float)
    ids = df["id"].to_numpy()
    files = df["file"].astype(str).to_numpy()

    half_angle = math.radians(APEX_ANGLE_DEG / 2.0)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)  # kužel “nahoru”

    print("[INFO] Běží fallback čtení LAS přes laspy (bez PDAL).")
    print(f"[INFO] Stromů ve featheru: {len(df)}")

    results = []
    for i in range(len(df)):
        tree_id = int(ids[i])
        las_path_target = files[i]
        x = float(xs[i])
        y = float(ys[i])
        z = float(zs[i])

        # vyber sousední stromy podle feather souřadnic (kromě cílového stromu)
        dx = xs - x
        dy = ys - y
        r2 = dx * dx + dy * dy
        neighbor_mask = (r2 <= NEIGHBOR_RADIUS * NEIGHBOR_RADIUS)

        # vynech cílový strom (podle indexu / file)
        neighbor_mask[i] = False

        neighbor_idx = np.where(neighbor_mask)[0]
        if neighbor_idx.size == 0:
            results.append({
                "ID": tree_id,
                "free_space_pct": 100.0,
                "shade_breakdown": "{}"
            })
            continue

        # načti body ze všech sousedních stromů (jejich LASy)
        z_min = Z_MIN
        z_max = z + CONE_HEIGHT + Z_MARGIN_UP

        pts_list = []
        payload_list = []

        for j in neighbor_idx:
            las_path = files[j]
            neighbor_id = int(ids[j])

            # načti jen okno kolem cíle (radius + výškové okno)
            pts_j = laspy_read_points_window(
                las_path=las_path,
                x=x, y=y,
                z_min=z_min, z_max=z_max,
                radius=NEIGHBOR_RADIUS
            )
            if pts_j.shape[0] == 0:
                continue

            pts_list.append(pts_j)
            payload_list.append(np.full(pts_j.shape[0], neighbor_id, dtype=np.int64))

        if not pts_list:
            results.append({
                "ID": tree_id,
                "free_space_pct": 100.0,
                "shade_breakdown": "{}"
            })
            continue

        pts = np.vstack(pts_list).astype(np.float32)
        payload_all = np.concatenate(payload_list).astype(np.int64)

        # omez na kužel
        apex = np.array([x, y, z], dtype=np.float64)
        mask_cone = points_in_inverted_cone(apex, axis, half_angle, CONE_HEIGHT, pts)
        pts_cone = pts[mask_cone]
        payload_cone = payload_all[mask_cone]

        if pts_cone.shape[0] == 0:
            results.append({
                "ID": tree_id,
                "free_space_pct": 100.0,
                "shade_breakdown": "{}"
            })
            continue

        # voxelizace
        occ_set, occ_payload = voxel_hash(pts_cone, VOXEL_SIZE, extra_payload=payload_cone, reducer="mode")

        # ray tracing
        dirs = sample_directions_in_cone(axis, half_angle, N_RAY_SAMPLES)
        hits = raytrace_voxels(apex, dirs, VOXEL_SIZE, CONE_HEIGHT, occ_set, occ_payload, t0=RAY_T0)

        total = len(hits)
        blocked = sum(1 for h, _ in hits if h)
        free = total - blocked
        free_pct = 100.0 * free / total if total > 0 else 100.0

        by_id = Counter()
        for h, pid in hits:
            if not h:
                continue
            # pid by tu měl být vždy neighbor_id (protože payload máme)
            by_id[int(pid) if pid is not None else None] += 1

        breakdown_pct = {}
        for k, v in by_id.items():
            label = "unknown" if (k is None) else str(k)
            breakdown_pct[label] = round(100.0 * v / total, 3)

        results.append({
            "ID": tree_id,
            "free_space_pct": round(free_pct, 3),
            "shade_breakdown": json.dumps(breakdown_pct, ensure_ascii=False)
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Uloženo: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
