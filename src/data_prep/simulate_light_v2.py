# -*- coding: utf-8 -*-
# ==============================================================
# FAST SUN ACCESS CONE RAYTRACING
# zachování funkčnosti, adaptivní sampling, numba
# ==============================================================

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from numba import njit
import laspy

# ==============================================================
# PARAMETRY
# ==============================================================

APEX_ANGLE_DEG = 60.0
VOXEL_SIZE = 0.25
CONE_HEIGHT = 60.0
NEIGHBOR_RADIUS = 40.0
Z_MARGIN_UP = 10.0
Z_MIN = 0.0

N_RAY_MAX = 5000
N_RAY_MIN = 500
RAY_BATCH = 250
MAX_ERR_PCT = 0.5

RAY_T0 = 2.0 * VOXEL_SIZE

INPUT_PROJECT_JSON = r"D:/GS_LCR_DELIVERABLE/Krivoklat/Krivoklat.json"
OUTPUT_PROJECT_JSON = r"D:/GS_LCR_DELIVERABLE/Krivoklat/Krivoklat_light.json"
LAS_DIR_OVERRIDE = r""
LAS_EXT = ".las"

APEX_CROWN_HEIGHT_SHIFT_FACTOR = 0.0
EXCLUDE_LABEL_SUBSTRINGS = ("ground", "unsegmented")

# ==============================================================
# POMOCNÉ
# ==============================================================

def unit(v):
    return v / np.linalg.norm(v)

def is_zero_center(c):
    try:
        return abs(c[0]) < 1e-6 and abs(c[1]) < 1e-6 and abs(c[2]) < 1e-6
    except:
        return False

# ==============================================================
# RYCHLÝ KUŽELOVÝ TEST
# ==============================================================

def points_in_cone_fast(apex, cos_half_angle, height, pts):
    v = pts - apex
    vz = v[:, 2]
    mask_h = (vz >= 0.0) & (vz <= height)
    v2 = (v * v).sum(axis=1)
    return mask_h & ((vz * vz) >= v2 * (cos_half_angle ** 2))

# ==============================================================
# VOXEL HASH
# ==============================================================

def voxel_keys(points, voxel):
    ijk = np.floor(points / voxel).astype(np.int32)
    return ijk[:, 0].astype(np.int64) \
         + ijk[:, 1].astype(np.int64) * 1000000 \
         + ijk[:, 2].astype(np.int64) * 1000000000000

# ==============================================================
# NUMBA DDA RAYTRACING
# ==============================================================

@njit
def raytrace_batch(apex, dirs, max_dist, voxel, occ_keys, occ_vals):
    hits = np.full(len(dirs), -1, dtype=np.int32)

    for i in range(len(dirs)):
        d = dirs[i]
        t = 0.0
        x, y, z = apex

        ix = int(math.floor(x / voxel))
        iy = int(math.floor(y / voxel))
        iz = int(math.floor(z / voxel))

        step_x = 1 if d[0] > 0 else -1
        step_y = 1 if d[1] > 0 else -1
        step_z = 1 if d[2] > 0 else -1

        tx = ((ix + (step_x > 0)) * voxel - x) / d[0] if d[0] != 0 else 1e9
        ty = ((iy + (step_y > 0)) * voxel - y) / d[1] if d[1] != 0 else 1e9
        tz = ((iz + (step_z > 0)) * voxel - z) / d[2] if d[2] != 0 else 1e9

        dtx = abs(voxel / d[0]) if d[0] != 0 else 1e9
        dty = abs(voxel / d[1]) if d[1] != 0 else 1e9
        dtz = abs(voxel / d[2]) if d[2] != 0 else 1e9

        while t <= max_dist:
            key = ix + iy * 1000000 + iz * 1000000000000
            for j in range(len(occ_keys)):
                if occ_keys[j] == key:
                    hits[i] = occ_vals[j]
                    t = max_dist + 1
                    break

            if tx < ty and tx < tz:
                ix += step_x
                t = tx
                tx += dtx
            elif ty < tz:
                iy += step_y
                t = ty
                ty += dty
            else:
                iz += step_z
                t = tz
                tz += dtz

    return hits

# ==============================================================
# DIRECTIONS CACHE
# ==============================================================

def sample_dirs(n, half_angle):
    u = np.random.rand(n)
    cos_t = 1 - u * (1 - math.cos(half_angle))
    sin_t = np.sqrt(1 - cos_t * cos_t)
    phi = np.random.rand(n) * 2 * math.pi
    return np.column_stack([
        sin_t * np.cos(phi),
        sin_t * np.sin(phi),
        cos_t
    ]).astype(np.float32)

# ==============================================================
# WRITE JSON (BEZE ZMĚN)
# ==============================================================

def write_json(original_path, df, output_path=None):
    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_map = df.set_index("id").to_dict("index")

    for seg in data.get("segments", []):
        sid = seg.get("id")
        if sid not in id_map:
            continue
        attr = seg.get("treeAttributes", {})
        row = id_map[sid]

        attr["light_avail"] = row["light_avail"]
        attr["light_comp"] = row["light_comp"]

        if row.get("crownCenter") is not None:
            if "crownCenter" not in attr or is_zero_center(attr["crownCenter"]):
                attr["crownCenter"] = row["crownCenter"]

    with open(output_path or original_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ==============================================================
# MAIN
# ==============================================================

def main():
    with open(INPUT_PROJECT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    project_dir = os.path.dirname(INPUT_PROJECT_JSON)
    las_dir = LAS_DIR_OVERRIDE or project_dir

    for seg in data.get("segments", []):
        label = seg.get("label", "")
        if any(x in label.lower() for x in EXCLUDE_LABEL_SUBSTRINGS):
            continue

        tid = seg.get("id")
        attrs = seg.get("treeAttributes", {})
        pos = attrs.get("position")
        cc = attrs.get("crownCenter")

        apex = None
        if cc and not is_zero_center(cc):
            apex = cc
        elif pos:
            apex = pos
        else:
            continue

        las_path = os.path.join(las_dir, label + LAS_EXT)

        rows.append({
            "id": tid,
            "x": apex[0],
            "y": apex[1],
            "z": apex[2],
            "las": las_path
        })

    df = pd.DataFrame(rows)
    xs, ys = df["x"].values, df["y"].values
    tree_xy = cKDTree(np.c_[xs, ys])

    half_angle = math.radians(APEX_ANGLE_DEG / 2)
    cos_half = math.cos(half_angle)
    DIRS = sample_dirs(N_RAY_MAX, half_angle)

    results = []

    for i, r in df.iterrows():
        neighbors = tree_xy.query_ball_point([r.x, r.y], NEIGHBOR_RADIUS)
        neighbors = [j for j in neighbors if j != i]

        if not neighbors:
            results.append({"id": r.id, "light_avail": 100.0, "light_comp": {}})
            continue

        pts_all = []
        payload = []

        for j in neighbors:
            las = laspy.read(df.iloc[j].las)
            X = las.x
            Y = las.y
            Z = las.z

            dx = X - r.x
            dy = Y - r.y
            mask = (dx*dx + dy*dy <= NEIGHBOR_RADIUS**2) & (Z >= Z_MIN) & (Z <= r.z + CONE_HEIGHT + Z_MARGIN_UP)

            if np.any(mask):
                pts_all.append(np.column_stack([X[mask], Y[mask], Z[mask]]))
                payload.append(np.full(np.sum(mask), df.iloc[j].id, np.int32))

        if not pts_all:
            results.append({"id": r.id, "light_avail": 100.0, "light_comp": {}})
            continue

        pts = np.vstack(pts_all).astype(np.float32)
        payload = np.concatenate(payload)

        mask_cone = points_in_cone_fast(
            np.array([r.x, r.y, r.z], np.float32),
            cos_half,
            CONE_HEIGHT,
            pts
        )

        pts = pts[mask_cone]
        payload = payload[mask_cone]

        if len(pts) == 0:
            results.append({"id": r.id, "light_avail": 100.0, "light_comp": {}})
            continue

        keys = voxel_keys(pts, VOXEL_SIZE)
        uniq, inv = np.unique(keys, return_inverse=True)

        occ_payload = np.zeros(len(uniq), np.int32)
        for k in range(len(uniq)):
            occ_payload[k] = Counter(payload[inv == k]).most_common(1)[0][0]

        blocked = 0
        sampled = 0
        by_id = Counter()

        while sampled < N_RAY_MAX:
            batch = DIRS[sampled:sampled+RAY_BATCH]
            hits = raytrace_batch(
                np.array([r.x, r.y, r.z], np.float32),
                batch,
                CONE_HEIGHT,
                VOXEL_SIZE,
                uniq,
                occ_payload
            )
            for h in hits:
                if h != -1:
                    blocked += 1
                    by_id[int(h)] += 1
            sampled += len(batch)

            if sampled >= N_RAY_MIN:
                p = blocked / sampled
                err = 1.96 * math.sqrt(p * (1 - p) / sampled) * 100
                if err < MAX_ERR_PCT:
                    break

        free_pct = 100.0 * (1 - blocked / sampled)

        results.append({
            "id": r.id,
            "light_avail": round(free_pct, 3),
            "light_comp": {str(k): round(v / sampled * 100, 3) for k, v in by_id.items()}
        })

    df_res = pd.DataFrame(results)
    df_out = df.merge(df_res, on="id", how="left")
    write_json(INPUT_PROJECT_JSON, df_out, OUTPUT_PROJECT_JSON)

    print("[OK] Hotovo:", OUTPUT_PROJECT_JSON)

if __name__ == "__main__":
    main()
