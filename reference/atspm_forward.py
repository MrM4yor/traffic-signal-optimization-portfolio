"""Build and evaluate a causal SUMO forward surrogate from preserved replays.

Inputs are route/driver/departure parameters plus exogenous signal and surrounding
traffic histories. Outputs are LLC sequence/times and compact local trajectories.
Vehicle IDs are retained only for leakage-safe grouping and never enter X.
"""

import argparse, bisect, csv, gzip, hashlib, json, time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import joblib, numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from atspm_rl_core import ROOT, NET, MPH, save, read_observations
from atspm_access import route_edges

PARAMS = [
    "speed_mph",
    "accel",
    "decel",
    "emergencyDecel",
    "apparentDecel",
    "tau",
    "sigma",
    "minGap",
    "startupDelay",
    "actionStepLength",
    "length",
    "width",
    "maxSpeed",
    "lcStrategic",
    "lcCooperative",
    "lcSpeedGain",
    "lcKeepRight",
    "lcAssertive",
    "lcImpatience",
    "lcTimeToImpatience",
]
TVARS = [
    "relative_x_m",
    "relative_y_m",
    "speed_mps",
    "accel_mps2",
    "stopped",
    "queue_rank",
    "leader_gap_m",
    "leader_relative_speed_mps",
    "green",
    "local_count",
    "available",
]
DEFAULT = {
    "accel": 2.6,
    "decel": 4.5,
    "emergencyDecel": 9,
    "apparentDecel": 4.5,
    "tau": 1,
    "sigma": 0.5,
    "minGap": 2.5,
    "startupDelay": 0,
    "actionStepLength": 0.1,
    "length": 5,
    "width": 1.8,
    "maxSpeed": 17.8816,
    "lcStrategic": 1,
    "lcCooperative": 1,
    "lcSpeedGain": 1,
    "lcKeepRight": 1,
    "lcAssertive": 0,
    "lcImpatience": 0,
    "lcTimeToImpatience": 1e6,
}


def geometry():
    """Load lane polyline coordinates and cumulative segment distances for projection of recorded positions."""
    root = ET.parse(NET).getroot()
    result = {}
    for lane in root.findall(".//lane"):
        shape = []
        for item in lane.get("shape", "").split():
            try:
                shape.append(tuple(map(float, item.split(","))))
            except ValueError:
                pass
        if len(shape) < 2:
            continue
        seg = [0.0]
        for a, b in zip(shape, shape[1:]):
            seg.append(seg[-1] + ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
        result[lane.get("id")] = (
            shape,
            np.asarray(seg),
            float(lane.get("length", seg[-1])),
        )
    return result


def xy(geo, lane, pos):
    """Project a lane position in metres onto its SUMO polyline; unknown lanes yield nonfinite coordinates."""
    if lane not in geo:
        return (np.nan, np.nan)
    shape, distance, length = geo[lane]
    d = np.clip(pos / max(length, 1e-6) * distance[-1], 0, distance[-1])
    i = min(len(shape) - 2, bisect.bisect_right(distance, d) - 1)
    f = (d - distance[i]) / max(1e-9, distance[i + 1] - distance[i])
    return tuple(shape[i][k] + f * (shape[i + 1][k] - shape[i][k]) for k in (0, 1))


def route_and_types(path):
    """Resolve replay vehicle/type controls and route intentions; read trusted local XML."""
    root = ET.parse(path).getroot()
    types = {t.get("id"): t for t in root.findall("vType")}
    vehicles = {}
    for v in root.findall("vehicle"):
        edges = route_edges(root, v)
        vehicles[v.get("id")] = (
            v,
            types[v.get("type")],
            edges,
            v.get("route") or " ".join(edges),
        )
    return vehicles


def snapshots(path, groups, signal_lane):
    """Read preserved traffic/signal histories and derive acceleration only where older snapshots omitted it."""
    times = []
    frames = []
    context = []
    previous = {}
    previous_time = None
    with gzip.open(path, "rt") as f:
        for line in f:
            z = json.loads(line)
            now = float(z["time"])
            vehicles = []
            dt = now - previous_time if previous_time is not None else 0
            for r in z["vehicles"]:
                speed = float(r[3])
                acc = (
                    float(r[4])
                    if len(r) > 4
                    else ((speed - previous.get(r[0], speed)) / dt if dt > 0 else 0.0)
                )
                vehicles.append((r[0], r[1], float(r[2]), speed, acc))
            aggregate = np.zeros((len(groups), 3), dtype=np.float32)
            lane_aggregate = defaultdict(lambda: np.zeros(5, dtype=np.float32))
            for _, lane, _, speed, _ in vehicles:
                if lane in signal_lane:
                    g = signal_lane[lane][0]
                    aggregate[g] += [1, float(speed < 0.1), speed]
                    lane_aggregate[lane][:3] += [1, float(speed < 0.1), speed]
            for _, lane, pos, _, _ in vehicles:
                if lane in lane_aggregate:
                    value = lane_aggregate[lane]
                    if pos > value[3]:
                        value[4] = value[3]
                        value[3] = pos
                    elif pos > value[4]:
                        value[4] = pos
            aggregate[:, 2] /= np.maximum(1, aggregate[:, 0])
            for value in lane_aggregate.values():
                value[2] /= max(1, value[0])
            phases = [
                {"red": 0, "yellow": 0.5, "green": 1}.get(
                    z["phases"].get(s, {}).get(str(p), "red"), 0
                )
                for s in ("7241", "7242")
                for p in range(1, 9)
            ]
            times.append(now)
            frames.append({r[0]: r for r in vehicles})
            context.append(
                (
                    aggregate,
                    np.asarray(phases, dtype=np.float32),
                    z["phases"],
                    dict(lane_aggregate),
                )
            )
            previous = {r[0]: r[3] for r in vehicles}
            previous_time = now
    return np.asarray(times), frames, context


def driver_features(
    v, vt, edges, route_id, guidance, edge_index, class_index, route_index, lane_index
):
    desired = (
        min(
            float(v.get("speedFactor", "1")) * 13.89,
            float(vt.get("maxSpeed", DEFAULT["maxSpeed"])),
        )
        / MPH
    )
    values = []
    for p in PARAMS:
        values.append(desired if p == "speed_mph" else float(vt.get(p, DEFAULT[p])))
    classes = [float(vt.get("vClass", "passenger") == c) for c in class_index]
    route = [0.0] * len(edge_index)
    for e in edges:
        if e in edge_index:
            route[edge_index[e]] = 1
    routes = [float(route_id == r) for r in route_index]
    lanes = [0.0] * (2 * len(lane_index))
    for role, offset in (("source", 0), ("destination", len(lane_index))):
        lane = guidance.get(role)
        if lane in lane_index:
            lanes[offset + lane_index[lane]] = 1
    depart = float(v.get("depart"))
    clock = 2 * np.pi * (depart % 86400) / 86400
    numeric = [
        (depart - 79440) / 4200,
        np.sin(clock),
        np.cos(clock),
        float(
            v.get("departPos", "0")
            if v.get("departPos", "0").replace(".", "", 1).isdigit()
            else 0
        ),
        float(
            v.get("departSpeed", "0")
            if v.get("departSpeed", "0").replace(".", "", 1).isdigit()
            else 0
        ),
    ]
    return np.asarray(
        values + classes + route + routes + lanes + numeric, dtype=np.float32
    )


def build_one(
    folder,
    env,
    window,
    context_horizon,
    context_step,
    geo,
    edges,
    classes,
    detectors,
    route_ids,
    guidance_lanes,
):
    route = folder / "candidate.rou.xml"
    vehicles = route_and_types(route)
    observed = defaultdict(list)
    guidance = defaultdict(dict)
    guide = folder.parent / (
        folder.name + "_guidance.csv"
        if folder.name.startswith("gui_")
        else "guidance_" + folder.name.split("_")[-1] + ".csv"
    )
    if guide.exists():
        with guide.open() as f:
            for row in csv.DictReader(f):
                d = env.nodes.get(row["detector_id"])
                guidance[row["vehicle_id"]][row["role"]] = d["sumo_lane"] if d else None
    for r in read_observations(folder):
        if r["detector_id"] in env.nodes:
            observed[r["vehicle_id"]].append(
                (r["detector_id"], float(r["simulation_time_s"]))
            )
    lane_groups = sorted({(n["signal_id"], n["direction"]) for n in env.nodes.values()})
    gi = {g: i for i, g in enumerate(lane_groups)}
    signal_lane = {
        n["sumo_lane"]: (
            gi[(n["signal_id"], n["direction"])],
            n["signal_id"],
            int(n["phase"]),
        )
        for n in env.nodes.values()
    }
    times, frames, contexts = snapshots(
        folder / "traffic_snapshots.jsonl.gz", lane_groups, signal_lane
    )
    edge_index = {e: i for i, e in enumerate(edges)}
    class_index = list(classes)
    route_index = list(route_ids)
    lane_index = {x: i for i, x in enumerate(guidance_lanes)}
    det_index = {d: i for i, d in enumerate(detectors)}
    none = len(detectors)
    # Lazily derive queue/leader state for only frames touched by trajectory windows.
    state_cache = {}

    def states(k):
        if k in state_cache:
            return state_cache[k]
        bylane = defaultdict(list)
        for row in frames[k].values():
            bylane[row[1]].append(row)
        out = {}
        for lane, rows in bylane.items():
            rows.sort(key=lambda r: r[2])
            stopped_ahead = 0
            for j in range(len(rows) - 1, -1, -1):
                vid, _, pos, speed, acc = rows[j]
                leader = rows[j + 1] if j + 1 < len(rows) else None
                gap = max(0, leader[2] - pos - 5) if leader else 100.0
                relative = (speed - leader[3]) if leader else 0
                q = stopped_ahead + (1 if speed < 2 else 0)
                stopped_ahead += int(speed < 2)
                xx, yy = xy(geo, lane, pos)
                meta = signal_lane.get(lane)
                green = -1
                if meta:
                    green = float(
                        contexts[k][2].get(meta[1], {}).get(str(meta[2]), "red")
                        == "green"
                    )
                local = sum(abs(other[2] - pos) <= 50 for other in rows) - 1
                out[vid] = [
                    xx,
                    yy,
                    speed,
                    acc,
                    float(speed < 0.1),
                    q,
                    gap,
                    relative,
                    green,
                    local,
                    1,
                ]
        state_cache[k] = out
        return out

    offsets = list(range(0, context_horizon + 1, context_step))
    relative = list(range(-window, window + 1))
    X = []
    Y = []
    first = []
    second = []
    timing = []
    ids = []
    dropped = 0
    for vid, (v, vt, redges, route_id) in vehicles.items():
        events = sorted(observed.get(vid, []), key=lambda x: x[1])
        if not events:
            continue
        if len(events) > 2:
            dropped += len(events) - 2
        events = events[:2]
        depart = float(v.get("depart"))
        base = list(
            driver_features(
                v,
                vt,
                redges,
                route_id,
                guidance[vid],
                edge_index,
                class_index,
                route_index,
                lane_index,
            )
        )
        # Exogenous environment: aggregate traffic with ego subtracted, plus known signals.
        for off in offsets:
            k = int(np.clip(np.searchsorted(times, depart + off), 0, len(times) - 1))
            agg = contexts[k][0].copy()
            ego = frames[k].get(vid)
            if ego and ego[1] in signal_lane:
                g = signal_lane[ego[1]][0]
                count = agg[g, 0]
                total = agg[g, 2] * count
                agg[g, 0] -= 1
                agg[g, 1] -= float(ego[3] < 0.1)
                agg[g, 2] = (total - ego[3]) / max(1, agg[g, 0])
            base.extend(agg.ravel())
            base.extend(contexts[k][1])
            for role in ("source", "destination"):
                lane = guidance[vid].get(role)
                value = contexts[k][3].get(lane, np.zeros(5, dtype=np.float32)).copy()
                if ego and ego[1] == lane:
                    total = value[2] * value[0]
                    value[0] -= 1
                    value[1] -= float(ego[3] < 0.1)
                    value[2] = (total - ego[3]) / max(1, value[0])
                    if abs(ego[2] - value[3]) < 1e-4:
                        value[3] = value[4]
                base.extend(value[:4])
        target = []
        for slot in range(2):
            if slot >= len(events):
                target.extend([0.0] * (len(relative) * len(TVARS)))
                continue
            center = events[slot][1]
            window_rows = []
            for off in relative:
                k = int(
                    np.clip(np.searchsorted(times, center + off), 0, len(times) - 1)
                )
                window_rows.append(list(states(k).get(vid, [0.0] * 10 + [0.0])))
            middle = window_rows[len(relative) // 2]
            if middle[-1]:
                for row in window_rows:
                    if row[-1]:
                        row[0] -= middle[0]
                        row[1] -= middle[1]
            for row in window_rows:
                target.extend(row)
        X.append(base)
        Y.append(target)
        ids.append(vid)
        first.append(det_index[events[0][0]])
        second.append(det_index[events[1][0]] if len(events) > 1 else none)
        timing.append(
            [events[0][1] - depart, events[1][1] - depart if len(events) > 1 else 0]
        )
    return dict(
        X=np.asarray(X, np.float32),
        trajectory=np.asarray(Y, np.float32),
        first=np.asarray(first, np.int16),
        second=np.asarray(second, np.int16),
        timing=np.asarray(timing, np.float32),
        ids=np.asarray(ids),
        run=np.repeat(folder.name, len(ids)),
        dropped=np.asarray([dropped]),
        schema=np.asarray(
            [
                "forward-v4",
                str(len(detectors)),
                *detectors,
                *edges,
                *classes,
                *route_ids,
                *guidance_lanes,
                *[f"{s}:{d}" for s, d in lane_groups],
                *TVARS,
                str(window),
                str(context_horizon),
                str(context_step),
            ]
        ),
    )


def build(a):
    env = joblib.load(a.run_dir / "environment.joblib")
    folders = []
    for pattern in ("replay_*", "gui_*"):
        for p in a.run_dir.glob(pattern):
            if (
                (p / "traffic_snapshots.jsonl.gz").exists()
                and (p / "candidate.rou.xml").exists()
                and (p / "metrics.json").exists()
                and json.loads((p / "metrics.json").read_text()).get("trainable")
            ):
                folders.append(p)
    folders = sorted(folders)
    all_edges = set()
    all_classes = set()
    all_routes = set()
    for folder in folders:
        for v, vt, edges, rid in route_and_types(folder / "candidate.rou.xml").values():
            all_edges.update(edges)
            all_classes.add(vt.get("vClass", "passenger"))
            all_routes.add(rid)
    edges = sorted(all_edges)
    classes = sorted(all_classes)
    route_ids = sorted(all_routes)
    detectors = sorted(env.nodes)
    guidance_lanes = sorted({n["sumo_lane"] for n in env.nodes.values()})
    geo = geometry()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    lane_groups = sorted({(n["signal_id"], n["direction"]) for n in env.nodes.values()})
    expected_schema = np.asarray(
        [
            "forward-v4",
            str(len(detectors)),
            *detectors,
            *edges,
            *classes,
            *route_ids,
            *guidance_lanes,
            *[f"{s}:{d}" for s, d in lane_groups],
            *TVARS,
            str(a.window),
            str(a.context_horizon),
            str(a.context_step),
        ]
    )
    reports = []
    for folder in folders:
        target = a.output_dir / (folder.name + ".npz")
        if target.exists():
            with np.load(target) as old:
                if not np.array_equal(old["schema"], expected_schema):
                    raise ValueError(
                        f"{target} has a stale schema; use a new output directory so prior data is preserved"
                    )
                reports.append(
                    dict(
                        run=folder.name,
                        rows=len(old["ids"]),
                        dropped_extra_detections=int(old["dropped"][0]),
                        seconds=0,
                        reused=True,
                    )
                )
            print(reports[-1])
            continue
        start = time.perf_counter()
        data = build_one(
            folder,
            env,
            a.window,
            a.context_horizon,
            a.context_step,
            geo,
            edges,
            classes,
            detectors,
            route_ids,
            guidance_lanes,
        )
        np.savez_compressed(target, **data)
        reports.append(
            dict(
                run=folder.name,
                rows=len(data["ids"]),
                dropped_extra_detections=int(data["dropped"][0]),
                seconds=time.perf_counter() - start,
                reused=False,
            )
        )
        print(reports[-1])
    save(
        a.output_dir / "dataset_manifest.json",
        dict(
            source_run=str(a.run_dir),
            window_s=a.window,
            context_horizon_s=a.context_horizon,
            context_step_s=a.context_step,
            runs=reports,
            interpretation="Vehicle ID is grouping metadata only. Traffic features subtract the focal vehicle. Existing snapshots derive acceleration when not explicitly logged.",
        ),
    )


def detector_ids(schema):
    """Decode explicit v4 counts or the fixed 33-channel historical v3 schema.

    Existing model artifacts retain their v3 layout. New dataset builders write
    v4 with a detector-count field; changing detector rosters needs fresh shards.
    """
    values = list(map(str, schema))
    if values[0] == "forward-v4":
        count = int(values[1])
        start = 2
    elif values[0] == "forward-v3":
        count, start = 33, 1
    else:
        raise ValueError("Unsupported forward dataset schema.")
    ids = values[start : start + count]
    if count <= 0 or len(ids) != count or len(set(ids)) != count:
        raise ValueError("Invalid detector roster in forward schema.")
    return ids


def load_data(folder):
    """Concatenate existing compatible shards. No collection or replay is performed."""
    shards = []
    for p in sorted(folder.glob("*.npz")):
        with np.load(p) as d:
            shards.append({k: d[k] for k in d.files})
    if not shards:
        raise ValueError("Build the dataset first")
    if any(not np.array_equal(s["schema"], shards[0]["schema"]) for s in shards):
        raise ValueError("Mixed schemas")
    return {
        k: np.concatenate([s[k] for s in shards])
        for k in ("X", "trajectory", "first", "second", "timing", "ids", "run")
    }, shards[0]["schema"]


def balanced(y, mask, rng):
    groups = [np.flatnonzero(mask & (y == c)) for c in np.unique(y[mask])]
    n = max(map(len, groups))
    return np.concatenate([rng.choice(g, n, replace=True) for g in groups])


def class_weights(y, mask):
    """Balance classes using frequencies from the fitting mask only."""
    values, counts = np.unique(y[mask], return_counts=True)
    weight = {v: len(y[mask]) / (len(values) * n) for v, n in zip(values, counts)}
    return np.asarray([weight[v] for v in y[mask]], dtype=np.float32)


def metrics(name, mask, data, model, tx, ty, pca, detectors):
    X = tx.transform(data["X"][mask])
    p1 = model["first"].predict(X)
    p2 = model["second"].predict(X)
    pt = model["timing"].predict(X)
    z = model["trajectory"].predict(X)
    recon = ty.inverse_transform(pca.inverse_transform(z))
    truth = data["trajectory"][mask]
    avail = truth[:, 10 :: len(TVARS)] > 0.5
    result = dict(
        rows=int(sum(mask)),
        first_detector_accuracy=float(np.mean(p1 == data["first"][mask])),
        second_detector_accuracy=float(np.mean(p2 == data["second"][mask])),
        exact_sequence_accuracy=float(
            np.mean((p1 == data["first"][mask]) & (p2 == data["second"][mask]))
        ),
        first_detection_time_mae_s=float(
            np.mean(abs(pt[:, 0] - data["timing"][mask, 0]))
        ),
    )
    has = data["second"][mask] < len(detectors)
    result["second_detection_time_mae_s"] = (
        float(np.mean(abs(pt[has, 1] - data["timing"][mask][has, 1])))
        if any(has)
        else None
    )
    for j, var in enumerate(TVARS[:-1]):
        m = avail
        actual = truth[:, j :: len(TVARS)]
        pred = recon[:, j :: len(TVARS)]
        result[var + "_mae"] = float(np.mean(abs(pred[m] - actual[m])))
    result["trajectory_latent_rmse"] = float(
        np.sqrt(np.mean((z - pca.transform(ty.transform(truth))) ** 2))
    )
    return name, result


def physical_errors(pred, truth):
    """Measure motion errors only at target timesteps marked available."""
    avail = truth[:, 10 :: len(TVARS)] > 0.5
    out = {}
    for j, var in enumerate(TVARS[:-1]):
        actual = truth[:, j :: len(TVARS)]
        estimate = pred[:, j :: len(TVARS)]
        out[var + "_mae"] = float(np.mean(abs(estimate[avail] - actual[avail])))
    return out


def simple_baseline(data, train_mask, test_mask, detector_count=33):
    """Oracle sequence-group mean: a meaningful hurdle for motion/timing, not detector classification."""
    truth = data["trajectory"]
    global_mean = np.mean(truth[train_mask], axis=0)
    pred = []
    times = []
    groups = defaultdict(list)
    for i in np.flatnonzero(train_mask):
        groups[(int(data["first"][i]), int(data["second"][i]))].append(i)
    means = {
        k: (np.mean(truth[v], axis=0), np.median(data["timing"][v], axis=0))
        for k, v in groups.items()
    }
    global_time = np.median(data["timing"][train_mask], axis=0)
    for i in np.flatnonzero(test_mask):
        m, t = means.get(
            (int(data["first"][i]), int(data["second"][i])), (global_mean, global_time)
        )
        pred.append(m)
        times.append(t)
    pred = np.asarray(pred)
    times = np.asarray(times)
    actual_time = data["timing"][test_mask]
    has = data["second"][test_mask] < detector_count
    return dict(
        first_detection_time_mae_s=float(np.mean(abs(times[:, 0] - actual_time[:, 0]))),
        second_detection_time_mae_s=(
            float(np.mean(abs(times[has, 1] - actual_time[has, 1])))
            if any(has)
            else None
        ),
        **physical_errors(pred, truth[test_mask]),
    )


def train(a):
    """Fit detector, timing, and PCA-motion heads without launching SUMO.

    Exogenous traffic histories condition the prediction; this is not a forecast
    from departure information alone. Group holdouts preserve vehicle identity.
    """
    start = time.perf_counter()
    data, schema = load_data(a.dataset)
    detectors = detector_ids(schema)
    ids = data["ids"]
    runs = data["run"]
    latest = sorted(set(runs))[-1]
    vehicle_test = np.asarray(
        [int(hashlib.sha256(v.encode()).hexdigest()[:8], 16) % 5 == 0 for v in ids]
    )
    fit = ~vehicle_test
    tx = StandardScaler().fit(data["X"][fit])
    ty = StandardScaler().fit(data["trajectory"][fit])
    scaled_y = ty.transform(data["trajectory"][fit])
    components = min(a.latent, scaled_y.shape[0] - 1, scaled_y.shape[1])
    pca = PCA(components, svd_solver="randomized", random_state=41).fit(scaled_y)
    X = tx.transform(data["X"])
    Z = pca.transform(ty.transform(data["trajectory"]))
    rng = np.random.default_rng(41)
    with threadpool_limits(1):
        common = dict(
            hidden_layer_sizes=tuple(a.hidden),
            max_iter=a.epochs,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=41,
            batch_size=256,
        )
        first = MLPClassifier(**common).fit(
            X[fit], data["first"][fit], sample_weight=class_weights(data["first"], fit)
        )
        second = MLPClassifier(**common).fit(
            X[fit],
            data["second"][fit],
            sample_weight=class_weights(data["second"], fit),
        )
        timing = MLPRegressor(**common).fit(X[fit], data["timing"][fit])
        trajectory = MLPRegressor(**common).fit(X[fit], Z[fit])
    model = dict(first=first, second=second, timing=timing, trajectory=trajectory)
    report = dict(
        dataset_rows=len(ids),
        independent_vehicle_ids=len(set(ids)),
        runs=sorted(set(runs)),
        latest_run=latest,
        input_features=data["X"].shape[1],
        trajectory_values=data["trajectory"].shape[1],
        latent_dimensions=components,
        pca_explained_variance=float(sum(pca.explained_variance_ratio_)),
        hidden_layers=a.hidden,
    )
    report["splits"] = dict(
        [
            metrics(
                "heldout_vehicle_ids", vehicle_test, data, model, tx, ty, pca, detectors
            ),
            metrics("training_vehicle_ids", fit, data, model, tx, ty, pca, detectors),
        ]
    )
    # A second, honest chronological/configuration test: no rows from latest replay are fitted.
    run_test = runs == latest
    run_fit = ~run_test
    if any(run_test) and any(run_fit):
        rx = StandardScaler().fit(data["X"][run_fit])
        XX = rx.transform(data["X"])
        ry = StandardScaler().fit(data["trajectory"][run_fit])
        rp = PCA(components, svd_solver="randomized", random_state=43).fit(
            ry.transform(data["trajectory"][run_fit])
        )
        RZ = rp.transform(ry.transform(data["trajectory"]))
        with threadpool_limits(1):
            c = dict(
                hidden_layer_sizes=tuple(a.hidden),
                max_iter=a.epochs,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=43,
                batch_size=256,
            )
            f = MLPClassifier(**c).fit(
                XX[run_fit],
                data["first"][run_fit],
                sample_weight=class_weights(data["first"], run_fit),
            )
            s = MLPClassifier(**c).fit(
                XX[run_fit],
                data["second"][run_fit],
                sample_weight=class_weights(data["second"], run_fit),
            )
            rm = dict(
                first=f,
                second=s,
                timing=MLPRegressor(**c).fit(XX[run_fit], data["timing"][run_fit]),
                trajectory=MLPRegressor(**c).fit(XX[run_fit], RZ[run_fit]),
            )
        report["splits"].update(
            [metrics("heldout_latest_run", run_test, data, rm, rx, ry, rp, detectors)]
        )
    report["fit_seconds"] = time.perf_counter() - start
    report["interpretation"] = (
        "Forward prediction from known SUMO causes. ID excluded from X. Heldout IDs are primary; latest-run test measures configuration shift, not independent real traffic."
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        dict(model=model, x_scaler=tx, y_scaler=ty, pca=pca, schema=schema),
        a.output_dir / "forward.joblib",
    )
    save(a.output_dir / "metrics.json", report)
    print(json.dumps(report, indent=2))


def score(a):
    """Score a trusted saved model on vehicle-ID holdouts; supplied future traffic context makes this conditional prediction."""
    data, schema = load_data(a.dataset)
    saved = joblib.load(a.model)
    detectors = detector_ids(schema)
    if not np.array_equal(schema, saved["schema"]):
        raise ValueError("Model and dataset schemas differ")
    test = np.asarray(
        [
            int(hashlib.sha256(v.encode()).hexdigest()[:8], 16) % 5 == 0
            for v in data["ids"]
        ]
    )
    name, result = metrics(
        "heldout_vehicle_ids",
        test,
        data,
        saved["model"],
        saved["x_scaler"],
        saved["y_scaler"],
        saved["pca"],
        detectors,
    )
    X = saved["x_scaler"].transform(data["X"][test])
    p1 = saved["model"]["first"].predict(X)
    p2 = saved["model"]["second"].predict(X)
    pt = saved["model"]["timing"].predict(X)
    rows = []
    indices = np.flatnonzero(test)
    for k, i in enumerate(indices):
        rows.append(
            dict(
                vehicle_id=data["ids"][i],
                run=data["run"][i],
                actual_first=detectors[data["first"][i]],
                predicted_first=detectors[p1[k]],
                actual_second=(
                    ""
                    if data["second"][i] >= len(detectors)
                    else detectors[data["second"][i]]
                ),
                predicted_second="" if p2[k] >= len(detectors) else detectors[p2[k]],
                actual_first_time_s=float(data["timing"][i, 0]),
                predicted_first_time_s=float(pt[k, 0]),
                actual_second_time_s=float(data["timing"][i, 1]),
                predicted_second_time_s=float(pt[k, 1]),
            )
        )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.output_dir / "heldout_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per = []
    for detector in detectors:
        first_rows = [r for r in rows if r["actual_first"] == detector]
        second_rows = [r for r in rows if r["actual_second"] == detector]
        if first_rows or second_rows:
            per.append(
                dict(
                    detector_id=detector,
                    first_count=len(first_rows),
                    first_accuracy=sum(
                        r["predicted_first"] == detector for r in first_rows
                    )
                    / max(1, len(first_rows)),
                    first_time_mae_s=sum(
                        abs(r["predicted_first_time_s"] - r["actual_first_time_s"])
                        for r in first_rows
                    )
                    / max(1, len(first_rows)),
                    second_count=len(second_rows),
                    second_accuracy=sum(
                        r["predicted_second"] == detector for r in second_rows
                    )
                    / max(1, len(second_rows)),
                    second_time_mae_s=sum(
                        abs(r["predicted_second_time_s"] - r["actual_second_time_s"])
                        for r in second_rows
                    )
                    / max(1, len(second_rows)),
                )
            )
    with (a.output_dir / "per_detector.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    train_mask = ~test
    baseline = simple_baseline(data, train_mask, test, len(detectors))
    Xtruth = saved["y_scaler"].transform(data["trajectory"][test])
    ceiling = saved["y_scaler"].inverse_transform(
        saved["pca"].inverse_transform(saved["pca"].transform(Xtruth))
    )
    report = dict(
        model=result,
        sequence_group_median_baseline=baseline,
        pca_reconstruction_ceiling=physical_errors(ceiling, data["trajectory"][test]),
        caveat="Baseline is given the actual detector sequence and therefore only compares timing/motion, not detector classification.",
    )
    save(a.output_dir / "score.json", report)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run-dir", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--window", type=int, default=20)
    b.add_argument("--context-horizon", type=int, default=180)
    b.add_argument("--context-step", type=int, default=10)
    t = sub.add_parser("train")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    t.add_argument("--latent", type=int, default=32)
    t.add_argument("--epochs", type=int, default=160)
    s = sub.add_parser("score")
    s.add_argument("--dataset", type=Path, required=True)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    build(a) if a.command == "build" else train(a) if a.command == "train" else score(a)


if __name__ == "__main__":
    main()
