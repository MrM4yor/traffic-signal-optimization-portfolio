#!/usr/bin/env python3
"""Generic four-movement arrival/pressure controller for arbitrary SUMO TLS networks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

import evaluate_multi_intersection as evaluate
import tl_data_common as common

FEATURES = [
    "phase_queue",
    "queue_wait_sum",
    "queue_wait_max",
    "phase_count",
    "phase_braking",
    "due_5",
    "due_10",
    "due_18",
    "eta_min",
    "eta_max",
    "pressure",
    "is_current",
    "green_elapsed_s",
    "switch_cost_s",
    "network_signal_count",
    "neighbor_queue",
]
from phase_feature_schema import RESIDUAL_FEATURES


def residual_vector(detail, elapsed, target, phase_count):
    """Build the ordered candidate-action vector defined in phase_feature_schema; defaults express unavailable context."""
    return [
        float(elapsed),
        float(detail.get("hold", False)),
        float(target / max(phase_count - 1, 1)),
        float(detail.get("expected_stops", 0)),
        float(detail.get("expected_braking", 0)),
        float(detail.get("lane_any_stop", 0)),
        float(detail.get("peak_stop_probability", 0)),
        float(detail.get("high_risk_vehicles", 0)),
        float(detail.get("queue_credit", 0)),
        float(detail.get("switch_penalty", 0)),
        float(len(detail.get("vehicle_ids", []))),
        float(detail.get("next_signal_exists_mean", 0)),
        float(detail.get("next_signal_distance_mean", 1000)),
        float(detail.get("next_signal_served_mean", 0)),
        float(detail.get("next_signal_time_to_switch_mean", 120)),
        float(detail.get("next_signal_count_mean", 0)),
        float(detail.get("next_signal_queue_mean", 0)),
    ]


def green_phases(sim, tls_id):
    """Return green-only phase indices from the selected program, excluding phases containing yellow."""
    logic = sim.trafficlight.getAllProgramLogics(tls_id)[0]
    return [
        i
        for i, phase in enumerate(logic.phases)
        if any(ch in "Gg" for ch in str(phase.state)) and "y" not in str(phase.state)
    ]


def phase_lanes(sim, tls_id, phase_index):
    """Return unique incoming lanes served by a candidate green phase."""
    logic = sim.trafficlight.getAllProgramLogics(tls_id)[0]
    state = str(logic.phases[phase_index].state)
    links = sim.trafficlight.getControlledLinks(tls_id)
    lanes = set()
    for index, char in enumerate(state):
        if char not in "Gg" or index >= len(links):
            continue
        for link in links[index]:
            if link and link[0]:
                lanes.add(str(link[0]))
    return sorted(lanes)


def phase_vehicle_ids(sim, tls_id, phase_index, detector_range):
    """Collect approaching vehicle IDs within the detector range in metres. IDs group observations and are not numeric model inputs."""
    ids = set()
    for lane in phase_lanes(sim, tls_id, phase_index):
        length = float(sim.lane.getLength(lane))
        for vehicle_id in sim.lane.getLastStepVehicleIDs(lane):
            if (
                length - float(sim.vehicle.getLanePosition(vehicle_id))
                <= detector_range
            ):
                ids.add(str(vehicle_id))
    return ids


def phase_features(
    sim,
    tls_id,
    phase_index,
    current_phase,
    elapsed,
    detector_range,
    signal_count,
    switch_cost_s=5.0,
):
    """Summarize queues, waiting, acceleration, ETA, and pressure for one candidate phase; future arrivals use a constant-speed approximation."""
    rows = []
    for lane in phase_lanes(sim, tls_id, phase_index):
        length = float(sim.lane.getLength(lane))
        for vehicle_id in sim.lane.getLastStepVehicleIDs(lane):
            distance = max(0.0, length - float(sim.vehicle.getLanePosition(vehicle_id)))
            if distance > detector_range:
                continue
            speed = max(0.0, float(sim.vehicle.getSpeed(vehicle_id)))
            rows.append(
                (
                    distance,
                    speed,
                    float(sim.vehicle.getAcceleration(vehicle_id)),
                    float(sim.vehicle.getWaitingTime(vehicle_id)),
                )
            )
    etas = [distance / max(speed, 0.1) for distance, speed, _, _ in rows]
    waits = [wait for _, speed, _, wait in rows if speed <= 0.5]
    return {
        "phase_queue": float(len(waits)),
        "queue_wait_sum": float(sum(waits)),
        "queue_wait_max": max(waits, default=0.0),
        "phase_count": float(len(rows)),
        "phase_braking": float(
            sum(accel <= -0.5 and speed > 0.5 for _, speed, accel, _ in rows)
        ),
        "due_5": float(sum(eta <= 5 for eta in etas)),
        "due_10": float(sum(eta <= 10 for eta in etas)),
        "due_18": float(sum(eta <= 18 for eta in etas)),
        "eta_min": min(etas, default=120.0),
        "eta_max": max(etas, default=120.0),
        "pressure": float(common.movement_phase_pressure(tls_id, phase_index)),
        "is_current": float(phase_index == current_phase),
        "green_elapsed_s": float(elapsed),
        "switch_cost_s": 0.0 if phase_index == current_phase else float(switch_cost_s),
        "network_signal_count": float(signal_count),
        "neighbor_queue": 0.0,
    }


def heuristic_score(features, weights):
    """Score one candidate with interpretable weighted demand/pressure terms, subtracting transition cost."""
    return (
        weights["queue"] * features["phase_queue"]
        + weights.get("wait_sum", 0.0) * features["queue_wait_sum"]
        + weights.get("wait_max", 0.0) * features["queue_wait_max"]
        + weights["count"] * features["phase_count"]
        + weights["braking"] * features["phase_braking"]
        + weights["due5"] * features["due_5"]
        + weights["due10"] * features["due_10"]
        + weights["due18"] * features["due_18"]
        + weights["pressure"] * features["pressure"]
        + weights["hold"] * features["is_current"]
        - weights["switch"] * features["switch_cost_s"]
    )


DEFAULT_WEIGHTS = {
    "queue": 8.0,
    "count": 1.0,
    "braking": 3.0,
    "due5": 3.0,
    "due10": 1.5,
    "due18": 0.5,
    "pressure": 1.0,
    "hold": 3.0,
    "switch": 1.0,
    "wait_sum": 0.08,
    "wait_max": 0.20,
}


def probability_phase_score(
    sim, tls_id, target, current, elapsed, args, bundle, histories, return_details=False
):
    """Negative expected stop/braking cost under a legal hold or switch plan."""
    from train_expansion_vehicle_predictor import (
        signal_lanes,
        downstream_lanes,
        next_signal_features,
        BIN_EDGES,
    )

    lanes, indices = signal_lanes(sim, tls_id)
    logic = sim.trafficlight.getAllProgramLogics(tls_id)[0]
    state = str(logic.phases[current].state)
    lane_ids = set(map(str, sim.lane.getIDList()))
    rows_by_lane = {}
    picture = []
    total_count = total_queue = 0
    for lane in lanes:
        length = float(sim.lane.getLength(lane))
        rows = []
        for vid in sim.lane.getLastStepVehicleIDs(lane):
            pos = float(sim.vehicle.getLanePosition(vid))
            distance = max(0.0, length - pos)
            if distance <= args.detector_range_m:
                rows.append((pos, str(vid), distance))
        rows.sort(reverse=True)
        rows_by_lane[lane] = (length, rows)
        total_count += len(rows)
        total_queue += sum(float(sim.vehicle.getSpeed(v)) <= 0.5 for _, v, _ in rows)
        served = any(i < len(state) and state[i] in "Gg" for i in indices[lane])
        for _, vid, d in rows:
            picture.append(
                (
                    d,
                    float(sim.vehicle.getSpeed(vid)),
                    float(sim.vehicle.getAcceleration(vid)),
                    served,
                )
            )
    pic = []
    for group in (True, False):
        for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            selected = [r for r in picture if r[3] == group and lo <= r[0] < hi]
            speeds = [r[1] for r in selected]
            pic += [
                len(selected),
                float(np.mean(speeds)) if speeds else 0.0,
                min(speeds, default=0.0),
                sum(r[2] <= -0.5 and r[1] > 0.1 for r in selected),
            ]
    down = downstream_lanes(sim, tls_id)
    downf = [
        sum(sim.lane.getLastStepVehicleNumber(l) for l in down),
        sum(sim.lane.getLastStepHaltingNumber(l) for l in down),
        (
            float(np.mean([sim.lane.getLastStepOccupancy(l) for l in down]))
            if down
            else 0.0
        ),
        (
            float(np.mean([sim.lane.getLastStepMeanSpeed(l) for l in down]))
            if down
            else 0.0
        ),
    ]
    vectors = []
    vector_lanes = []
    vector_vids = []
    target_wait = 0.0
    target_queue = 0.0
    now = float(sim.simulation.getTime())
    hold = target == current
    remaining = max(0.0, args.max_green_s - elapsed) if hold else 0.0
    for lane, (length, rows) in rows_by_lane.items():
        served_now = any(i < len(state) and state[i] in "Gg" for i in indices[lane])
        target_state = str(logic.phases[target].state)
        served_target = any(
            i < len(target_state) and target_state[i] in "Gg" for i in indices[lane]
        )
        lane_queue = sum(float(sim.vehicle.getSpeed(v)) <= 0.5 for _, v, _ in rows)
        if served_target:
            target_queue += lane_queue
            target_wait += sum(
                float(sim.vehicle.getWaitingTime(v))
                for _, v, _ in rows
                if float(sim.vehicle.getSpeed(v)) <= 0.5
            )
        for idx, (pos, vid, distance) in enumerate(rows):
            speed = max(0.0, float(sim.vehicle.getSpeed(vid)))
            acc = float(sim.vehicle.getAcceleration(vid))
            if speed <= 2 or acc <= -0.5:
                continue
            qa = sum(float(sim.vehicle.getSpeed(v)) <= 0.5 for _, v, _ in rows[:idx])
            eta = min(120.0, distance / max(speed, 0.1))
            clearance = 1.8 * (qa + (speed <= 0.5))
            if idx:
                lp, lid, _ = rows[idx - 1]
                lg = max(0.0, lp - pos - float(sim.vehicle.getLength(lid)))
                ls = float(sim.vehicle.getSpeed(lid))
            else:
                lg = args.detector_range_m
                ls = float(sim.lane.getMaxSpeed(lane))
            route = list(sim.vehicle.getRoute(vid))
            ri = int(sim.vehicle.getRouteIndex(vid))
            ne = route[ri + 1] if ri + 1 < len(route) else ""
            nl = ne + "_0" if ne and ne + "_0" in lane_ids else lane
            turn = np.deg2rad(
                float(sim.lane.getAngle(nl)) - float(sim.lane.getAngle(lane))
            )
            vf = [
                sim.vehicle.getAccel(vid),
                sim.vehicle.getDecel(vid),
                sim.vehicle.getEmergencyDecel(vid),
                sim.vehicle.getMinGap(vid),
                sim.vehicle.getTau(vid),
                sim.vehicle.getImperfection(vid),
                sim.vehicle.getSpeedFactor(vid),
                sim.vehicle.getMaxSpeed(vid),
                sim.lane.getMaxSpeed(nl),
                np.sin(turn),
                np.cos(turn),
            ]
            chain = []
            for peers in (list(reversed(rows[:idx])), rows[idx + 1 :]):
                for rank in range(4):
                    if rank < len(peers):
                        pp, pid, _ = peers[rank]
                        ps = float(sim.vehicle.getSpeed(pid))
                        chain += [
                            max(0.0, abs(pp - pos) - sim.vehicle.getLength(pid)),
                            ps,
                            sim.vehicle.getAcceleration(pid),
                            float(ps <= 0.5),
                        ]
                    else:
                        chain += [
                            args.detector_range_m,
                            sim.lane.getMaxSpeed(lane),
                            0.0,
                            0.0,
                        ]
            expected_features = int(
                getattr(bundle["stop_classifier"], "n_features_in_", 138)
            )
            if expected_features > 138:
                nc = histories.setdefault(
                    "__neighbor_feature_cache__", {"time": None, "values": {}}
                )
                if nc["time"] != now:
                    nc["time"] = now
                    nc["values"] = {}
                key = (str(tls_id), str(vid))
                neighborf = nc["values"].get(key)
                if neighborf is None:
                    neighborf = next_signal_features(
                        sim, vid, tls_id, args.detector_range_m
                    )[: expected_features - 138]
                    nc["values"][key] = neighborf
            else:
                neighborf = []
            base = (
                [
                    distance,
                    speed,
                    acc,
                    pos / max(length, 1),
                    qa,
                    idx - qa,
                    len(rows),
                    lane_queue,
                    total_count - len(rows),
                    total_queue - lane_queue,
                    float(served_now),
                    elapsed,
                    remaining,
                    eta,
                    clearance,
                    eta - clearance,
                    len(sim.trafficlight.getIDList()),
                    current / max(len(logic.phases) - 1, 1),
                    lg,
                    ls,
                    sim.lane.getMaxSpeed(lane),
                ]
                + list(map(float, vf))
                + chain
                + pic
                + downf
                + neighborf
            )
            if hold:
                tg = 1.0 if served_now else 60.0
                tr = max(1.0, remaining) if served_now else 1.0
                fractions = [
                    min(h, remaining) / h if served_now else 0.0
                    for h in (5, 10, 20, 40)
                ]
            else:
                tg = 5.0 if served_target else 60.0
                tr = 1.0 if served_now else (60.0 if served_target else 1.0)
                fractions = [
                    max(0.0, h - 5) / h if served_target else 0.0
                    for h in (5, 10, 20, 40)
                ]
            hist = histories.get(vid, [])

            def past(sec, which, default):
                old = [r for r in hist if r[0] <= now - sec]
                return old[-1][which] if old else default

            recent = [r for r in hist if now - 4 <= r[0] <= now]
            base += (
                [tg, tr]
                + fractions
                + [
                    past(1, 1, speed),
                    past(2, 1, speed),
                    past(4, 1, speed),
                    speed - past(1, 1, speed),
                    speed - past(4, 1, speed),
                    past(1, 2, acc),
                    min([r[1] for r in recent], default=speed),
                    float(np.mean([r[2] for r in recent])) if recent else acc,
                ]
            )
            vectors.append(base)
            vector_lanes.append(lane)
            vector_vids.append(vid)
    release_credit = (
        float(getattr(args, "probability_queue_weight", 0.5)) * target_queue
        + float(getattr(args, "probability_wait_weight", 0.02)) * target_wait
    )
    if not vectors:
        detail = {
            "target_phase": int(target),
            "hold": bool(hold),
            "vehicle_ids": [],
            "stop_probabilities": [],
            "brake_probabilities": [],
            "expected_stops": 0.0,
            "expected_braking": 0.0,
            "queue_credit": float(release_credit),
            "switch_penalty": 0.0,
        }
        return (release_credit, detail) if return_details else release_credit
    X = np.asarray(vectors)
    sp = bundle["stop_classifier"].predict_proba(X)[:, 1]
    bp = bundle["slow_classifier"].predict_proba(X)[:, 1]
    # Phase ranker: retain expected affected-car count, but separately encode
    # correlated lane/queue events so a platoon is not treated as independent
    # Bernoulli trials. The weights are trained end-to-end in SUMO.
    lane_any = 0.0
    for lane in set(vector_lanes):
        ps = sp[np.asarray([x == lane for x in vector_lanes])]
        lane_any += 1.0 - float(np.prod(1.0 - np.clip(ps, 0.0, 1.0)))
    cost = float(
        getattr(args, "rank_stop_sum_weight", 1.0) * sp.sum()
        + getattr(args, "probability_slow_weight", 0.35) * bp.sum()
        + getattr(args, "rank_lane_any_weight", 0.0) * lane_any
        + getattr(args, "rank_peak_weight", 0.0) * float(max(sp, default=0.0))
        + getattr(args, "rank_high_risk_weight", 0.0) * float(np.sum(sp >= 0.5))
        - release_credit
    )
    switch_penalty = (
        0.0 if hold else float(getattr(args, "probability_switch_cost", 0.20))
    )
    cost += switch_penalty
    score = -cost
    detail = {
        "target_phase": int(target),
        "hold": bool(hold),
        "vehicle_ids": vector_vids,
        "stop_probabilities": [float(x) for x in sp],
        "brake_probabilities": [float(x) for x in bp],
        "expected_stops": float(sp.sum()),
        "expected_braking": float(bp.sum()),
        "lane_any_stop": float(lane_any),
        "peak_stop_probability": float(max(sp, default=0.0)),
        "high_risk_vehicles": int(np.sum(sp >= 0.5)),
        "queue_credit": float(release_credit),
        "switch_penalty": float(switch_penalty),
        "score": float(score),
    }
    feature_names = list(bundle.get("features", []))
    for name, key, default in [
        ("next_signal_exists", "next_signal_exists_mean", 0),
        ("next_signal_distance_m", "next_signal_distance_mean", 1000),
        ("next_signal_is_served", "next_signal_served_mean", 0),
        ("next_signal_time_to_switch_s", "next_signal_time_to_switch_mean", 120),
        ("next_signal_count", "next_signal_count_mean", 0),
        ("next_signal_queue", "next_signal_queue_mean", 0),
    ]:
        detail[key] = (
            float(np.mean(X[:, feature_names.index(name)]))
            if name in feature_names
            else float(default)
        )
    return (score, detail) if return_details else score


def no_conflict_queue_release(sim, tls_id, candidates, current_phase, elapsed, args):
    """Serve an aging stopped queue only when the current approach can safely lose green."""
    current = next(row for row in candidates if row[1] == current_phase)
    alternatives = [
        row
        for row in candidates
        if row[1] != current_phase and row[2]["phase_queue"] > 0
    ]
    if not alternatives:
        return None
    target = max(
        alternatives,
        key=lambda row: (row[2]["queue_wait_max"], row[2]["queue_wait_sum"]),
    )
    if target[2]["queue_wait_max"] < float(getattr(args, "release_wait_s", 8.0)):
        return None
    # Strict invariant: every active vehicle must be a stopped member of the target
    # movement, and SUMO must have no pending (not-yet-departed) demand. This mirrors
    # a real deployment's sensor-memory condition "all known vehicles are in this
    # queue" and prevents the gate from creating a stop on another movement.
    active = set(map(str, sim.vehicle.getIDList()))
    target_ids = phase_vehicle_ids(sim, tls_id, target[1], args.detector_range_m)
    all_target_stopped = (
        bool(active)
        and active.issubset(target_ids)
        and all(float(sim.vehicle.getSpeed(vehicle_id)) <= 0.5 for vehicle_id in active)
    )
    no_pending_departures = int(sim.simulation.getMinExpectedNumber()) == len(active)
    current_safe = (
        current[2]["phase_count"] <= 0 and all_target_stopped and no_pending_departures
    )
    return target[1] if current_safe and elapsed >= args.min_green_s else None


@dataclass
class SignalState:
    phase: int
    phase_start: float
    pending_target: int | None = None
    release_at: float = 0.0


def request_target(sim, tls_id, state, target, now, yellow_s=3.0, all_red_s=2.0):
    """Request a green through the validated green/yellow/all-red program layout.

    This controller supports generated expansion networks with explicit clearance
    phases. A pending request cannot be replaced halfway through its clearance.
    """
    if state.pending_target is not None:
        raise ValueError("Cannot replace a target while clearance is pending.")
    if target == state.phase:
        return False
    logic = sim.trafficlight.getAllProgramLogics(tls_id)[0]
    if target not in green_phases(sim, tls_id):
        raise ValueError(f"{tls_id}: target {target} is not a green phase")
    yellow = state.phase + 1
    if yellow >= len(logic.phases) or "y" not in str(logic.phases[yellow].state):
        raise RuntimeError(
            f"{tls_id}: green phase {state.phase} has no following yellow"
        )
    all_red = yellow + 1
    if all_red >= len(logic.phases) or any(
        c in "Ggy" for c in str(logic.phases[all_red].state)
    ):
        raise ValueError(f"{tls_id}: expected an explicit all-red phase after yellow")
    if not all(np.isfinite(x) and x > 0 for x in (yellow_s, all_red_s)):
        raise ValueError("Clearance durations must be finite and positive.")
    if not np.isclose(float(logic.phases[all_red].duration), all_red_s):
        raise ValueError(
            f"{tls_id}: all-red duration differs from the requested clearance"
        )
    sim.trafficlight.setPhase(tls_id, yellow)
    sim.trafficlight.setPhaseDuration(tls_id, yellow_s)
    state.pending_target = target
    state.release_at = now + yellow_s + all_red_s
    return True


def run(args, weights=None, model_bundle=None, action_residual_bundle=None):
    if getattr(args, "gui", False):
        import traci as sim
    else:
        import libsumo as sim
    common.traci = sim
    evaluate.start(sim, args, Path(args.route), args.sumo_seed)
    if getattr(args, "gui", False) and getattr(args, "gui_label", None):
        from run_stop_predictor import add_scene_badge, label_gui_window

        label_gui_window(args.gui_label)
        add_scene_badge(sim, getattr(args, "gui_role", "model"))
    if getattr(args, "start_barrier", None):
        import time

        while not Path(args.start_barrier).exists():
            time.sleep(0.05)
    tls_ids = list(sim.trafficlight.getIDList())
    weights = weights or DEFAULT_WEIGHTS
    states = {
        tls: SignalState(int(sim.trafficlight.getPhase(tls)), 0.0) for tls in tls_ids
    }
    greens = {tls: green_phases(sim, tls) for tls in tls_ids}
    recorder = common.RolloutMetrics()
    recorder.update()
    switches = 0
    next_decision = 0.0
    histories = {}
    decision_log = []
    pending_diagnostics = []
    rng = np.random.default_rng(int(args.sumo_seed) + 9173)
    try:
        while sim.simulation.getMinExpectedNumber() > 0:
            sim.simulationStep()
            recorder.update()
            now = float(sim.simulation.getTime())
            if model_bundle and "stop_classifier" in model_bundle:
                for vid in sim.vehicle.getIDList():
                    h = histories.setdefault(str(vid), [])
                    h.append(
                        (
                            now,
                            float(sim.vehicle.getSpeed(vid)),
                            float(sim.vehicle.getAcceleration(vid)),
                        )
                    )
                    if len(h) > 6:
                        del h[:-6]
            # Close realized-outcome windows without using future information in control.
            active_ids = set(map(str, sim.vehicle.getIDList()))
            for record in pending_diagnostics:
                if record.get("closed"):
                    continue
                for vid in record["tracked_vehicle_ids"]:
                    if vid not in active_ids:
                        continue
                    speed = float(sim.vehicle.getSpeed(vid))
                    acc = float(sim.vehicle.getAcceleration(vid))
                    if speed <= 0.1:
                        record["realized_stops"][vid] = True
                    if speed > 0.1 and acc <= -0.5:
                        record["realized_braking"][vid] = True
                if now + 1e-9 >= record["outcome_window_end_s"]:
                    record["closed"] = True
                    record["realized_stop_count"] = sum(
                        record["realized_stops"].values()
                    )
                    record["realized_braking_count"] = sum(
                        record["realized_braking"].values()
                    )
            for tls, state in states.items():
                current = int(sim.trafficlight.getPhase(tls))
                if state.pending_target is not None and now + 1e-9 >= state.release_at:
                    sim.trafficlight.setPhase(tls, state.pending_target)
                    state.phase = state.pending_target
                    state.phase_start = now
                    state.pending_target = None
                    continue
                if current != state.phase and current in greens[tls]:
                    state.phase = current
                    state.phase_start = now
            if now + 1e-9 < next_decision:
                continue
            next_decision = now + args.decision_period_s
            for tls, state in states.items():
                if state.pending_target is not None or state.phase not in greens[tls]:
                    continue
                elapsed = now - state.phase_start
                if elapsed < args.min_green_s:
                    continue
                candidates = []
                for target in greens[tls]:
                    feats = phase_features(
                        sim,
                        tls,
                        target,
                        state.phase,
                        elapsed,
                        args.detector_range_m,
                        len(tls_ids),
                    )
                    if model_bundle is None:
                        score = heuristic_score(feats, weights)
                    elif "stop_classifier" in model_bundle:
                        if (
                            getattr(args, "decision_log", None)
                            or action_residual_bundle is not None
                        ):
                            score, detail = probability_phase_score(
                                sim,
                                tls,
                                target,
                                state.phase,
                                elapsed,
                                args,
                                model_bundle,
                                histories,
                                True,
                            )
                            feats = dict(feats)
                            feats["prediction_detail"] = detail
                            if action_residual_bundle is not None:
                                phase_count = len(
                                    sim.trafficlight.getAllProgramLogics(tls)[0].phases
                                )
                                rv = np.asarray(
                                    [
                                        residual_vector(
                                            detail, elapsed, target, phase_count
                                        )
                                    ]
                                )
                                raw_correction = float(
                                    action_residual_bundle["model"].predict(rv)[0]
                                )
                                if (
                                    action_residual_bundle.get("target")
                                    == "paired_candidate_regret"
                                ):
                                    correction = (
                                        float(
                                            getattr(args, "action_residual_scale", 1.0)
                                        )
                                        * float(np.clip(raw_correction, 0.0, 500.0))
                                        / 100.0
                                    )
                                else:
                                    correction = float(
                                        getattr(args, "action_residual_scale", 1.0)
                                    ) * float(np.clip(raw_correction, -1.0, 1.0))
                                score -= correction
                                detail["residual_cost_correction"] = correction
                                detail["score"] = float(score)
                        else:
                            score = probability_phase_score(
                                sim,
                                tls,
                                target,
                                state.phase,
                                elapsed,
                                args,
                                model_bundle,
                                histories,
                            )
                    else:
                        matrix = np.asarray(
                            [[feats[name] for name in model_bundle["features"]]]
                        )
                        score = -float(model_bundle["regressor"].predict(matrix)[0])
                    candidates.append((score, target, feats))
                candidates.sort(reverse=True, key=lambda row: row[0])
                release_target = (
                    no_conflict_queue_release(
                        sim, tls, candidates, state.phase, elapsed, args
                    )
                    if getattr(args, "queue_release", False)
                    else None
                )
                explore = getattr(
                    args, "diagnostic_exploration_rate", 0.0
                ) > 0 and rng.random() < float(args.diagnostic_exploration_rate)
                target = (
                    int(rng.choice([row[1] for row in candidates]))
                    if explore
                    else (
                        release_target
                        if release_target is not None
                        else candidates[0][1]
                    )
                )
                if elapsed >= args.max_green_s and target == state.phase:
                    target = next(
                        (row[1] for row in candidates if row[1] != state.phase), target
                    )
                if getattr(args, "decision_log", None):
                    selected = next(row for row in candidates if row[1] == target)
                    detail = selected[2].get("prediction_detail", {})
                    record = {
                        "time_s": now,
                        "tls_id": str(tls),
                        "current_phase": int(state.phase),
                        "elapsed_green_s": float(elapsed),
                        "selected_phase": int(target),
                        "selection_reason": (
                            "exploration"
                            if explore
                            else (
                                "queue_release"
                                if release_target is not None
                                else (
                                    "max_green"
                                    if elapsed >= args.max_green_s
                                    else "model"
                                )
                            )
                        ),
                        "candidates": [
                            {
                                "phase": int(row[1]),
                                "score": float(row[0]),
                                **row[2].get("prediction_detail", {}),
                            }
                            for row in candidates
                        ],
                        "tracked_vehicle_ids": list(detail.get("vehicle_ids", [])),
                        "outcome_window_end_s": now
                        + float(getattr(args, "diagnostic_horizon_s", 20.0)),
                        "realized_stops": {
                            v: False for v in detail.get("vehicle_ids", [])
                        },
                        "realized_braking": {
                            v: False for v in detail.get("vehicle_ids", [])
                        },
                        "closed": False,
                    }
                    decision_log.append(record)
                    pending_diagnostics.append(record)
                if target != state.phase and request_target(
                    sim, tls, state, target, now
                ):
                    switches += 1
                elif target == state.phase:
                    sim.trafficlight.setPhaseDuration(
                        tls, args.decision_period_s + 0.01
                    )
        metrics = recorder.metrics()
        metrics.update(
            {
                "switch_count": switches,
                "elapsed_s": float(sim.simulation.getTime()),
                "throughput_veh": recorder.metrics()["arrived"],
            }
        )
        if getattr(args, "decision_log", None):
            for record in decision_log:
                if not record.get("closed"):
                    record["realized_stop_count"] = sum(
                        record["realized_stops"].values()
                    )
                    record["realized_braking_count"] = sum(
                        record["realized_braking"].values()
                    )
                record.pop("realized_stops", None)
                record.pop("realized_braking", None)
                record.pop("closed", None)
            out = Path(args.decision_log)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {
                        "route": str(args.route),
                        "sumo_seed": int(args.sumo_seed),
                        "metrics": metrics,
                        "decisions": decision_log,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return metrics
    finally:
        try:
            sim.close(False)
        except TypeError:
            sim.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--route", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--action-residual-model", default=None)
    p.add_argument("--weights-json", default=None)
    p.add_argument("--action-residual-scale", type=float, default=0.25)
    p.add_argument("--weights-file", default=None)
    p.add_argument("--sumo-seed", type=int, default=2200000)
    p.add_argument("--detector-range-m", type=float, default=220)
    p.add_argument("--decision-period-s", type=float, default=1)
    p.add_argument("--min-green-s", type=float, default=8)
    p.add_argument("--max-green-s", type=float, default=50)
    p.add_argument("--release-wait-s", type=float, default=1)
    p.add_argument("--clearance-s", type=float, default=5)
    p.add_argument(
        "--queue-release", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--probability-slow-weight", type=float, default=0.35)
    p.add_argument("--probability-queue-weight", type=float, default=0.5)
    p.add_argument("--probability-wait-weight", type=float, default=0.02)
    p.add_argument("--probability-switch-cost", type=float, default=0.20)
    p.add_argument("--rank-stop-sum-weight", type=float, default=1.0)
    p.add_argument("--rank-lane-any-weight", type=float, default=0.0)
    p.add_argument("--rank-peak-weight", type=float, default=0.0)
    p.add_argument("--rank-high-risk-weight", type=float, default=0.0)
    p.add_argument("--out", default=None)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--gui-delay", type=int, default=40)
    p.add_argument("--gui-autostart", action="store_true")
    p.add_argument("--gui-label", default=None)
    p.add_argument("--window-pos", default=None)
    p.add_argument(
        "--gui-role", choices=["model", "baseline", "native"], default="model"
    )
    p.add_argument("--start-barrier", default=None)
    p.add_argument("--decision-log", default=None)
    p.add_argument("--diagnostic-horizon-s", type=float, default=20)
    p.add_argument("--diagnostic-exploration-rate", type=float, default=0.0)
    args = p.parse_args()
    args.delay_detector_range = args.detector_range_m
    if args.weights_file:
        payload = json.loads(Path(args.weights_file).read_text(encoding="utf-8"))
        weights = payload.get("best", payload).get("weights", payload)
    else:
        weights = (
            json.loads(args.weights_json) if args.weights_json else DEFAULT_WEIGHTS
        )
    result = run(
        args,
        weights,
        joblib.load(args.model) if args.model else None,
        joblib.load(args.action_residual_model) if args.action_residual_model else None,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
