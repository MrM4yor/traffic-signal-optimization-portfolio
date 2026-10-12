"""Fixed-location sensor frame for traffic-signal control.

SUMO is only the current sensor adapter. A real deployment can populate the
same numeric frame from lidar/radar/loop detections without changing training
or controller code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

GREEN = {"g", "G"}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def build_sensor_frame(sim, tls_id: str, coverage_m: float = 180.0) -> Dict[str, float]:
    """Build queue, zone, speed, and ETA-bin features from fixed approaches."""
    state = sim.trafficlight.getRedYellowGreenState(tls_id)
    links = sim.trafficlight.getControlledLinks(tls_id)
    lane_green: Dict[str, bool] = {}
    for index, group in enumerate(links):
        green = index < len(state) and state[index] in GREEN
        for link in group:
            if link:
                lane_green[link[0]] = lane_green.get(link[0], False) or green

    records = defaultdict(
        lambda: {
            "queue": 0.0,
            "moving": 0.0,
            "near50": 0.0,
            "near100": 0.0,
            "near150": 0.0,
            "eta3": 0.0,
            "eta6": 0.0,
            "eta10": 0.0,
            "eta15": 0.0,
            "speed_sum": 0.0,
            "min_eta": 60.0,
        }
    )
    edge_served: Dict[str, bool] = {}

    for lane, served in lane_green.items():
        try:
            edge = sim.lane.getEdgeID(lane)
            lane_len = float(sim.lane.getLength(lane))
            vehicle_ids = sim.lane.getLastStepVehicleIDs(lane)
        except Exception:
            continue
        edge_served[edge] = edge_served.get(edge, False) or served
        rec = records[edge]
        for vid in vehicle_ids:
            try:
                distance = max(0.0, lane_len - float(sim.vehicle.getLanePosition(vid)))
                speed = max(0.0, float(sim.vehicle.getSpeed(vid)))
            except Exception:
                continue
            if distance > coverage_m:
                continue
            if distance <= 50.0:
                rec["near50"] += 1.0
            if distance <= 100.0:
                rec["near100"] += 1.0
            if distance <= 150.0:
                rec["near150"] += 1.0
            if speed <= 0.2:
                rec["queue"] += 1.0
                continue
            rec["moving"] += 1.0
            rec["speed_sum"] += speed
            eta = distance / max(speed, 0.5)
            rec["min_eta"] = min(rec["min_eta"], eta)
            if eta <= 3.0:
                rec["eta3"] += 1.0
            if eta <= 6.0:
                rec["eta6"] += 1.0
            if eta <= 10.0:
                rec["eta10"] += 1.0
            if eta <= 15.0:
                rec["eta15"] += 1.0

    features: Dict[str, float] = {"sensor_coverage_m": float(coverage_m)}
    totals = {
        "served": defaultdict(float),
        "opposing": defaultdict(float),
    }
    for edge in sorted(edge_served):
        rec = records[edge]
        side = "served" if edge_served[edge] else "opposing"
        prefix = f"sensor_approach_{_safe_name(edge)}"
        mean_speed = rec["speed_sum"] / rec["moving"] if rec["moving"] else 0.0
        values = {
            "queue_count": rec["queue"],
            "moving_count": rec["moving"],
            "vehicles_within_50m": rec["near50"],
            "vehicles_within_100m": rec["near100"],
            "vehicles_within_150m": rec["near150"],
            "arrivals_eta_le_3s": rec["eta3"],
            "arrivals_eta_le_6s": rec["eta6"],
            "arrivals_eta_le_10s": rec["eta10"],
            "arrivals_eta_le_15s": rec["eta15"],
            "min_arrival_eta_s": rec["min_eta"],
            "mean_moving_speed_mps": mean_speed,
            "estimated_queue_clear_s": rec["queue"] * 2.0,
        }
        for name, value in values.items():
            features[f"{prefix}_{name}"] = float(value)
            totals[side][name] += float(value)

    for side in ("served", "opposing"):
        for name, value in totals[side].items():
            features[f"sensor_{side}_{name}"] = float(value)
    features["sensor_queue_pressure"] = features.get(
        "sensor_opposing_queue_count", 0.0
    ) - features.get("sensor_served_queue_count", 0.0)
    features["sensor_arrival_pressure_6s"] = features.get(
        "sensor_opposing_arrivals_eta_le_6s", 0.0
    ) - features.get("sensor_served_arrivals_eta_le_6s", 0.0)
    return features
