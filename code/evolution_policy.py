#!/usr/bin/env python3
"""Genome, observations, and multi-output residual actions for evolutionary TLS control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Topology-agnostic quantities obtainable from fixed upstream sensors plus the
# signal controller. Values are normalized by fixed physical scales so a genome
# can transfer between traffic regimes and networks without dataset statistics.
FEATURE_SCALES = {
    "sensor_served_queue_count": 20.0,
    "sensor_opposing_queue_count": 20.0,
    "sensor_served_moving_count": 30.0,
    "sensor_opposing_moving_count": 30.0,
    "sensor_served_vehicles_within_50m": 20.0,
    "sensor_opposing_vehicles_within_50m": 20.0,
    "sensor_served_vehicles_within_100m": 35.0,
    "sensor_opposing_vehicles_within_100m": 35.0,
    "sensor_served_vehicles_within_150m": 50.0,
    "sensor_opposing_vehicles_within_150m": 50.0,
    "sensor_served_arrivals_eta_le_3s": 10.0,
    "sensor_opposing_arrivals_eta_le_3s": 10.0,
    "sensor_served_arrivals_eta_le_6s": 15.0,
    "sensor_opposing_arrivals_eta_le_6s": 15.0,
    "sensor_served_arrivals_eta_le_10s": 25.0,
    "sensor_opposing_arrivals_eta_le_10s": 25.0,
    "sensor_served_arrivals_eta_le_15s": 35.0,
    "sensor_opposing_arrivals_eta_le_15s": 35.0,
    "sensor_served_estimated_queue_clear_s": 30.0,
    "sensor_opposing_estimated_queue_clear_s": 30.0,
    "sensor_served_mean_moving_speed_mps": 15.0,
    "sensor_opposing_mean_moving_speed_mps": 15.0,
    "sensor_served_min_arrival_eta_s": 20.0,
    "sensor_opposing_min_arrival_eta_s": 20.0,
    "sensor_current_movement_pressure": 40.0,
    "sensor_next_movement_pressure": 40.0,
    "sensor_next_minus_current_movement_pressure": 40.0,
    "sensor_queue_pressure": 30.0,
    "sensor_arrival_pressure_6s": 20.0,
    "phase_elapsed_s": 60.0,
    "phase_remaining_s": 30.0,
    "num_green_links": 16.0,
}
FEATURES = list(FEATURE_SCALES)
OUTPUTS = [
    "noop_logit",
    "early_logit",
    "extend_logit",
    "early_seconds",
    "extend_seconds",
    "confidence",
]


def condition_bin(features: dict[str, float]) -> str:
    """Categorize local demand from queues and six-second arrival counts; thresholds are heuristic."""
    served_q = float(features.get("sensor_served_queue_count", 0))
    opposing_q = float(features.get("sensor_opposing_queue_count", 0))
    served_a = float(features.get("sensor_served_arrivals_eta_le_6s", 0))
    opposing_a = float(features.get("sensor_opposing_arrivals_eta_le_6s", 0))
    queue_total = served_q + opposing_q
    intensity = queue_total + 2.0 * (served_a + opposing_a)
    density = "sparse" if intensity < 5 else "nominal" if intensity < 20 else "busy"
    queue_bin = "low" if queue_total < 4 else "medium" if queue_total < 15 else "high"
    queue_side = (
        "served"
        if served_q - opposing_q > 2
        else "opposing" if opposing_q - served_q > 2 else "balanced"
    )
    arrival_side = (
        "served"
        if served_a - opposing_a > 1
        else "opposing" if opposing_a - served_a > 1 else "balanced"
    )
    return f"{density}|{queue_bin}|q:{queue_side}|a:{arrival_side}"


def observation(features: dict[str, float], architecture: str) -> np.ndarray:
    """Normalize sensor values by fixed physical scales, clip to [-3, 3], and optionally add polynomial interactions."""
    base = np.array(
        [
            np.clip(float(features.get(name, 0.0)) / scale, -3.0, 3.0)
            for name, scale in FEATURE_SCALES.items()
        ],
        dtype=np.float64,
    )
    if architecture == "linear" or architecture == "mlp":
        return base
    if architecture == "polynomial":
        # Squares capture saturation. These targeted interactions encode the
        # useful traffic questions without the O(n^2) explosion of every pair.
        interactions = np.array(
            [
                base[0] * base[13],  # served queue x opposing arrivals
                base[1] * base[12],  # opposing queue x served arrivals
                base[18] * base[12],  # served clearance x served arrivals
                base[19] * base[13],  # opposing clearance x opposing arrivals
                base[26] * base[29],  # pressure difference x phase elapsed
                base[27] * base[28],  # queue pressure x arrival pressure
                base[29] * base[30],  # elapsed x native remaining
            ]
        )
        return np.concatenate([base, np.square(base), interactions])
    raise ValueError(f"Unknown architecture: {architecture}")


def parameter_count(architecture: str, hidden: int = 16) -> int:
    """Count weights and biases; hidden width must be a positive integer."""
    if not isinstance(hidden, (int, np.integer)) or hidden <= 0:
        raise ValueError("Hidden width must be a positive integer.")
    inputs = len(observation({}, architecture))
    outputs = len(OUTPUTS)
    if architecture in {"linear", "polynomial"}:
        return (inputs + 1) * outputs
    if architecture == "mlp":
        return inputs * hidden + hidden + hidden * outputs + outputs
    raise ValueError(architecture)


def forward(
    genome: np.ndarray, obs: np.ndarray, architecture: str, hidden: int = 16
) -> np.ndarray:
    """Evaluate a flat genome using fixed feature order; reject unknown layouts."""
    if architecture not in {"linear", "polynomial", "mlp"}:
        raise ValueError(f"Unknown architecture: {architecture}")
    genome = np.asarray(genome, dtype=float)
    obs = np.asarray(obs, dtype=float)
    if genome.ndim != 1 or genome.size != parameter_count(architecture, hidden):
        raise ValueError("Genome shape does not match the selected architecture.")
    if obs.ndim != 1 or obs.size != len(observation({}, architecture)):
        raise ValueError("Observation shape does not match the selected architecture.")
    if not np.isfinite(genome).all() or not np.isfinite(obs).all():
        raise ValueError("Genome and observation must be finite.")
    outputs = len(OUTPUTS)
    if architecture in {"linear", "polynomial"}:
        weights = genome[: obs.size * outputs].reshape(obs.size, outputs)
        bias = genome[obs.size * outputs : obs.size * outputs + outputs]
        return obs @ weights + bias
    cursor = 0
    w1 = genome[cursor : cursor + obs.size * hidden].reshape(obs.size, hidden)
    cursor += obs.size * hidden
    b1 = genome[cursor : cursor + hidden]
    cursor += hidden
    w2 = genome[cursor : cursor + hidden * outputs].reshape(hidden, outputs)
    cursor += hidden * outputs
    b2 = genome[cursor : cursor + outputs]
    return np.tanh(obs @ w1 + b1) @ w2 + b2


def sigmoid(value: float) -> float:
    """Map a clipped scalar to a stable probability-like value; this is not calibrated confidence."""
    value = float(np.clip(value, -30, 30))
    return 1.0 / (1.0 + np.exp(-value))


@dataclass(frozen=True)
class ResidualAction:
    kind: str
    early_seconds: float = 0.0
    extend_seconds: float = 0.0
    confidence: float = 0.0
    logits: tuple[float, float, float] = (0.0, 0.0, 0.0)


def decode(
    outputs: np.ndarray,
    max_early_s: float,
    max_extend_s: float,
    confidence_floor: float,
) -> ResidualAction:
    """Bound a six-output proposal; timing limits are seconds, confidence is 0--1."""
    outputs = np.asarray(outputs, dtype=float)
    if outputs.shape != (len(OUTPUTS),) or not np.isfinite(outputs).all():
        raise ValueError("Residual action requires six finite outputs.")
    if not all(np.isfinite(x) and x >= 0 for x in (max_early_s, max_extend_s)):
        raise ValueError("Residual timing limits must be finite and nonnegative.")
    if not np.isfinite(confidence_floor) or not 0 <= confidence_floor <= 1:
        raise ValueError("Confidence floor must be between zero and one.")
    logits = tuple(float(x) for x in outputs[:3])
    winner = int(np.argmax(outputs[:3]))
    confidence = sigmoid(outputs[5])
    if winner == 0 or confidence < confidence_floor:
        return ResidualAction("noop", confidence=confidence, logits=logits)
    if winner == 1:
        return ResidualAction(
            "early",
            early_seconds=max_early_s * sigmoid(outputs[3]),
            confidence=confidence,
            logits=logits,
        )
    return ResidualAction(
        "extend",
        extend_seconds=max_extend_s * sigmoid(outputs[4]),
        confidence=confidence,
        logits=logits,
    )


def action(
    genome,
    features,
    architecture="mlp",
    hidden=16,
    max_early_s=15.0,
    max_extend_s=15.0,
    confidence_floor=0.55,
):
    """Map a sensor frame through a genome to a bounded residual proposal; applying it is the controller adapter's responsibility."""
    obs = observation(features, architecture)
    outputs = forward(np.asarray(genome, dtype=float), obs, architecture, hidden)
    return decode(outputs, max_early_s, max_extend_s, confidence_floor)


def native_genome(architecture: str, hidden: int = 16) -> np.ndarray:
    """Create a zero-weight policy biased toward leaving the native controller unchanged."""
    genome = np.zeros(parameter_count(architecture, hidden), dtype=float)
    # Bias the immutable starting policy strongly toward no-op.
    if architecture in {"linear", "polynomial"}:
        genome[-len(OUTPUTS)] = 4.0
    else:
        genome[-len(OUTPUTS)] = 4.0
    return genome


def align_mlp_genome(reference, donor, hidden=16):
    """Permute donor hidden units into the reference network's coordinate system."""
    from scipy.optimize import linear_sum_assignment

    inputs = len(FEATURES)
    outputs = len(OUTPUTS)

    def unpack(genome):
        cursor = 0
        w1 = genome[cursor : cursor + inputs * hidden].reshape(inputs, hidden)
        cursor += inputs * hidden
        b1 = genome[cursor : cursor + hidden]
        cursor += hidden
        w2 = genome[cursor : cursor + hidden * outputs].reshape(hidden, outputs)
        cursor += hidden * outputs
        b2 = genome[cursor : cursor + outputs]
        return w1, b1, w2, b2

    rw1, rb1, rw2, _ = unpack(np.asarray(reference))
    dw1, db1, dw2, db2 = unpack(np.asarray(donor))
    cost = np.empty((hidden, hidden))
    for i in range(hidden):
        for j in range(hidden):
            cost[i, j] = (
                np.mean(np.square(rw1[:, i] - dw1[:, j]))
                + (rb1[i] - db1[j]) ** 2
                + np.mean(np.square(rw2[i] - dw2[j]))
            )
    rows, cols = linear_sum_assignment(cost)
    order = cols[np.argsort(rows)]
    return np.concatenate([dw1[:, order].ravel(), db1[order], dw2[order].ravel(), db2])


def semantic_crossover(parent_a, parent_b, architecture, hidden, rng):
    """Uniform crossover after transforming equivalent MLP neurons into alignment."""
    a = np.asarray(parent_a, dtype=float)
    b = np.asarray(parent_b, dtype=float)
    if architecture == "mlp":
        b = align_mlp_genome(a, b, hidden)
    mask = rng.random(a.size) < 0.5
    return np.where(mask, a, b)
