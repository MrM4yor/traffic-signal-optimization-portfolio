#!/usr/bin/env python3
"""Generate same-state multi-phase candidate labels with incumbent continuation."""

from __future__ import annotations
import argparse, copy, csv, json, tempfile
from pathlib import Path
import joblib, numpy as np
import evaluate_multi_intersection as evaluate
import tl_data_common as common
from multi_phase_controller import (
    SignalState,
    green_phases,
    phase_features,
    probability_phase_score,
    request_target,
    residual_vector,
)

SPECS = [
    ("single_arterial_3lane", 240),
    ("corridor_3x1_2lane", 150),
    ("grid_2x2_2lane", 130),
    ("corridor_5x1_3lane", 120),
    ("grid_3x2_2lane", 100),
]


def args_for(net, route, seed):
    return type(
        "Args",
        (),
        {
            "net": str(net),
            "route": str(route),
            "sumo_seed": seed,
            "gui": False,
            "gui_delay": 0,
            "gui_autostart": False,
            "delay_detector_range": 220,
            "detector_range_m": 220,
            "decision_period_s": 1,
            "min_green_s": 8,
            "max_green_s": 50,
            "queue_release": False,
            "release_wait_s": 1,
            "probability_slow_weight": 0.35,
            "probability_queue_weight": 0.5,
            "probability_wait_weight": 0.02,
            "probability_switch_cost": 0.2,
            "rank_stop_sum_weight": 1.0,
            "rank_lane_any_weight": 0.0,
            "rank_peak_weight": 0.0,
            "rank_high_risk_weight": 0.0,
        },
    )


def advance_policy(sim, args, bundle, states, greens, histories, end):
    """Continue a restored branch with the frozen incumbent and integrate short-horizon outcome metrics."""
    rec = common.RolloutMetrics()
    rec.update()
    next_decision = float(sim.simulation.getTime()) + 1
    while (
        sim.simulation.getMinExpectedNumber() > 0
        and float(sim.simulation.getTime()) < end
    ):
        sim.simulationStep()
        rec.update()
        now = float(sim.simulation.getTime())
        for vid in sim.vehicle.getIDList():
            h = histories.setdefault(str(vid), [])
            h.append(
                (
                    now,
                    float(sim.vehicle.getSpeed(vid)),
                    float(sim.vehicle.getAcceleration(vid)),
                )
            )
            h[:] = h[-6:]
        for tls, s in states.items():
            cur = int(sim.trafficlight.getPhase(tls))
            if s.pending_target is not None and now + 1e-9 >= s.release_at:
                sim.trafficlight.setPhase(tls, s.pending_target)
                s.phase = s.pending_target
                s.phase_start = now
                s.pending_target = None
                continue
            if cur != s.phase and cur in greens[tls]:
                s.phase = cur
                s.phase_start = now
        if now + 1e-9 < next_decision:
            continue
        next_decision = now + 1
        for tls, s in states.items():
            if s.pending_target is not None or s.phase not in greens[tls]:
                continue
            elapsed = now - s.phase_start
            if elapsed < args.min_green_s:
                continue
            candidates = []
            for target in greens[tls]:
                candidates.append(
                    (
                        probability_phase_score(
                            sim, tls, target, s.phase, elapsed, args, bundle, histories
                        ),
                        target,
                    )
                )
            target = max(candidates)[1]
            if elapsed >= args.max_green_s and target == s.phase:
                target = next(
                    (x[1] for x in sorted(candidates, reverse=True) if x[1] != s.phase),
                    target,
                )
            if target != s.phase:
                request_target(sim, tls, s, target, now)
            else:
                sim.trafficlight.setPhaseDuration(tls, 1.01)
    return rec.metrics()


def cost(m):
    """Combine stops, deceleration, waiting, time-loss proxy, and CO2 into the experiment's weighted scalar cost."""
    return (
        100 * m["stops_created"]
        + m["decel_sum"]
        + 0.5 * m["waiting_time_s"]
        + 0.2 * m["time_loss_s"]
        + 1e-5 * m["co2_mg"]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="models/expansion_vehicle_predictor_v10_mlp_refined.joblib"
    )
    p.add_argument("--routes-per-network", type=int, default=1)
    p.add_argument("--duration-s", type=float, default=120)
    p.add_argument("--horizon-s", type=float, default=30)
    p.add_argument("--label-period-s", type=float, default=15)
    p.add_argument("--max-states-per-route", type=int, default=10)
    p.add_argument("--max-clearance-s", type=float, default=300)
    p.add_argument(
        "--networks", default=None, help="Comma-separated subset for smoke tests."
    )
    p.add_argument("--seed", type=int, default=9500000)
    p.add_argument("--out", default="data/multiphase_paired_v10.csv")
    a = p.parse_args()
    rows = []
    import libsumo as sim

    common.traci = sim
    bundle = joblib.load(a.model)
    state_dir = Path("logs/paired_states")
    state_dir.mkdir(parents=True, exist_ok=True)
    selected = set(a.networks.split(",")) if a.networks else None
    for ni, (name, demand) in enumerate(SPECS):
        if selected is not None and name not in selected:
            continue
        net = Path("net/expansion") / f"{name}.clearance2.net.xml"
        for ri in range(a.routes_per_network):
            seed = a.seed + ni * 100 + ri
            route = (
                Path("data/routes/multiphase_paired")
                / name
                / f"route_{ri}_seed{seed}.rou.xml"
            )
            evaluate.make_route(net, route, a.duration_s, demand, seed)
            args = args_for(net, route, seed + 50000)
            evaluate.start(sim, args, route, args.sumo_seed)
            tls_ids = list(sim.trafficlight.getIDList())
            states = {
                t: SignalState(int(sim.trafficlight.getPhase(t)), 0.0) for t in tls_ids
            }
            greens = {t: green_phases(sim, t) for t in tls_ids}
            histories = {}
            next_label = 15.0
            labels = 0
            try:
                while (
                    sim.simulation.getMinExpectedNumber() > 0
                    and float(sim.simulation.getTime())
                    < a.duration_s + a.max_clearance_s
                ):
                    sim.simulationStep()
                    now = float(sim.simulation.getTime())
                    for vid in sim.vehicle.getIDList():
                        h = histories.setdefault(str(vid), [])
                        h.append(
                            (
                                now,
                                float(sim.vehicle.getSpeed(vid)),
                                float(sim.vehicle.getAcceleration(vid)),
                            )
                        )
                        h[:] = h[-6:]
                    for tls, s in states.items():
                        cur = int(sim.trafficlight.getPhase(tls))
                        if s.pending_target is not None and now + 1e-9 >= s.release_at:
                            sim.trafficlight.setPhase(tls, s.pending_target)
                            s.phase = s.pending_target
                            s.phase_start = now
                            s.pending_target = None
                            continue
                        if cur != s.phase and cur in greens[tls]:
                            s.phase = cur
                            s.phase_start = now
                    due = now + 1e-9 >= next_label and labels < a.max_states_per_route
                    if due:
                        next_label = now + a.label_period_s
                        for tls, s in states.items():
                            elapsed = now - s.phase_start
                            if (
                                s.pending_target is not None
                                or s.phase not in greens[tls]
                                or elapsed < args.min_green_s
                                or elapsed >= args.max_green_s
                            ):
                                continue
                            snapshot = state_dir / f"state_{seed}_{labels}_{tls}.xml"
                            sim.simulation.saveState(str(snapshot))
                            candidates = []
                            for target in greens[tls]:
                                score, detail = probability_phase_score(
                                    sim,
                                    tls,
                                    target,
                                    s.phase,
                                    elapsed,
                                    args,
                                    bundle,
                                    histories,
                                    True,
                                )
                                candidates.append((target, score, detail))
                            branch = []
                            for target, score, detail in candidates:
                                sim.simulation.loadState(str(snapshot))
                                bs = copy.deepcopy(states)
                                bh = copy.deepcopy(histories)
                                start = float(sim.simulation.getTime())
                                if target != bs[tls].phase:
                                    request_target(sim, tls, bs[tls], target, start)
                                else:
                                    sim.trafficlight.setPhaseDuration(tls, 1.01)
                                metrics = advance_policy(
                                    sim,
                                    args,
                                    bundle,
                                    bs,
                                    greens,
                                    bh,
                                    start + a.horizon_s,
                                )
                                branch.append(
                                    (target, score, detail, metrics, cost(metrics))
                                )
                            sim.simulation.loadState(str(snapshot))
                            best = min(x[4] for x in branch)
                            for target, score, detail, metrics, c in branch:
                                row = {
                                    "network": name,
                                    "route": str(route),
                                    "state_time_s": now,
                                    "tls_id": str(tls),
                                    "current_phase": s.phase,
                                    "target_phase": target,
                                    "cost": c,
                                    "regret": c - best,
                                    **{
                                        f"x_{i}": v
                                        for i, v in enumerate(
                                            residual_vector(
                                                detail,
                                                elapsed,
                                                target,
                                                max(greens[tls]) + 1,
                                            )
                                        )
                                    },
                                }
                                for k, v in metrics.items():
                                    row[f"outcome_{k}"] = v
                                rows.append(row)
                            labels += 1
                    # Continue the unbranched episode with the incumbent at each legal second.
                    for tls, s in states.items():
                        elapsed = now - s.phase_start
                        if (
                            s.pending_target is not None
                            or s.phase not in greens[tls]
                            or elapsed < args.min_green_s
                        ):
                            continue
                        cand = [
                            (
                                probability_phase_score(
                                    sim,
                                    tls,
                                    t,
                                    s.phase,
                                    elapsed,
                                    args,
                                    bundle,
                                    histories,
                                ),
                                t,
                            )
                            for t in greens[tls]
                        ]
                        target = max(cand)[1]
                        if target != s.phase:
                            request_target(sim, tls, s, target, now)
                        else:
                            sim.trafficlight.setPhaseDuration(tls, 1.01)
            finally:
                try:
                    sim.close(False)
                except TypeError:
                    sim.close()
            print(name, ri, "labeled states", labels, flush=True)
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with o.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"out": str(o), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
