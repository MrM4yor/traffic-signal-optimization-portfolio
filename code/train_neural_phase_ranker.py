#!/usr/bin/env python3
"""Fit phase regret offline, selecting models on validation routes only.

Rows for every candidate of one decision stay in the same route group. A sealed
test group is scored only after neural architecture selection. Existing saved
models/results are not rewritten unless this training command is invoked.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase_feature_schema import RESIDUAL_FEATURES


def decision_metrics(df, predictions):
    """Score selected action regret, grouping candidates by complete decision ID."""
    if df.empty:
        raise ValueError("Decision evaluation requires at least one row.")
    scored = df.copy()
    scored["pred"] = predictions
    regrets = []
    for _, group in scored.groupby(["route", "state_time_s", "tls_id"]):
        regrets.append(float(group.loc[group.pred.idxmin()].regret))
    zero_rate = float(np.mean(np.asarray(regrets) <= 1e-9))
    return {
        "states": len(regrets),
        "mean_selected_regret": float(np.mean(regrets)),
        "exact_best_rate": zero_rate,
        "zero_regret_rate": zero_rate,
    }


def split_routes(df):
    """Reserve whole routes: 60% training, 20% validation, 20% final test.

    Sorting makes the split reproducible regardless of CSV row order. Five routes
    are the minimum; route IDs must encode independent scenarios across networks.
    """
    routes = sorted(df.route.unique())
    if len(routes) < 5:
        raise ValueError(
            "At least five independent routes are required for three-way splitting."
        )
    validation = set(routes[1::5])
    test = set(routes[2::5])
    return {
        "train": df[~df.route.isin(validation | test)].copy(),
        "validation": df[df.route.isin(validation)].copy(),
        "test": df[df.route.isin(test)].copy(),
    }


def model_candidates(seed):
    """Return neural candidates and reference tree models with fixed RNG seeds."""
    models = {}
    for name, layers, rate in [
        ("mlp_32_16", (32, 16), 5e-4),
        ("mlp_64_32", (64, 32), 3e-4),
    ]:
        models[name] = TransformedTargetRegressor(
            regressor=make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=layers,
                    activation="relu",
                    learning_rate_init=rate,
                    alpha=1e-3,
                    early_stopping=True,
                    n_iter_no_change=30,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
            transformer=StandardScaler(),
        )
    models["hgb"] = HistGradientBoostingRegressor(
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=1,
        random_state=seed,
    )
    models["extra"] = ExtraTreesRegressor(
        n_estimators=128,
        min_samples_leaf=3,
        max_features=0.8,
        n_jobs=8,
        random_state=seed,
    )
    return models


def evaluate_model(model, rows, columns):
    """Measure regression error and the regret of each selected action."""
    predictions = model.predict(rows[columns].to_numpy())
    return {
        "mae": float(mean_absolute_error(rows.regret, predictions)),
        "r2": float(r2_score(rows.regret, predictions)),
        **decision_metrics(rows, predictions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/multiphase_paired_v10_large.csv")
    parser.add_argument("--out", default="models/neural_phase_ranker_v1.joblib")
    parser.add_argument("--seed", type=int, default=9700000)
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    columns = [f"x_{i}" for i in range(len(RESIDUAL_FEATURES))]
    required = columns + ["route", "state_time_s", "tls_id", "regret"]
    missing = set(required) - set(df.columns)
    if missing:
        parser.error(f"Missing dataset columns: {sorted(missing)}")
    if (
        df[required].isna().any().any()
        or not np.isfinite(df[columns + ["regret"]].to_numpy()).all()
    ):
        parser.error("Dataset contains missing or nonfinite features/targets.")
    groups = split_routes(df)
    report = {
        "split_protocol": "route_train_validation_test_v1",
        "rows": {name: len(rows) for name, rows in groups.items()},
        "routes": {
            name: sorted(rows.route.unique().tolist()) for name, rows in groups.items()
        },
        "models": {},
    }
    fitted = model_candidates(args.seed)
    for name, model in fitted.items():
        started = time.perf_counter()
        model.fit(
            groups["train"][columns].to_numpy(), groups["train"].regret.to_numpy()
        )
        report["models"][name] = {
            "fit_s": time.perf_counter() - started,
            "validation": evaluate_model(model, groups["validation"], columns),
        }
        print(name, report["models"][name], flush=True)
    selected = min(
        (name for name in fitted if name.startswith("mlp")),
        key=lambda name: report["models"][name]["validation"]["mean_selected_regret"],
    )
    # Freeze selection before touching the test set; reference scores are descriptive.
    for name, model in fitted.items():
        report["models"][name]["test"] = evaluate_model(model, groups["test"], columns)
    report["selected_neural"] = selected
    bundle = {
        "model": fitted[selected],
        "features": RESIDUAL_FEATURES,
        "target": "paired_candidate_regret",
        "model_name": selected,
        "report": report,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output, compress=3)
    report["size_mb"] = output.stat().st_size / 1048576
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
