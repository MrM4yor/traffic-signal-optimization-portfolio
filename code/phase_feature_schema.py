"""Ordered candidate-action inputs shared by collection, training, and inference.

Changing order requires a new dataset/model schema. Timing and distance features
are seconds and metres; queue/vehicle quantities are counts.
"""

RESIDUAL_FEATURES = [
    "elapsed_green_s",
    "is_hold",
    "target_phase_fraction",
    "expected_stops",
    "expected_braking",
    "lane_any_stop",
    "peak_stop_probability",
    "high_risk_vehicles",
    "queue_credit",
    "switch_penalty",
    "tracked_vehicle_count",
    "next_signal_exists_mean",
    "next_signal_distance_mean",
    "next_signal_served_mean",
    "next_signal_time_to_switch_mean",
    "next_signal_count_mean",
    "next_signal_queue_mean",
]
