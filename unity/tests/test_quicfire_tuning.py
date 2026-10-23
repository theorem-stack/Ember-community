"""Defaults and overrides for the allowlisted QuicFire constants.

Run from the repo root:
    python tests/test_quicfire_tuning.py
"""

import json
import os
import sys
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_ROOT = os.path.join(ROOT, "Assets", "Python4Unity_Sample", "Python")
if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)

from QuicFire.QuicFireConstants import QuicFireConstants
from QuicFire.QuicFireTuning import (
    TUNABLE_KEYS,
    apply_document,
    apply_tuning,
    plume_k,
    plume_rise,
    plume_tilt_length,
    restore_defaults,
)

CONFIG_PATH = os.path.join(
    PYTHON_ROOT,
    "TrainedModels",
    "firefighter_policy_config_20250804_143517.json",
)


def _snapshot():
    return {key: getattr(QuicFireConstants, key) for key in TUNABLE_KEYS}


def test_defaults_match_the_reviewed_numbers():
    restore_defaults()
    assert QuicFireConstants.DT == 20.0
    assert QuicFireConstants.H_SF == 0.5
    assert QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD == 0.05
    assert QuicFireConstants.PENALTY_ACTIVE_FIRE_OUTSIDE_BB == -100.0
    assert QuicFireConstants.PENALTY_FUEL_CONSUMED_OUTSIDE_BB == -80.0
    assert QuicFireConstants.REWARD_FUEL_CONSUMED_INSIDE_BB == 80.0
    assert QuicFireConstants.PENALTY_GLOBAL_WATER_USAGE == -5.0
    assert QuicFireConstants.PENALTY_GLOBAL_RETARDANT_USAGE == -7.0
    assert QuicFireConstants.PENALTY_GLOBAL_AIR_SUPPORT_USAGE == -10.0
    assert QuicFireConstants.PENALTY_TIME_STEP == -0.5
    assert QuicFireConstants.REWARD_MISSION_CONTAINED_SUCCESS == 100.0
    assert QuicFireConstants.PENALTY_MISSION_FAILURE_TIMEOUT == -100.0
    assert QuicFireConstants.PENALTY_MISSION_FAILURE_SPREAD == -150.0
    assert QuicFireConstants.REWARD_AGENT_FIRE_REDUCTION == 5.0
    assert QuicFireConstants.PENALTY_RESOURCE_INVALID_ACTION == -2.0
    assert QuicFireConstants.MAX_GLOBAL_REWARD_CLIP == 100.0
    assert QuicFireConstants.MAX_AGENT_REWARD_CLIP == 10.0
    assert QuicFireConstants.MOVEMENT_PENALTY == -0.5
    assert QuicFireConstants.DEFAULT_REWARD_ALPHA == 0.8
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.8
    assert QuicFireConstants.PLUME_HEIGHT_FACTOR == 0.015
    assert QuicFireConstants.PLUME_POWER == 0.6
    assert QuicFireConstants.PLUME_CONSTANT == 0.377
    assert QuicFireConstants.PLUME_K == 4.0
    assert QuicFireConstants.PLUME_K == QuicFireConstants.LAMBDA_EP / QuicFireConstants.LAMBDA_AIR
    assert QuicFireConstants.PLUME_TILT_SCALE == 1.0
    assert QuicFireConstants.PLUME_TILT_WIND_FLOOR == 1e-6
    assert plume_k() == 4.0

    heat = 1_000_000.0
    assert plume_rise(heat) == 0.5 + 0.015 * (heat ** 0.6)
    width = 2.0
    wind = 3.0
    expected_ell = 10.0 * (1.0 + (1.0 + width ** 2 / wind ** 2) ** 0.5)
    assert plume_tilt_length(10.0, width, wind) == expected_ell
    calm = 2.0 * (1.0 + (1.0 + 1.0 / 1e-6) ** 0.5)
    assert plume_tilt_length(2.0, 1.0, 0.0) == calm


def test_partial_override_then_restore():
    restore_defaults()
    apply_tuning({
        "DT": 10,
        "H_SF": "1.5",
        "WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD": 0.1,
        "not_a_constant": 3,
    })
    assert QuicFireConstants.DT == 10.0
    assert isinstance(QuicFireConstants.DT, float)
    assert QuicFireConstants.H_SF == 1.5
    assert QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD == 0.1
    assert QuicFireConstants.PENALTY_TIME_STEP == -0.5
    assert QuicFireConstants.PLUME_K == 4.0

    apply_tuning({"DEFAULT_REWARD_ALPHA": 0.5})
    assert QuicFireConstants.DEFAULT_REWARD_ALPHA == 0.5
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.5
    assert QuicFireConstants.DT == 10.0

    apply_tuning({"DT": 0, "PLUME_TILT_WIND_FLOOR": -1})
    assert QuicFireConstants.DT == 10.0
    assert QuicFireConstants.PLUME_TILT_WIND_FLOOR == 1e-6

    restore_defaults()
    assert QuicFireConstants.DT == 20.0
    assert QuicFireConstants.H_SF == 0.5
    assert QuicFireConstants.DEFAULT_REWARD_ALPHA == 0.8
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.8
    assert QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD == 0.05


def test_shipped_config_is_a_noop_and_tuning_overrides_it():
    restore_defaults()
    with open(CONFIG_PATH, "r") as handle:
        document = json.load(handle)
    before = _snapshot()
    apply_document(document)
    after = _snapshot()
    changed = {key: (before[key], after[key]) for key in before if before[key] != after[key]}
    assert changed == {}
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.8

    apply_document({
        "quicfire_constants": {"DT": 20, "H_SF": 0.5, "SURFACE_HEIGHTS": {"200": 150}},
        "tuning": {"DT": 15, "PLUME_K": 2},
    })
    assert QuicFireConstants.DT == 15.0
    assert QuicFireConstants.H_SF == 0.5
    assert QuicFireConstants.PLUME_K == 2.0
    assert QuicFireConstants.PENALTY_TIME_STEP == -0.5
    restore_defaults()


def test_entry_points_apply_tuning_before_dividing_by_dt():
    infer = open(os.path.join(PYTHON_ROOT, "InferFirefighterOptionCritic.py"), encoding="utf-8").read()
    setup_env = infer.split("def setup_environment", 1)[1].split("def setup_agent", 1)[0]
    assert setup_env.index("apply_document(") < setup_env.index("initialize_environment(")

    train = open(os.path.join(PYTHON_ROOT, "TrainFirefighterOptionCritic.py"), encoding="utf-8").read()
    main = train.split("def main_training", 1)[1].split("def initialize_environment", 1)[0]
    assert main.index("apply_tuning(") < main.index("initialize_environment(")
    assert "QuicFireConstants.DEFAULT_REWARD_ALPHA" in train

    elo = open(os.path.join(PYTHON_ROOT, "AI", "EloCurriculumManager.py"), encoding="utf-8").read()
    curriculum = elo.split("def train_firefighter_agent_with_curriculum", 1)[1]
    assert curriculum.index("apply_tuning(") < curriculum.index("create_training_environment(")

    engine = open(os.path.join(PYTHON_ROOT, "QuicFire", "QuicFireEngine.py"), encoding="utf-8").read()
    assert "plume_rise(" in engine
    assert "plume_tilt_length(" in engine
    assert "plume_k(" in engine
    assert "LAMBDA_EP / QuicFireConstants.LAMBDA_AIR" not in engine.split("def _get_xr_plume_dynamics", 1)[1]


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok {test.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    restore_defaults()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
