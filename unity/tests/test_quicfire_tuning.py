"""Defaults and overrides for the allowlisted QuicFire constants.

Run from the unity/ directory:
    python tests/test_quicfire_tuning.py

The headset engine uses the same Briggs/McCaffrey coefficients as the root.
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
from QuicFire.QuicFireTuning import TUNABLE_KEYS, apply_document, apply_tuning, restore_defaults

CONFIG_PATH = os.path.join(
    PYTHON_ROOT,
    "TrainedModels",
    "firefighter_policy_config_20250804_143517.json",
)

_PACKET_NAMES = (
    "PLUME_HEIGHT_FACTOR",
    "PLUME_POWER",
    "PLUME_K",
    "PLUME_TILT_SCALE",
    "PLUME_TILT_WIND_FLOOR",
)

_PLUME_FIELDS = (
    "origin_position",
    "plume_u",
    "plume_v",
    "plume_w",
    "total_heat",
    "h_sub",
    "ell",
    "k",
)


def _snapshot():
    return {key: getattr(QuicFireConstants, key) for key in TUNABLE_KEYS}


def _training_document():
    """Shipped training JSON when it is on disk. This snapshot gitignores TrainedModels/."""
    if os.path.isfile(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as handle:
            return json.load(handle)
    raw = {
        key: value
        for key, value in vars(QuicFireConstants).items()
        if not key.startswith("__")
    }
    return {"config": {}, "quicfire_constants": json.loads(json.dumps(raw))}


def _active_lines(source):
    return [line for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def test_defaults_match_the_briggs_numbers():
    restore_defaults()
    assert QuicFireConstants.DT == 20.0
    assert QuicFireConstants.H_SF == 5.0
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
    assert QuicFireConstants.PLUME_CONSTANT == 0.377
    assert QuicFireConstants.BRIGGS_RISE_COEFF == 1.6
    assert QuicFireConstants.BRIGGS_WIND_THRESHOLD == 0.5
    assert QuicFireConstants.MORTON_RISE_COEFF == 5.0
    assert QuicFireConstants.MCCAFFREY_VELOCITY_COEFF == 1.9
    assert QuicFireConstants.PLUME_LENGTH_FLOOR == 1.0
    assert QuicFireConstants.PLUME_MIN_RISE_DIAMETERS == 2.0
    assert QuicFireConstants.PLUME_MAX_HEIGHT == 1000.0
    assert QuicFireConstants.PLUME_CALM_TILT_FACTOR == 1.1
    assert QuicFireConstants.PLUME_MAX_TILT_FACTOR == 5.0
    assert QuicFireConstants.PLUME_MAX_VERTICAL_VELOCITY == 50.0
    assert QuicFireConstants.PLUME_K_BASE == 0.5
    assert QuicFireConstants.PLUME_K_FROUDE_WEIGHT == 0.5
    assert QuicFireConstants.PLUME_K_FROUDE_CAP == 2.0
    assert QuicFireConstants.PLUME_K_SHEAR_WEIGHT == 0.5
    assert QuicFireConstants.PLUME_K_SHEAR_WIND == 10.0
    assert QuicFireConstants.PLUME_K_MAX == 2.0
    for name in _PACKET_NAMES:
        assert not hasattr(QuicFireConstants, name)

    heat_flux = 64.0
    length = 8.0
    wind = 4.0
    rise = QuicFireConstants.BRIGGS_RISE_COEFF * (heat_flux ** (1.0 / 3.0)) * (length ** (2.0 / 3.0)) / wind
    assert rise == 1.6 * (heat_flux ** (1.0 / 3.0)) * (length ** (2.0 / 3.0)) / wind
    calm = QuicFireConstants.MORTON_RISE_COEFF * (heat_flux ** (2.0 / 5.0))
    assert calm == 5.0 * (heat_flux ** (2.0 / 5.0))
    velocity = QuicFireConstants.MCCAFFREY_VELOCITY_COEFF * ((heat_flux / QuicFireConstants.PLUME_LENGTH_FLOOR) ** (1.0 / 3.0))
    assert velocity == 1.9 * (heat_flux ** (1.0 / 3.0))


def test_partial_override_then_restore():
    restore_defaults()
    apply_tuning({
        "DT": 10,
        "H_SF": "1.5",
        "BRIGGS_RISE_COEFF": "2",
        "WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD": 0.1,
        "PLUME_K": 9,
        "not_a_constant": 3,
    })
    assert QuicFireConstants.DT == 10.0
    assert isinstance(QuicFireConstants.DT, float)
    assert QuicFireConstants.H_SF == 1.5
    assert QuicFireConstants.BRIGGS_RISE_COEFF == 2.0
    assert QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD == 0.1
    assert QuicFireConstants.PENALTY_TIME_STEP == -0.5
    assert QuicFireConstants.MCCAFFREY_VELOCITY_COEFF == 1.9

    apply_tuning({"DEFAULT_REWARD_ALPHA": 0.5})
    assert QuicFireConstants.DEFAULT_REWARD_ALPHA == 0.5
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.5
    assert QuicFireConstants.DT == 10.0

    apply_tuning({"DT": 0, "BRIGGS_RISE_COEFF": -1, "PLUME_K_SHEAR_WIND": 0, "PLUME_LENGTH_FLOOR": -2})
    assert QuicFireConstants.DT == 10.0
    assert QuicFireConstants.BRIGGS_RISE_COEFF == 2.0
    assert QuicFireConstants.PLUME_K_SHEAR_WIND == 10.0
    assert QuicFireConstants.PLUME_LENGTH_FLOOR == 1.0

    restore_defaults()
    assert QuicFireConstants.DT == 20.0
    assert QuicFireConstants.H_SF == 5.0
    assert QuicFireConstants.BRIGGS_RISE_COEFF == 1.6
    assert QuicFireConstants.DEFAULT_REWARD_ALPHA == 0.8
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.8
    assert QuicFireConstants.WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD == 0.05


def test_shipped_config_is_a_noop_and_tuning_overrides_it():
    restore_defaults()
    document = _training_document()
    before = _snapshot()
    apply_document(document)
    after = _snapshot()
    changed = {key: (before[key], after[key]) for key in before if before[key] != after[key]}
    assert changed == {}
    assert QuicFireConstants.BASE_CONFIG["reward_alpha"] == 0.8

    apply_document({
        "quicfire_constants": {"DT": 20, "H_SF": 5.0, "SURFACE_HEIGHTS": {"200": 150}, "PLUME_K": 4},
        "tuning": {"DT": 15, "MCCAFFREY_VELOCITY_COEFF": 2.2, "PLUME_HEIGHT_FACTOR": 0.1},
    })
    assert QuicFireConstants.DT == 15.0
    assert QuicFireConstants.H_SF == 5.0
    assert QuicFireConstants.MCCAFFREY_VELOCITY_COEFF == 2.2
    assert QuicFireConstants.BRIGGS_RISE_COEFF == 1.6
    assert QuicFireConstants.PENALTY_TIME_STEP == -0.5
    restore_defaults()


def test_entry_points_apply_tuning_before_dividing_by_dt():
    infer = open(os.path.join(PYTHON_ROOT, "InferFirefighterOptionCritic.py"), encoding="utf-8").read()
    setup_env = infer.split("def setup_environment", 1)[1].split("def setup_agent", 1)[0]
    assert setup_env.index("apply_document(") < setup_env.index("initialize_environment(")
    assert "enable_viz_recorder=False" in infer
    assert "def get_status" in infer

    train = open(os.path.join(PYTHON_ROOT, "TrainFirefighterOptionCritic.py"), encoding="utf-8").read()
    main = train.split("def main_training", 1)[1].split("def initialize_environment", 1)[0]
    assert main.index("apply_tuning(") < main.index("initialize_environment(")
    assert "QuicFireConstants.DEFAULT_REWARD_ALPHA" in train
    assert not any("animate_replay" in line or "RewardVisualizer" in line for line in _active_lines(train))
    assert not any(line.startswith("import matplotlib") or line.startswith("from matplotlib") for line in train.splitlines())

    elo = open(os.path.join(PYTHON_ROOT, "AI", "EloCurriculumManager.py"), encoding="utf-8").read()
    curriculum = elo.split("def train_firefighter_agent_with_curriculum", 1)[1]
    assert curriculum.index("apply_tuning(") < curriculum.index("create_training_environment(")

    engine = open(os.path.join(PYTHON_ROOT, "QuicFire", "QuicFireEngine.py"), encoding="utf-8").read()
    assert not any(line.startswith("import matplotlib") or line.startswith("from matplotlib") for line in engine.splitlines())
    init = engine.split("def __init__", 1)[1].split("def _setup_rng_generators", 1)[0]
    assert init.index("apply_tuning(") < init.index("_validate_config_file(")
    plume = engine.split("def _get_xr_plume_dynamics", 1)[1].split("def get_xr_visualization_data", 1)[0]
    assert "Briggs" in plume
    assert "McCaffrey" in plume
    assert "BRIGGS_RISE_COEFF" in plume
    assert "MCCAFFREY_VELOCITY_COEFF" in plume
    assert "plume_rise(" not in engine
    assert "plume_k(" not in engine
    assert "plume_tilt_length(" not in engine
    for field in _PLUME_FIELDS:
        assert field in plume

    simulation = open(os.path.join(PYTHON_ROOT, "QuicFire", "QuicFireSimulation.py"), encoding="utf-8").read()
    assert not any("QuicFireVisualizer" in line or "matplotlib" in line for line in _active_lines(simulation))


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
