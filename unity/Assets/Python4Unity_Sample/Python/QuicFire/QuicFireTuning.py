"""
Override a small allowlist of QuicFireConstants without touching the rest.

Defaults are snapshotted at import. apply_tuning changes only the keys it is
given, so a partial update leaves every other constant alone. restore_defaults
puts the snapshot back, including BASE_CONFIG['reward_alpha'].

Call apply_tuning or apply_document before an environment is constructed.
max_steps is total_time / DT in the training and inference entry points, and
that division happens before FirefighterEnvironment.__init__.

H_SF is one constant. It is both the surface-fuel height used by ignition and
the offset in the headset packet rise h_sub = H_SF + PLUME_HEIGHT_FACTOR * Q^PLUME_POWER.
Tuning it changes both, which matches the single constant the code already had.

Saved training JSON dumps every constant, including dicts such as
SURFACE_HEIGHTS whose keys become strings. apply_document copies only the
allowlist, then applies a "tuning" object on top of that. Unknown keys inside
"tuning" are warned about. Unknown keys in the full dump are ignored.
"""

from typing import Dict, Optional

from QuicFire.QuicFireConstants import QuicFireConstants

TUNABLE_KEYS = (
    "DT",
    "H_SF",
    "PENALTY_ACTIVE_FIRE_OUTSIDE_BB",
    "PENALTY_FUEL_CONSUMED_OUTSIDE_BB",
    "REWARD_FUEL_CONSUMED_INSIDE_BB",
    "PENALTY_GLOBAL_WATER_USAGE",
    "PENALTY_GLOBAL_RETARDANT_USAGE",
    "PENALTY_GLOBAL_AIR_SUPPORT_USAGE",
    "PENALTY_TIME_STEP",
    "REWARD_MISSION_CONTAINED_SUCCESS",
    "PENALTY_MISSION_FAILURE_TIMEOUT",
    "PENALTY_MISSION_FAILURE_SPREAD",
    "REWARD_AGENT_FIRE_REDUCTION",
    "PENALTY_RESOURCE_INVALID_ACTION",
    "MAX_GLOBAL_REWARD_CLIP",
    "MAX_AGENT_REWARD_CLIP",
    "MOVEMENT_PENALTY",
    "DEFAULT_REWARD_ALPHA",
    "WIN_PERCENT_FUEL_BURNED_OUTSIDE_BB_THRESHOLD",
    "PLUME_HEIGHT_FACTOR",
    "PLUME_POWER",
    "PLUME_K",
    "PLUME_TILT_SCALE",
    "PLUME_TILT_WIND_FLOOR",
    "PLUME_CONSTANT",
)

_DEFAULTS = {key: getattr(QuicFireConstants, key) for key in TUNABLE_KEYS}
_DEFAULT_BASE_REWARD_ALPHA = QuicFireConstants.BASE_CONFIG.get("reward_alpha", QuicFireConstants.DEFAULT_REWARD_ALPHA)


def restore_defaults() -> None:
    """Put every allowlisted constant back to the value captured at import."""
    for key, value in _DEFAULTS.items():
        setattr(QuicFireConstants, key, value)
    QuicFireConstants.BASE_CONFIG["reward_alpha"] = _DEFAULT_BASE_REWARD_ALPHA


def apply_tuning(values: Optional[Dict], unknown: str = "warn") -> None:
    """
    Set allowlisted constants from a dict. Other keys are left as they are.

    unknown="warn" prints names that are not on the allowlist.
    unknown="ignore" skips them silently, which is what the full constants
    dump needs.
    """
    if not values:
        return
    if not isinstance(values, dict):
        print("QuicFireTuning: tuning value is not a dict; nothing applied")
        return

    for key, value in values.items():
        if key not in _DEFAULTS:
            if unknown == "warn":
                print(f"QuicFireTuning: ignoring unknown key {key}")
            continue
        default = _DEFAULTS[key]
        try:
            coerced = _coerce(value, default)
        except (TypeError, ValueError) as exc:
            print(f"QuicFireTuning: skipping {key}: {exc}")
            continue
        if key == "DT" and not coerced > 0:
            print("QuicFireTuning: DT must be positive; keeping the current value")
            continue
        if key == "PLUME_TILT_WIND_FLOOR" and not coerced > 0:
            print("QuicFireTuning: PLUME_TILT_WIND_FLOOR must be positive; keeping the current value")
            continue
        setattr(QuicFireConstants, key, coerced)
        if key == "DEFAULT_REWARD_ALPHA":
            QuicFireConstants.BASE_CONFIG["reward_alpha"] = coerced


def apply_document(document: Optional[Dict]) -> None:
    """
    Apply allowlisted quicfire_constants from a training JSON document, then
    let a top-level "tuning" object override those values.
    """
    if not isinstance(document, dict):
        return
    constants = document.get("quicfire_constants")
    if isinstance(constants, dict):
        picked = {key: constants[key] for key in TUNABLE_KEYS if key in constants}
        apply_tuning(picked, unknown="ignore")
    tuning = document.get("tuning")
    if isinstance(tuning, dict):
        apply_tuning(tuning, unknown="warn")


def plume_rise(total_heat: float) -> float:
    """h_sub = H_SF + PLUME_HEIGHT_FACTOR * Q ** PLUME_POWER."""
    heat = float(total_heat)
    return (
        QuicFireConstants.H_SF
        + QuicFireConstants.PLUME_HEIGHT_FACTOR
        * (heat ** QuicFireConstants.PLUME_POWER)
    )


def plume_k() -> float:
    """Packet turbulence factor. Default 4, from LAMBDA_EP / LAMBDA_AIR."""
    return QuicFireConstants.PLUME_K


def plume_tilt_length(h_sub: float, width: float, wind_speed_horizontal: float) -> float:
    """
    ell = PLUME_TILT_SCALE * h_sub * (1 + sqrt(1 + width^2 / max(wind^2, floor))).
    The floor replaces the previous literal 1e-6.
    """
    floor = QuicFireConstants.PLUME_TILT_WIND_FLOOR
    wind2 = float(wind_speed_horizontal) * float(wind_speed_horizontal)
    if wind2 < floor:
        wind2 = floor
    ell = float(h_sub) * (1.0 + (1.0 + (float(width) * float(width)) / wind2) ** 0.5)
    return ell * QuicFireConstants.PLUME_TILT_SCALE


def _coerce(value, default):
    if isinstance(value, (dict, list)):
        raise TypeError("expected a scalar")
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return type(default)(value)
