"""
Handles acceleration bounding for torque control according to

    Del Prete, Andrea. "Joint Position and Velocity Bounds in
    Discrete-Time Acceleration/Torque Control of Robot Manipulators."
    IEEE Robotics and Automation Letters (January 2018)
    https://ieeexplore.ieee.org/document/8007337/
"""

import copy
from enum import Enum
import functools

import numpy as np

from control_study.limits import PlantLimits, VectorLimits

STRICT = False


def _acceleration_bounds_from_naive_position_limits(q, v, q_limits, dt, dt2):
    # q = q0 + v0*dt + 0.5*dt^2*vd
    sub = q + v * dt
    scale = 2 / dt2
    vd_limits = VectorLimits(
        lower=scale * (q_limits.lower - sub),
        upper=scale * (q_limits.upper - sub),
    )
    if STRICT:
        assert vd_limits.is_valid()
    return vd_limits


def vectorize(**kwargs):
    # Decorator form of `np.vectorize`.
    return functools.partial(np.vectorize, **kwargs)


def _acceleration_bounds_from_position_limits(q, v, q_limits, dt, dt2):
    # Algorithm 1: "accBoundsFromPosLimits".
    num = len(v)
    v2 = v * v
    q_min, q_max = q_limits

    dq_max = q_max - q
    dq_min = q - q_min
    # Avoid divide-by-zero.
    tol = 1e-4
    dq_max[dq_max < tol] = tol
    dq_min[dq_min < tol] = tol

    vd_max_1 = -v / dt
    vd_max_2 = -v2 / (2 * dq_max)
    vd_max_3 = 2 * (q_max - q - dt * v) / dt2
    vd_min_2 = v2 / (2 * dq_min)
    vd_min_3 = 2 * (q_min - q - dt * v) / dt2

    vd_min, vd_max = _acceleration_bounds_from_position_limits_vectorized(
        v, vd_max_1, vd_max_2, vd_max_3, vd_min_2, vd_min_3
    )
    vd_limits = VectorLimits(vd_min, vd_max)
    if STRICT:
        assert vd_limits.is_valid()
    return vd_limits


@vectorize(cache=True)
def _acceleration_bounds_from_position_limits_vectorized(
    v, vd_max_1, vd_max_2, vd_max_3, vd_min_2, vd_min_3
):
    # N.B. We could write this in a vectorized form, but given the nested
    # branching it seemed simpler to use `np.vectorize`.
    if v >= 0:
        vd_min = vd_min_3
        if vd_max_3 > vd_max_1:
            vd_max = vd_max_3
        else:
            vd_max = min(vd_max_1, vd_max_2)
    else:
        vd_max = vd_max_3
        if vd_min_3 < vd_max_1:
            vd_min = vd_min_3
        else:
            vd_min = max(vd_max_1, vd_min_2)
    return vd_min, vd_max


def _acceleration_bounds_from_viability(q, v, q_limits, vd_limits, dt, dt2):
    # Algorithm 2: "accBoundsFromViability".
    q_min, q_max = q_limits
    vd_min_const, vd_max_const = vd_limits
    v2 = v * v
    # Compute for upper limits.
    a = dt2
    b = dt * (2 * v + vd_max_const * dt)
    c = v2 - 2 * vd_max_const * (vd_max_const - q - dt * v)
    vd_1 = -v / dt
    delta = b * b - 4 * a * c
    delta_positive = delta >= 0
    delta[~delta_positive] = 0.0  # Suppress warning message.
    vd_max_from_delta = (-b + np.sqrt(delta)) / (2 * a)
    vd_max_delta_positive = np.maximum(vd_1, vd_max_from_delta)
    vd_max = vd_1.copy()
    vd_max[delta_positive] = vd_max_delta_positive[delta_positive]
    # Compute for lower limits.
    # TODO(eric.cousineau): I naively replaced `-vd_max_const` with
    # `vd_min_const`. I hope this doesn't mess anything up.
    b = 2 * dt * v + vd_min_const * dt2
    c = v2 + 2 * vd_min_const * (q + dt * v - q_min)
    delta = b * b - 4 * a * c
    delta_positive = delta >= 0
    delta[~delta_positive] = 0.0  # Suppress warning message.
    vd_min_from_delta = (-b - np.sqrt(delta)) / (2 * a)
    vd_min_delta_positive = np.minimum(vd_1, vd_min_from_delta)
    vd_min = vd_1.copy()
    vd_min[delta_positive] = vd_min_delta_positive[delta_positive]
    vd_limits_new = VectorLimits(vd_min, vd_max)
    if STRICT:
        assert vd_limits_new.is_valid()
    return vd_limits_new


def _acceleration_bounds_from_velocity_limits(v, v_limits, dt):
    # Algorithm 3, Lines 4 and 5.
    vd_limits = VectorLimits(
        lower=(v_limits.lower - v) / dt,
        upper=(v_limits.upper - v) / dt,
    )
    if STRICT:
        assert vd_limits.is_valid()
    return vd_limits


def _limits_intersection(*limits):
    # Provides an intersection from a list of limits.
    limit = limits[0]
    for next_limit in limits[1:]:
        limit = limit.intersection(next_limit)
    return limit


def zero_invalid_acceleration_limits(vd_limits):
    vd_limits = copy.deepcopy(vd_limits)
    bad = vd_limits.lower > vd_limits.upper
    vd_limits.lower[bad] = 0.0
    vd_limits.upper[bad] = 0.0
    return vd_limits


def _saturated(nominal, limits):
    return VectorLimits(
        lower=np.minimum(limits.lower, nominal.upper),
        upper=np.maximum(limits.upper, nominal.lower),
    )


def max_out_invalid_acceleration_limits(vd_limits_nominal, vd_limits):
    vd_limits = _saturated(vd_limits_nominal, vd_limits)
    bad = vd_limits.lower > vd_limits.upper
    lower_margin = vd_limits.lower - vd_limits_nominal.lower
    upper_margin = vd_limits_nominal.upper - vd_limits.upper
    lower_violates_most = lower_margin > upper_margin
    # Keep lower where it is, and shift upper back to nominal.
    bad_lower = bad & lower_violates_most
    vd_limits.upper[bad_lower] = vd_limits_nominal.upper[bad_lower]
    # Keep upper where it is, and shift lower back to nominal.
    bad_upper = bad & ~lower_violates_most
    vd_limits.lower[bad_upper] = vd_limits_nominal.lower[bad_upper]
    vd_limits = _limits_intersection(vd_limits, vd_limits_nominal)
    assert vd_limits.is_valid()
    return vd_limits


class BoundingMethod(Enum):
    Naive = 0
    Prete2018 = 1
    Margin = 2


class ResolutionMethod(Enum):
    Nothing = 0
    SetInvalidToZero = 1
    SetInvalidToMax = 2


def _bound_via_margin(y_bound, x, x_bound, x_margin, y_scale_min):
    assert y_scale_min <= 0
    # Written in positive form.
    start = x_bound - x_margin
    end = x_bound
    s = 1 - (x - start) / x_margin
    s = np.clip(s, 0, 1)
    # Now stretch scaling.
    scale_span = 1 - y_scale_min
    s = scale_span * s + y_scale_min
    return s * y_bound


def bounds_via_margin_scaling(
    *,
    y_limits,
    y_scale_min,
    x,
    x_limits,
    x_margin,
):
    y_min, y_max = y_limits
    x_min, x_max = x_limits
    vd_min_new = -_bound_via_margin(-y_min, -x, -x_min, x_margin, y_scale_min)
    vd_max_new = _bound_via_margin(y_max, x, x_max, x_margin, y_scale_min)
    return VectorLimits(vd_min_new, vd_max_new)


def compute_acceleration_bounds(
    *,
    q,
    v,
    plant_limits,
    dt,
    dt2=None,
    check=True,
    bounding_method=BoundingMethod.Naive,
    # bounding_method=BoundingMethod.Prete2018,
    resolution_method=ResolutionMethod.Nothing,
    # resolution_method=ResolutionMethod.SetInvalidToMax,
):
    """
    Provides acceleration bounds that are modulated according to Algorithm 3
    described in paper cited above.

    Warning:
        Per Fig. 4(c) of the paper, without jerk limiting, this may cause
        large discontinuities. Should either try to accommodate this, or
        investigate control barrier functions more deeply.
    """
    # TODO(eric.cousineau): I have a sneaking suspicion that if the controller
    # has effective torque bounding that is strictly bounded by nominal
    # acceleration and torque limits, u_limits, then this may fall apart.
    # Should investigate; possibly, can incorporate by projecting torque limits
    # to acceleration limits, either naively or via incorporation into the QP.
    # This may be related to control barrier functions.
    vd_limits_nominal = plant_limits.vd
    if dt2 is None:
        # dt2 = 0.2 * dt * dt
        dt2 = dt * dt

    if bounding_method == BoundingMethod.Naive:
        limits = [vd_limits_nominal]
        if plant_limits.q.any_finite():
            limits.append(
                _acceleration_bounds_from_naive_position_limits(
                    q, v, plant_limits.q, dt, dt2
                )
            )
        if plant_limits.v.any_finite():
            limits.append(
                _acceleration_bounds_from_velocity_limits(
                    v, plant_limits.v, dt
                )
            )
        vd_limits = _limits_intersection(*limits)
    elif bounding_method == BoundingMethod.Prete2018:
        limits = [vd_limits_nominal]
        if plant_limits.q.any_finite():
            limits.append(
                _acceleration_bounds_from_position_limits(
                    q, v, plant_limits.q, dt, dt2
                )
            )
        if plant_limits.v.any_finite():
            limits.append(
                _acceleration_bounds_from_velocity_limits(
                    v, plant_limits.v, dt,
                )
            )
        if plant_limits.q.any_finite() and vd_limits_nominal.any_finite():
            limits.append(
                _acceleration_bounds_from_viability(
                    q, v, plant_limits.q, vd_limits_nominal, dt, dt2
                )
            )
        # N.B. These are provided in order as described in Algorithm 3.
        vd_limits = _limits_intersection(*limits)
    elif bounding_method == BoundingMethod.Margin:
        vd_limits_position = bounds_via_margin_scaling(
            y_limits=vd_limits_nominal,
            # Allow deceleration.
            y_scale_min=-0.25,
            x=q,
            x_limits=plant_limits.q,
            x_margin=0.3,
        )
        vd_limits_velocity = bounds_via_margin_scaling(
            y_limits=vd_limits_nominal,
            # No deceleration needed to keep constant velocity.
            y_scale_min=-0.05,
            x=v,
            x_limits=plant_limits.v,
            x_margin=0.5,
        )
        vd_limits = _limits_intersection(
            vd_limits_position,
            vd_limits_velocity,
            vd_limits_nominal,
        )
    else:
        assert False

    if vd_limits is None or not vd_limits.isfinite():
        return vd_limits

    if resolution_method == ResolutionMethod.SetInvalidToZero:
        vd_limits = zero_invalid_acceleration_limits(vd_limits)
    elif resolution_method == ResolutionMethod.SetInvalidToMax:
        vd_limits = max_out_invalid_acceleration_limits(
            vd_limits_nominal, vd_limits
        )
    else:
        assert resolution_method == ResolutionMethod.Nothing

    if check:
        assert vd_limits.is_valid()
    return vd_limits
