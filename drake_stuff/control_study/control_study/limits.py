import dataclasses as dc

import numpy as np


def _as_lines(xs):
    return "\n" + "\n".join(str(x) for x in xs)


@dc.dataclass
class VectorLimits:
    lower: np.ndarray
    upper: np.ndarray

    def __iter__(self):
        as_tuple = (self.lower, self.upper)
        return iter(as_tuple)

    def is_valid(self):
        return (self.lower <= self.upper).all()

    def isfinite(self):
        return np.isfinite(self.lower).all() and np.isfinite(self.upper).all()

    def any_finite(self):
        return np.isfinite(self.lower).any() or np.isfinite(self.upper).any()

    def assert_value_with_limits(self, value, *, name, tol=0.0):
        too_low = value < self.lower - tol
        too_high = value > self.upper + tol
        bad = too_low | too_high
        if bad.any():
            lines = (
                f"Limit violation for '{name}'",
                bad,
                value,
                self.lower,
                self.upper,
            )
            raise RuntimeError(_as_lines(lines))

    def select(self, mask):
        return VectorLimits(lower=self.lower[mask], upper=self.upper[mask])

    def saturate(self, x):
        return np.clip(x, self.lower, self.upper)

    def make_valid(self):
        return VectorLimits(
            lower=np.minimum(self.lower, self.upper),
            upper=np.maximum(self.lower, self.upper),
        )

    def intersection(self, other):
        return VectorLimits(
            lower=np.maximum(self.lower, other.lower),
            upper=np.minimum(self.upper, other.upper),
        )

    def scaled(self, scale):
        if not np.isfinite(scale):
            ones = np.ones_like(self.upper)
            return VectorLimits(-np.inf * ones, np.inf * ones)
        center = (self.upper + self.lower) / 2
        width = self.upper - self.lower
        new_width = width * scale
        return VectorLimits(
            lower=center - new_width / 2,
            upper=center + new_width / 2,
        )


@dc.dataclass
class PlantLimits:
    q: VectorLimits
    v: VectorLimits
    u: VectorLimits
    vd: VectorLimits

    @staticmethod
    def from_plant(plant):
        return PlantLimits(
            q=VectorLimits(
                lower=plant.GetPositionLowerLimits(),
                upper=plant.GetPositionUpperLimits(),
            ),
            v=VectorLimits(
                lower=plant.GetVelocityLowerLimits(),
                upper=plant.GetVelocityUpperLimits(),
            ),
            u=VectorLimits(
                lower=plant.GetEffortLowerLimits(),
                upper=plant.GetEffortUpperLimits(),
            ),
            vd=VectorLimits(
                lower=plant.GetAccelerationLowerLimits(),
                upper=plant.GetAccelerationUpperLimits(),
            ),
        )

    def assert_values_within_limits(
        self, *, q=None, v=None, u=None, vd=None, tol=0.0
    ):
        assert (
            q is not None or v is not None or u is not None or vd is not None
        )
        if q is not None:
            self.q.assert_value_with_limits(q, name="q", tol=tol)
        if v is not None:
            self.v.assert_value_with_limits(v, name="v", tol=tol)
        if u is not None:
            self.u.assert_value_with_limits(u, name="u", tol=tol)
        if vd is not None:
            self.vd.assert_value_with_limits(vd, name="vd", tol=tol)
