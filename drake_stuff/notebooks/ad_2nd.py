import numpy as np


class AutoDiff2nd:
    """
    Simple 2nd-order forward autodiff, but based solely on time.
    """

    def __init__(self, x, xd=None, xdd=None):
        if isinstance(x, AutoDiff2nd):
            assert xd is None and xdd is None
            x, xd, xdd = x
        else:
            x = np.asarray(x)
            assert x.dtype == float
        if xd is None:
            xd = np.zeros_like(x)
        if xdd is None:
            xdd = np.zeros_like(x)
        self.x = x
        self.xd = xd
        self.xdd = xdd

    def as_tuple(self):
        return (self.x, self.xd, self.xdd)

    def __iter__(self):
        return iter(self.as_tuple())

    @property
    def shape(self):
        return self.x.shape

    def __len__(self):
        return len(self.x)

    def __add__(self, other):
        a, ad, add = self
        b, bd, bdd = as_ad(other)
        return AD(a + b, ad + bd, add + bdd)

    def _fail_right_op(self):
        # TODO(eric.cousineau): How to fix?
        assert False, "Must left-multiply to avoid ambiguity w/ numpy"

    def __radd__(self, other):
        self._fail_right_op()

    def __mul__(self, other):
        # import pdb; pdb.set_trace()
        a, ad, add = self
        b, bd, bdd = as_ad(other)
        c = a * b
        cd = a * bd + ad * b
        cdd = a * bdd + 2 * (ad * bd) + add * b
        return AD(c, cd, cdd)

    def __rmul__(self, other):
        self._fail_right_op()

    def __repr__(self):
        return f"AD({self.x}, {self.xd}, {self.xdd})"


AD = AutoDiff2nd


def as_ad(x):
    if isinstance(x, AD):
        return x
    else:
        return AD(x)


def chain_value_2nd(f, s):
    assert isinstance(s, AD)
    t, td, tdd = s
    f, fd, fdd = f
    y = f
    yd = fd * td
    ydd = fdd * td * td + fd * tdd
    return AD(y, yd, ydd)


def _clip_derivs(t, low, high):
    # assert isinstance(t, float)
    td = 1.0
    tdd = 0.0
    if t <= low:
        t = low
        td = 0.0
    elif t >= high:
        t = high
        td = 0.0
    return t, td, tdd


def clip_2nd(s, low, high):
    s = as_ad(s)
    f = _clip_derivs(s.x, low, high)
    return chain_value_2nd(f, s)


def _min_jerk_derivs(t):
    # assert isinstance(t, float)
    # TODO(eric.cousineau): Use Drake't polynomial and/or symbolic stuff.
    c = [0, 0, 0, 10, -15, 6]
    p = [1, t, t**2, t**3, t**4, t**5]
    pd = [0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4]
    pdd = [0, 0, 2, 6 * t, 12 * t**2, 20 * t**3]
    f = np.dot(c, p)
    fd = np.dot(c, pd)
    fdd = np.dot(c, pdd)
    return f, fd, fdd


def min_jerk_2nd(s):
    """
    Simple polynomial f(t) for minimum-jerk (zero velocity and acceleration at
    the start and end):

        s=0: f=0, f'=0, f''=0
        s>=1: f=1, f'=0, f''=0
    """
    s = clip_2nd(s, 0.0, 1.0)
    f = _min_jerk_derivs(s.x)
    return chain_value_2nd(f, s)


def _sin_derivs(t):
    s = np.sin(t)
    c = np.cos(t)
    return s, c, -s


def sin_2nd(s):
    s = as_ad(s)
    f = _sin_derivs(s.x)
    return chain_value_2nd(f, s)


def make_min_jerk_sinusoid(*, As, Ts, T0_ratios=None, y0=None):
    As = np.asarray(As)
    Ts = np.asarray(Ts)
    fs = 1 / Ts
    if T0_ratios is None:
        T0_ratios = np.zeros_like(Ts)
    T0_ratios = np.asarray(T0_ratios)
    if y0 is None:
        y0 = np.zeros_like(Ts)
    y0 = np.asarray(y0)

    def func(t):
        t = as_ad(t)
        Bs = min_jerk_2nd(t) * As
        ws = 2 * np.pi / Ts
        x = (t * fs + T0_ratios) * (2 * np.pi)
        y = Bs * sin_2nd(x) + y0
        return y

    return func
