import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Integrator,
    LeafSystem,
    Quaternion,
    RotationMatrix,
    Simulator,
)


class SimpleVector(LeafSystem):
    def __init__(self, num_q, calc_qd, *, pass_time=False):
        super().__init__()

        self.q = self.DeclareVectorInputPort("q", num_q)

        def calc_qd_sys(context, output):
            q = self.q.Eval(context)
            if pass_time:
                t = context.get_time()
                qd = calc_qd(t, q)
            else:
                qd = calc_qd(q)
            output.set_value(qd)

        self.qd = self.DeclareVectorOutputPort("qd", num_q, calc_qd_sys)


def integrate(q0, calc_qd, tf, *, check=False, pass_time=False, accuracy=1e-5):
    num_q = len(q0)

    builder = DiagramBuilder()
    integ = builder.AddSystem(Integrator(num_q))
    q_actual_port = integ.get_output_port()
    controller = builder.AddSystem(
        SimpleVector(num_q, calc_qd, pass_time=pass_time)
    )
    builder.Connect(q_actual_port, controller.q)
    builder.Connect(controller.qd, integ.get_input_port())

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    integ_context = integ.GetMyContextFromRoot(diagram_context)
    controller_context = controller.GetMyContextFromRoot(diagram_context)

    integ.set_integral_value(integ_context, q0)

    ts = []
    q_as = []

    def monitor(diagram_context_in):
        assert diagram_context_in is diagram_context
        t = diagram_context.get_time()
        q_actual = q_actual_port.Eval(integ_context)
        if check:
            # Compute derivatives passing `check=True` which signifies we have
            # a committed solution.
            if pass_time:
                calc_qd(t, q_actual, check=True)
            else:
                calc_qd(q_actual, check=True)
        ts.append(t)
        q_as.append(q_actual)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_monitor(monitor)
    integrator = simulator.get_mutable_integrator()
    integrator.set_target_accuracy(accuracy)
    try:
        simulator.AdvanceTo(tf)
    except Exception:
        t = diagram_context.get_time()
        print(f"Error at t={t}")
        raise
    return ts, q_as


def maxabs(x):
    return np.max(np.abs(x))


def cat(*args):
    return np.concatenate(args)


def flatten(R):
    return R.flat[:]


def unflatten(q):
    n = int(np.sqrt(len(q)))
    assert len(q) == n * n
    return q.reshape((n, n))


def split(q, lens):
    qs = []
    i = 0
    for len_i in lens:
        q_i = q[i:i + len_i]
        qs.append(q_i)
        i += len_i
    assert i == len(q)
    return qs


def normalize(x, *, tol=1e-10):
    x = np.asarray(x)
    n = np.linalg.norm(x)
    assert n >= tol
    return x / n


def rot2d(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.array([
        [c, -s],
        [s, c],
    ])


def rot2d_jac(th):
    s = np.sin(th)
    c = np.cos(th)
    return np.array([
        [-s, -c],
        [c, -s],
    ])


def so2_angle(R):
    x = R[0, 0]
    y = R[1, 0]
    return np.arctan2(y, x)

def so2_dist(R_a, R_d):
    return so2_angle(R_a.T @ R_d)


def skew(r):
    r1, r2, r3 = r
    return np.array(
        [
            [0, -r3, r2],
            [r3, 0, -r1],
            [-r2, r1, 0],
        ]
    )


def unskew(S, *, tol=1e-10):
    if tol is not None:
        dS = S + S.T
        assert np.all(maxabs(dS) < tol), (dS)
    r1 = S[2, 1]
    r2 = S[0, 2]
    r3 = S[1, 0]
    return np.array([r1, r2, r3])


def axang(ax, th):
    """Exponential-map thinger, using Eq. (2.14) of [MLS]."""
    # return RotationMatrix(AngleAxis(angle, axis)).matrix()
    c = np.cos(th)
    s = np.sin(th)
    L = skew(ax)
    I = np.eye(3)
    R = I + L * s + L @ L * (1 - c)
    return R


def axang_dot(ax, axd, th, thd):
    """Time derivative of above."""
    c = np.cos(th)
    s = np.sin(th)
    L = skew(ax)
    Ld = skew(axd)
    Rd = Ld * s + L * c * thd + (L @ Ld + Ld @ L) * (1 - c) + (L @ L) * s * thd
    return Rd


def axang3(r, *, tol=1e-10):
    """Expononential map against R^3, splitting into (θ, λ)"""
    r = np.asarray(r)
    th = np.linalg.norm(r)
    if th < tol:
        ax = np.array([0, 0, 1])
    else:
        ax = r / th
    return axang(ax, th)


def axang3_dot(r, rd, *, tol=1e-10):
    """Time derivative of the above."""
    th = np.linalg.norm(r)
    if th < tol:
        ax = np.array([0, 0, 1])
        thd = np.linalg.norm(rd)
        # Err, dunno.
        if thd < tol:
            axd = np.zeros(3)
        else:
            axd = rd / thd
    else:
        ax = r / th
        thd = ax.dot(rd)
        axd = (rd * th - r * thd) / (th * th)
    Rd = axang_dot(ax, axd, th, thd)
    return Rd


def so3_angle(R):
    # same as AngleAxis(R).angle()
    inner = (np.trace(R) - 1) / 2
    if inner > 1:
        # Reflect crossing at 1... I guess?
        inner = 2 - inner
    th = np.arccos(inner)
    assert np.isfinite(th)
    return th


def to_axang(R, *, tol=1e-10):
    """Log-map from Eq. (2.17) and (2.18) of [MLS]."""
    # axang = AngleAxis(R)
    # ax, th = axang.axis(), axang.angle()
    th = so3_angle(R)
    if np.abs(th) < tol:
        ax = np.array([0, 0, 1])
    else:
        s = np.sin(th)
        ax = 1 / (2 * s) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ])
    return ax, th


def to_axang3(R, *, tol=1e-10):
    """Log-map thinger."""
    ax, th = to_axang(R, tol=tol)
    return ax * th


def so3_dist(R_a, R_b):
    return so3_angle(R_a.T @ R_b)


def assert_so3(R, *, tol):
    err = R @ R.T - np.eye(3)
    assert maxabs(err) < tol, err


def rot_to_quat(R):
    quat = RotationMatrix(R).ToQuaternion().wxyz()
    return quat

def quat_to_rot(quat):
    return RotationMatrix(Quaternion(wxyz=quat)).matrix()


def angular_to_quat_dot(q):
    qw, qx, qy, qz = q
    Ninv = 0.5 * np.array([
        [-qx, -qy, -qz],
        [qw, qz, -qy],
        [-qz, qw, qx],
        [qy, -qx, qw]])
    return Ninv


def quat_dot_to_angular(q):
    Ninv = angular_to_quat_dot(q)
    N = 4 * Ninv.T
    return N
