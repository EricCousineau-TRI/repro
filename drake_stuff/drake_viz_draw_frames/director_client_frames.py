import lcm
from lcm import LCM

from drake import lcmt_viewer_draw
from pydrake.math import RigidTransform


def draw_frames(names, X_WF_list, *, base=None, suffix=None, lcm=None):
    if base is None:
        base = "DRAKE_DRAW_FRAMES"
    if suffix is None:
        suffix = "_PY"
    if lcm is None:
        lcm = LCM()
    msg = lcmt_viewer_draw()
    msg.num_links = len(names)
    msg.link_name = names
    msg.robot_num = [0] * len(names)
    for X_WF in X_WF_list:
        X_WF = RigidTransform(X_WF)
        msg.position.append(X_WF.translation())
        msg.quaternion.append(X_WF.rotation().ToQuaternion().wxyz())
    lcm.publish(base + suffix, msg.encode())


def draw_frames_dict(d, *, base=None, suffix=None, lcm=None):
    names, X_WF_list = zip(*d.items())
    draw_frames(names, X_WF_list, base=base, suffix=suffix, lcm=lcm)


def draw_frames_args(*, base=None, suffix=None, lcm=None, **kwargs):
    draw_frames_dict(kwargs, base=base, suffix=suffix, lcm=lcm)
