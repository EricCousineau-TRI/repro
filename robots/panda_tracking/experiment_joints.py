import datetime
from os.path import expanduser
import pickle
import time

from lcm import LCM
import numpy as np

from drake import lcmt_panda_command, lcmt_panda_status

from <module>.lcm_util import (
    LcmMessagePoll,
    lcm_handle_all,
)

robot_lcm_url = "<url>?ttl=0"
cmd_channel = "PANDA_COMMAND"
status_channel = "PANDA_STATUS"
ndof = 7


def make_command(vd):
    cmd = lcmt_panda_command()
    cmd.utime = int(1e6 * time.time())
    cmd.control_mode_expected = lcmt_panda_status.CONTROL_MODE_VELOCITY
    cmd.num_joints = ndof
    cmd.joint_position = np.full(ndof, np.nan).tolist()
    cmd.joint_velocity = vd.tolist()
    cmd.joint_torque = np.full(ndof, np.nan).tolist()
    return cmd


def get_timestamp():
    return (
        datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds")
        .replace(":", "_")
    )


def main():
    lcm = LCM(robot_lcm_url)
    status_sub = LcmMessagePoll(lcm, status_channel, lcmt_panda_status)
    events = []

    print("Waiting for first message...")
    lcm.handle()

    count = 0
    dt_des = 0.005

    print("Received! Starting")
    t_abs_start = time.time()
    t = 0.0
    t_transient_start = 3.0
    t_transient_end = 0.25

    v = None

    try:
        while True:
            lcm_handle_all(lcm)
            status = status_sub.get()
            if status is None:
                continue

            t_prev = t
            t = time.time() - t_abs_start
            s = min(t / t_transient_start, 1.0)
            dt = t - t_prev

            v = np.array(status.joint_velocity)
            assert v.shape == (ndof,), v.shape

            T_sec = np.array([4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]) / 4
            w = 2 * np.pi / T_sec
            v_max = np.array([0.5, 1.5, 0.5, 1.0, 1.0, 1.5, 1.5])

            vd = np.zeros(ndof)
            vd[:] = s * v_max * np.sin(w * t)

            cmd = make_command(vd)
            lcm.publish(cmd_channel, cmd.encode())
            events.append((status, cmd))

            count += 1

            t_next = t + dt_des
            while time.time() - t_abs_start < t_next:
                time.sleep(dt_des / 100)

    except KeyboardInterrupt:
        # Wind down.
        assert v is not None
        v0 = v
        vd = np.zeros(ndof)

        t_abs_start = time.time()
        t = 0.0
        while t < t_transient_end:
            lcm_handle_all(lcm)
            status = status_sub.get()
            if status is None:
                continue

            t = time.time() - t_abs_start
            s = min(t / t_transient_end, 1.0)
            vd = v0 * (1 - s)
            cmd = make_command(vd)
            lcm.publish(cmd_channel, cmd.encode())
            events.append((status, cmd))

            t_next = t + dt_des
            while time.time() - t_abs_start < t_next:
                time.sleep(dt_des / 100)

    finally:
        # Save data.
        base = get_timestamp()
        file = expanduser(f"~/data/panda/tracking/{base}.pkl")
        with open(file, "wb") as f:
            pickle.dump(events, f)
        print(f"Wrote: {file}")


assert __name__ == "__main__"
main()
print("[ Done ]")
