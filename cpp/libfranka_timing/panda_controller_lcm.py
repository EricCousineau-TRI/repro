import time

from lcm import LCM
import numpy as np

from drake import lcmt_panda_command, lcmt_panda_status

RIGHT_LCM_URL = ""

cmd_channel = "PANDA_COMMAND"
status_channel = "PANDA_STATUS"
ndof = 7


class LcmMessagePoll:
    """Subscribes to a message, and returns the latest message (per LCM's queue
    consumption) or None since the last request.

    Not thread safe.
    """
    def __init__(self, lcm, channel, message_type):
        self._value = None
        self._message_type = message_type
        self._sub = lcm.subscribe(channel, self._callback)
        self._sub.set_queue_capacity(1)

    def _callback(self, _, raw):
        self._value = self._message_type.decode(raw)

    def get(self):
        value = self._value
        self._value = None
        return value


def make_low_level_command(qd):
    cmd = lcmt_panda_command()
    cmd.utime = int(1e6 * time.time())
    cmd.control_mode_expected = lcmt_panda_status.CONTROL_MODE_POSITION
    cmd.num_joint_position = ndof
    cmd.joint_position = qd.tolist()
    return cmd


def run():
    right_lcm = LCM(RIGHT_LCM_URL)
    right_status_sub = LcmMessagePoll(
        right_lcm, status_channel, lcmt_panda_status
    )

    print("Waiting for first messages...")
    right_lcm.handle()

    print("Received! Starting")

    dt_des = 0.001
    t_abs_start = time.time()

    right_status = right_status_sub.get()
    q0 = np.array(right_status.joint_position)
    print(q0)
    # sys.exit(1)
    qd = np.zeros(ndof)

    while True:
        while right_lcm.handle_timeout(0) > 0:
            pass
        right_status = right_status_sub.get()
        if right_status is None:
            continue

        t = time.time() - t_abs_start
        dq = np.pi / 8 * (1 - np.cos(np.pi / 2.5 * t))
        qd[:] = q0 + dq

        # print(qd)
        cmd = make_low_level_command(qd)
        right_lcm.publish(cmd_channel, cmd.encode())

        if t >= 5.0:
            break

        t_next = t + dt_des
        while time.time() - t_abs_start < t_next:
            time.sleep(dt_des / 10)


def main():
    run()


assert __name__ == "__main__"
main()
print("[ Done ]")
