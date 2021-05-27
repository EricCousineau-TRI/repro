import ctypes
import os

# https://stackoverflow.com/a/9410056/7829525
_libc = ctypes.cdll.LoadLibrary("libc.so.6")


def gettid():
    # /usr/include/x86_64-linux-gnu/asm/unistd_64.h
    __NR_gettid = 186
    return _libc.syscall(__NR_gettid)


# @dc.dataclass  # Needs venv for Python 3.6 :(
class SchedParam:
    def __init__(self, sched_priority, sched_affinity_cpus):
        self.sched_priority = sched_priority
        self.sched_affinity_cpus = sched_affinity_cpus

    def __repr__(self):
        return (
            f"SchedParam(sched_priority={self.sched_priority}, "
            f"sched_affinity_cpus={self.sched_affinity_cpus})"
        )


def set_sched_param(pid, param):
    if param.sched_priority != -1:
        os.sched_setscheduler(
            pid, os.SCHED_RR, os.sched_param(param.sched_priority)
        )
    if len(param.sched_affinity_cpus) > 0:
        os.sched_setaffinity(pid, param.sched_affinity_cpus)


def get_sched_param_from_pid(pid):
    scheduler = os.sched_getscheduler(pid)
    if scheduler == os.SCHED_RR:
        priority = os.sched_getparam(pid).sched_priority
    else:
        assert scheduler == os.SCHED_OTHER
        priority = -1
    cpus = list(os.sched_getaffinity(pid))
    cpus_all = list(range(os.cpu_count()))
    if cpus == cpus_all:
        cpus = []
    return SchedParam(priority, cpus)


def main():
    pid = 0  # gettid()
    init_param = get_sched_param_from_pid(pid)
    print(f"get_sched_param_from_pid({pid}) = {init_param}")

    user_param = SchedParam(20, [2])
    print(f"set_sched_param({pid}, {user_param})")
    set_sched_param(pid, user_param)

    post_param = get_sched_param_from_pid(pid)
    print(f"get_sched_param_from_pid({pid}, {post_param})")
    print()


if __name__ == "__main__":
    main()
