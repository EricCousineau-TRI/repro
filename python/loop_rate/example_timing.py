from enum import Enum
import math
import time


class Clock:
    def now(self):
        raise NotImplementedError()

    def sleep(self, dt):
        raise NotImplementedError()

    def sleep_until(self, t, *, poll=None):
        raise NotImplementedError()


class WallClock(Clock):
    def __init__(self, *, dt_sleep=1e-4):
        self._dt_sleep = dt_sleep
        self._t_start = time.perf_counter()

    def now(self):
        return time.perf_counter() - self._t_start

    def sleep(self, dt):
        time.sleep(dt)

    def sleep_until(self, t, *, poll=None):
        if poll is not None:
            poll()
        while self.now() < t:
            self.sleep(self._dt_sleep)
            if poll is not None:
                poll()


class FakeClock(Clock):
    def __init__(self):
        self._time = 0.0

    def now(self):
        return self._time

    def sleep(self, dt):
        self._time += dt

    def sleep_until(self, t, *, poll=None):
        if t > self.now():
            self._time = t
        if poll is not None:
            poll()


class CatchupMode(Enum):
    # Just free-run until we're on original timing. Can potentially screw with
    # subsequent timings if we have big payload.
    Nothing = 0
    # Stick to grid of timing... but this costs more time, and allows
    # for some skew.
    Grid = 1
    # Grid timing, but strictly slow - will wait the post-sleep time is on the
    # grid, whereas the above will not require strict on-grid timing.
    GridStrict = 2
    # This has better inter-step timing, but allows for "misalignment" with
    # original timings - only a problem if we have slow things where aliasing
    # can hurt us.
    Reset = 3


class LoopRate:
    def __init__(self, dt, *, clock=time, catchup_mode=CatchupMode.Reset):
        self.dt = dt
        self._clock = clock
        self._catchup_mode = catchup_mode
        self.reset()

    def reset(self):
        self.t_next = self._clock.now() + self.dt

    def step(self, *, poll=None):

        self._clock.sleep_until(self.t_next, poll=poll)
        self.t_next += self.dt

        if self._catchup_mode == CatchupMode.Nothing:
            pass
        elif self._catchup_mode == CatchupMode.Grid:
            while self.t_next < self._clock.now():
                self.t_next += self.dt
        elif self._catchup_mode == CatchupMode.GridStrict:
            while self.t_next < self._clock.now():
                self.t_next += self.dt
            budget = self.t_next - self._clock.now()
            ratio = 0.95
            if budget < self.dt * ratio:
                self.t_next += self.dt
        elif self._catchup_mode == CatchupMode.Reset:
            if self.t_next < self._clock.now():
                self.t_next = self._clock.now() + self.dt
        else:
            assert False


def run(catchup_mode):
    dt = 0.1

    # clock = WallClock()
    clock = FakeClock()
    loop_rate = LoopRate(dt, clock=clock, catchup_mode=catchup_mode)

    times = [clock.now()]

    def step():
        t_next = loop_rate.t_next
        loop_rate.step()
        now = clock.now()
        print(f"t = {now:.4g}")

        dt = now - times[-1]
        overrun = now - t_next
        print(f"  dt: {dt:.4g}")
        print(f"  overrun: {overrun:.4g}")

        times.append(now)

    dt_nominal = dt * 0.9
    dt_big = dt * 2.5

    for _ in range(3):
        clock.sleep(dt_nominal)
        step()

    clock.sleep(dt_big)

    for _ in range(4):
        clock.sleep(dt_nominal)
        step()


def main():
    for catchup_mode in CatchupMode:
        print(f"[ {catchup_mode} ]")
        run(catchup_mode)
        print()


if __name__ == "__main__":
    main()
