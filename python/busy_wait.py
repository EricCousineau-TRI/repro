import argparse
import signal
from threading import Thread
import time

import numpy as np

DONE = False
READY = False


def on_sigint(sig, stack):
    global DONE
    DONE = True


def busy_wait():
    print("Start")
    while not READY:
        time.sleep(0.01)

    x = np.zeros((100, 100))
    while not DONE:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] += i + j

    print("Finish")


def main():
    global READY

    parser = argparse.ArgumentParser()
    parser.add_argument("num_threads", type=int)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, on_sigint)

    num_threads = args.num_threads
    threads = [
        Thread(target=busy_wait)
        for _  in range(num_threads)
    ]
    for thread in threads:
        thread.start()

    # Start busy waiting. Let sigint stop.
    READY = True

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
