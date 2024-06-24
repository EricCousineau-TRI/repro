import argparse
import signal
from threading import Thread
import time

DONE = False
READY = False


def on_sigint(sig, stack):
    global DONE
    DONE = True


def busy_wait():
    print("Start")
    while not READY:
        time.sleep(0.01)
    count = 0
    while not DONE:
        count += 1
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
