from threading import Thread
import time
import sys


def run():
    print("Start; sleep for 1 sec")
    time.sleep(1.0)
    print("Stop")


def main():
    thread = Thread(target=run)
    thread.start()
    print("Exit Main")
    sys.exit(0)


assert __name__ == "__main__"
main()

"""
Output (CPython 3.10.6)

Start; sleep for 1 sec
Exit Main
Stop

"""
