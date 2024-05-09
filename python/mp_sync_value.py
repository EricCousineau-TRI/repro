import ctypes
import multiprocessing as mp
import time


class MyProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self.status = mp.Value(ctypes.c_int)
        self.status.value = 0
        self.stop = mp.Value(ctypes.c_bool)
        self.stop.value = False

    def run(self):
        print("Server: Start")
        self.status.value = 1
        while not self.stop.value:
            time.sleep(1e-3)
        print("Server: Done")
        self.status.value = 2


def main():
    mp.set_start_method("spawn")

    proc = MyProcess()
    print(proc.status.value)
    proc.start()

    time.sleep(0.1)
    print(proc.status.value)

    proc.stop.value = True
    time.sleep(0.1)
    print(proc.status.value)
    proc.join()


if __name__ == "__main__":
    main()
