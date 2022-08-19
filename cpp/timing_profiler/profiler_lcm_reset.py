from lcm import LCM

from timing_profiler import switch_t


def main():
    lcm = LCM("udpm://231.255.66.76:6666?ttl=0")
    lcm.publish("PROFILER_RESET", switch_t().encode())
    print("published")


assert __name__ == "__main__"
main()
