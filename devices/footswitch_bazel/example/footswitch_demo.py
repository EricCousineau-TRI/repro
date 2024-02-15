import time

from footswitch_bazel.example.footswitch import FootSwitch


def main():
    footswitch = FootSwitch()

    while True:
        events = footswitch.get_events()
        for pedal, value in events.items():
            print(f"{pedal}: {value}")
        print("---")
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
