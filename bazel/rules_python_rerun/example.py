import os
import re
import sys


def reformat_path(text):
    pattern = r".*\.runfiles"
    text = re.sub(pattern, "{runfiles}", text)
    text = text.replace(os.getcwd(), "${PWD}")
    return text


def print_paths():
    for p in sys.path:
        print(reformat_path(p))


def main():
    print_paths()

    import rerun as rr
    rr.init("test", spawn=True)
    rr.spawn()
    rr.set_time_sequence("frame", 42)


if __name__ == "__main__":
    main()
