import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_bool", type=bool, default=True)
    args = parser.parse_args(["--my_bool=false"])
    print(args)


if __name__ == "__main__":
    main()
