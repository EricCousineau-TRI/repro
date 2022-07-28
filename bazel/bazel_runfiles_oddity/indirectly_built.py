from runfiles import Rlocation


def main():
    file = Rlocation("bazel_runfiles_oddity/some_data.txt")
    with open(file, "r") as f:
        text = f.read()
    print(f"Read: {text.strip()}")


if __name__ == "__main__":
    main()
