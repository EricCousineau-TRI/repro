import sys


def main():
    assert sys.stdin.isatty()
    input("Press enter...")
    print("Done!")


assert __name__ == "__main__"
main()
