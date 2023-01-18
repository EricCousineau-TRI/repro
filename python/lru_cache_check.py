from functools import lru_cache


@lru_cache
def cached_1_arg(x):
    print("  called cached_1_arg")
    return 2 * x


@lru_cache
def cached_0_arg():
    print("  called cached_0_arg")
    return 10


def main():
    for i in range(3):
        print(f"i={i}")
        assert cached_1_arg(10) == 20
        assert cached_0_arg() == 10


assert __name__ == "__main__"
main()

"""
Output (CPython 3.10.6)

i=0
  called cached_1_arg
  called cached_0_arg
i=1
i=2
"""
