from functools import lru_cache, partial


@lru_cache
def print_on_miss(x):
    print(f"Cache miss: {x}")


def func(x):
    pass


def main():
    print_on_miss(2)
    print_on_miss(2)
    
    print_on_miss(lambda x: x)
    print_on_miss(lambda x: x)
    
    print_on_miss(partial(func, 2))
    print_on_miss(partial(func, 2))
    
    x = object()
    print_on_miss(x)
    print_on_miss(x)


if __name__ == "__main__":
    main()
