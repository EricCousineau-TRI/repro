import functools


def show_bad_closure():
    funcs = []
    for x in range(3):
        # BAD: This is a closure of x as non-local variable. Thus, when called
        # later it will use the latest value of x, thus all prints will show the same value.
        func = lambda: print(x)
        funcs.append(func)
    for func in funcs:
        func()


def show_good_closure():
    funcs = []
    for x in range(3):
        # GOOD: We're using `functools.partial` to take the current value of x.
        func = functools.partial(print, x)
        funcs.append(func)
    for func in funcs:
        func()


def main():
    print("[ bad ]")
    show_bad_closure()
    print()
    print("[ good ]")
    show_good_closure()


if __name__ == "__main__":
    main()


"""
Output:

[ bad ]
2
2
2

[ good ]
0
1
2
"""
