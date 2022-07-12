#!/usr/bin/env python3

from bdb import BdbQuit


def func_1():
    pass


def func_2():
    import pdb; pdb.set_trace()
    pass


def main():
    func_1()
    func_2()


if __name__ == "__main__":
    try:
        main()
    except BdbQuit:
        pass
