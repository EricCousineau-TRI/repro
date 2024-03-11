from top import sub_1, sub_2


def main():
    print(sub_1.__file__)
    print(sub_1.my_func())
    print(sub_2.__file__)
    print(sub_2.my_func())


if __name__ == "__main__":
    main()
