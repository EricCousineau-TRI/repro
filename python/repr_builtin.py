class Thing:
    def __init__(self, x):
        self.x = x

    @staticmethod
    def Zero():
        return Thing(0)

    def __repr__(self):
        if self.x == 0:
            return "Thing.Zero()"
        elif self.x == 1:
            return "Thing(1)"
        else:
            return object.__repr__(self)


def main():
    print(Thing.Zero())
    print(Thing(1))
    print(Thing(2))


if __name__ == "__main__":
    main()
