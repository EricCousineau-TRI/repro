import dataclasses as dc

@dc.dataclass
class Node:
    other: "Node"


def main():
    field, = dc.fields(Node)
    assert field.type == "Node"
    field.type = Node
    field, = dc.fields(Node)
    assert field.type is Node
    print("Good!")


if __name__ == "__main__":
    main()
