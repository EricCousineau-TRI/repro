import dataclasses as dc
import pickle
from pathlib import Path

import numpy as np


@dc.dataclass
class SubDataclass:
    value: int


class SubObject:
    def __init__(self, value):
        self.value = value


class SubSpecialObject:
    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        return (self.value, "special")

    def __setstate__(self, state):
        value, marker = state
        assert marker == "special"
        self.value


@dc.dataclass
class MyObject:
    value: str
    sub_dataclass: SubDataclass
    sub_object: SubObject
    sub_special: SubSpecialObject


def main():
    parent_dir = Path(__file__).parents[1]
    obj = MyObject(
        value="abc",
        sub_dataclass=SubDataclass(10),
        sub_object=SubObject(np.array([1.0, 2.0])),
        sub_special=SubSpecialObject(30),
    )
    with open(parent_dir / "old_data.pkl", "wb") as f:
        pickle.dump(obj, f)


assert __name__ == "__main__"
main()
