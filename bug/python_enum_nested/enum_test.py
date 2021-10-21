import dataclasses as dc
from enum import Enum
from io import BytesIO
import pickle
import unittest


class Works1:
    @dc.dataclass(frozen=True)
    class NestedValue1:
        value: int

    a = NestedValue1(1)


@dc.dataclass(frozen=True)
class TopValue:
    value: int


class Works2(Enum):
    # N.B. Causes recursion error w/ dill==0.3.4
    a = TopValue(1)


class DoesNotWork(Enum):
    @dc.dataclass(frozen=True)
    class NestedValue2:
        value: int

    a = NestedValue2(1)


def pickle_round_trip(obj):
    f = BytesIO()
    pickle.dump(obj, f)
    f.seek(0)
    obj_again = pickle.load(f)
    return obj_again


class Test(unittest.TestCase):
    def check(self, cls):
        print(cls)
        obj = cls.a
        obj_again = pickle_round_trip(obj)
        self.assertEqual(obj, obj_again)

    def test_issue7862(self):
        self.check(Works1)
        self.check(Works2)
        self.check(DoesNotWork)


if __name__ == "__main__":
    unittest.main()
