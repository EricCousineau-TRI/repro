# import trace
import _test_alias_wrapper as m

class ChildUnique(m.BaseUnique):
    def __init__(self, *args):
        m.BaseUnique.__init__(self, *args)

    def value(self):
        # Use a different value, so that we can detect slicing.
        return 10 * m.BaseUnique.value(self)

obj = ChildUnique()
