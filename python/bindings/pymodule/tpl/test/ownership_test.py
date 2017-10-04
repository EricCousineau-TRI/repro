#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl import ownership

factory_a = lambda: ownership.A(2)
factory_b = lambda: ownership.B(3)

bc = ownership.check_creation_b(factory_b, False)
print(bc.value())
bc2 = ownership.check_creation_b(factory_b, True)
print(bc2.value())

# This causes memory leaks, as expected.
ac2 = ownership.check_creation_a(factory_a, True)
print(ac2.value())
ac = ownership.check_creation_a(factory_a, False)
print(ac.value())
