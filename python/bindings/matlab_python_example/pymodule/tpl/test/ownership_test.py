#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl import ownership

factory_a = lambda: ownership.create_instance()  # ownership.A(2)

ac = ownership.check_creation(factory_a, False)
print(ac.value())
# ac2 = ownership.check_creation(ownership.create_instance(), True)
# print(ac2.value())
