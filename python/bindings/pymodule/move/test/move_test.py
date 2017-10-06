#!/usr/bin/env python
from __future__ import print_function, absolute_import

from pymodule.tpl import py_move
from pymodule.tpl import move

def create_obj():
    return py_move.move(move.Test(10))

obj = move.check_creation(create_obj)
print(obj.value())
