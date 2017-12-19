revive = None


class Test(object):
    def __init__(self, value):
        self.value = value

    def __del__(self):
        global revive
        if self.value > 0:
            print("Revive")
            revive = self
        else:
            print("Destroy")


obj = Test(10)
del obj

assert revive is not None
obj = revive
revive = None

obj.value = -10
del obj

assert revive is None
print("[ Done ]")

"""
$ python2 ./revive_test.py 
Revive
Destroy
[ Done ]

$ python3 ./revive_test.py 
Revive
[ Done ]
"""
