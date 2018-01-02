class StrictMap(object):
  def __init__(self):
    self._values = dict()

  def _strict_key(self, key):
    # Ensures types are strictly scoped to the values.
    return (type(key), key)

  def add(self, key, value):
    skey = self._strict_key(key)
    assert skey not in self._values, "Already added: {}".format(skey)
    self._values[skey] = value

  def get(self, key):
    skey = self._strict_key(value)
    return self._values[skey]

s = StrictMap()
s.add(1, 'a')
s.add(2, 'b')
s.add(True, 'c')
s.add(1, 'd')