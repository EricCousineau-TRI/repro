# numpy.typing + Drake Expression / AutoDiff

Towards <https://stackoverflow.com/q/75068535/7829525>

```sh
$ source ./setup.sh
$ python ./repro.py
numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]
numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
numpy.ndarray[typing.Any, numpy.dtype[object]]
numpy.ndarray[typing.Any, numpy.dtype[pydrake.symbolic.Expression]]
numpy.ndarray[typing.Any, numpy.dtype[pydrake.autodiffutils.AutoDiffXd]]
```
