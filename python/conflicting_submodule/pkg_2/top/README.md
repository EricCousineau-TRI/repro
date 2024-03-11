Shadowing with Python submodules

```sh
$ python --version
Python 3.10.12

$ ./run.sh 2>&1 | sed "s#${PWD}#\${PWD}##g"
[ Without 'top/__init__.py', works ]
${PWD}/pkg_1/top/sub_1.py
sub_1
${PWD}/pkg_2/top/sub_2.py
sub_2

[ With 'top/__init__.py', fails ]
Traceback (most recent call last):
  File "${PWD}/./main.py", line 1, in <module>
    from top import sub_1, sub_2
ImportError: cannot import name 'sub_2' from 'top' (${PWD}/pkg_1/top/__init__.py)
```
