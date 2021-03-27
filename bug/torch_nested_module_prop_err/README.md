# Bug in PyTorch

Repro, tested in `torch==1.7.1` and `torch==1.8.1` (latest as of time of
writing).

Only manually tested on Ubuntu 18.04. Requires `apt install python3-venv`.

Confirmed that this is due to: <br/>
<https://github.com/pytorch/pytorch/issues/49726> <br/>
Reduxed to use that code

## Quick Run

```sh
$ ./setup.sh ./repro.py
Descriptor (good): <property object at 0x7f8568189ef8>
A.attr (unexpected error):
  'A' object has no attribute 'attr'
prop.fget(A) (expected error):
  'A' object has no attribute 'bad_prop'
```
