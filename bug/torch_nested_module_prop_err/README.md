# Bug in PyTorch

Repro, tested in `torch==1.7.1` and `torch==1.8.1` (latest as of time of
writing).

Only manually tested on Ubuntu 18.04. Requires `apt install python3-venv`.

## Quick Run

```sh
$ ./setup.sh ./repro.py
Descriptor (good): <property object at 0x7fa030541638>
nested=A (good):
  1
nested=B, getattr (unexpected error):
  'Top' object has no attribute 'proxy_property'
nested=B, fget (expected error):
  'B' object has no attribute 'good_attr'
```
