Why does cProfile not report the same time as deltas using `time.time()`?

To repro:

```sh
$ ./setup.sh python3 ./repro.py | tee ./output.txt
```

All timing results recorded with:

* Ubuntu 18.04
* CPython 3.6.9
* nvidia-driver-450 (450.102.04-0ubuntu0.18.04.1)
* NVidia Titan RTX
