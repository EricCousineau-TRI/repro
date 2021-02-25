Why does cProfile not report the same time as deltas using `time.time()`?

To repro:

```sh
$ ./setup.sh python3 ./repro.py | tee ./output.txt
```
