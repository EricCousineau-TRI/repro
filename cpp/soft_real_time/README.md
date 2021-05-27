Just checking CPU affinity and prioritization.

```
cd .../soft_real_time
bazel run :pthread_whachu_doin_c

bazel run :pthread_whachu_doin_py
```

See also:

`man taskset`

`man chrt`
