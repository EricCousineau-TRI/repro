# Soft Real-Time Stuff

## Background

See:

* `man taskset`
* `man chrt`

Suggestion for doing soft-realtime: `chrt -r 20`

Consider using `taskset -c {cpus}` as well to pin to process (if your code
doesn't need highly multithreaded stuff, like NumPy, OpenCV, etc.).

If pinning CPUs and you're hyperthreaded (vCPUs), use
`../../python/print_physical_cpus.py` to identify sets of vCPUs that correspond
to same physical core.

## rtprio without sudo

adopted from anzu, run:

```sh
sudo ./rtprio_setup.sh
```

## Checking how it propagates across threads

Just checking CPU affinity and prioritization.

```sh
cd .../soft_real_time
bazel run :pthread_whachu_doin_c

bazel run :pthread_whachu_doin_py
```
