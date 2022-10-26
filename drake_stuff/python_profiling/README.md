# Example Profiling Drake and Python

To setup, in relevant terminals:

```sh
source ./setup.sh
```

## cprofile

https://docs.python.org/3.8/library/profile.html

```sh
python ./sample_sim.py --cprofile /tmp/cprofile.stats
python ./sample_sim.py --cprofile /tmp/cprofile_no_control.stats --no_control
snakeviz /tmp/cprofile.stats
snakeviz /tmp/cprofile_no_control.stats
```

viewers:
- snakeviz: https://jiffyclub.github.io/snakeviz/
- tuna: https://github.com/nschloe/tuna

Note that you will only see the Python code profiled, nothing from `pydrake`
bindings itself.

## py-spy

https://github.com/benfred/py-spy

Per their docs, if you don't want `sudo`, consider temporary relaxation of
`SYS_PTRACE` :(

```sh
echo "0" | sudo tee /proc/sys/kernel/yama/ptrace_scope  # Relax
# Run profiling...
echo "1" | sudo tee /proc/sys/kernel/yama/ptrace_scope  # Reenable
```

For more focused usage w/ code instrumentation:
```sh
python ./sample_sim.py --py_spy /tmp/pyspy.svg
python ./sample_sim.py --py_spy /tmp/pyspy_no_control.svg --no_control
x-www-browser /tmp/pyspy*.svg
```

See <https://github.com/benfred/py-spy/issues/531> for maybe better stuff.

For profiling without instrumentation:

```
py-spy record -o /tmp/pyspy.svg -- python ./sample_sim.py
py-spy record -o /tmp/pyspy_no_control.svg -- python ./sample_sim.py --no_control
x-www-browser /tmp/pyspy*.svg
```

**WARNING**: I (Eric) couldn't get `py-spy` to work thorugh Bazel and
`drake-ros` generated wrappers, even using `--subprocesses`. Unsure why.

## Transcribing Python to C++

Compare `components.py` and `components_cc_py.cc` for a very hacky example of
low-effort transcription. This attempts to maintain similar idioms in the code.
