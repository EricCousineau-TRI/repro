# Drake DiagramBuilder + scipy optimization: UB or OOM?

gist: https://gist.github.com/ggould-tri/5c0e1f276e90177a8aafd0fd7da6d213

## Doing the thing

```
$ ./setup.sh python3 ./repro.py
...
170 experiments run so far...
external/lcm/lcm/lcm_udpm.c pipe(create): Too many open files
LCM instance not initialized.  Ignoring call to fileno()
LCM instance not initialized.  Ignoring call to publish()
```
