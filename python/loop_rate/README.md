# Loop Rate Stuff

Ensuring we select the right thing.

Should look up formal names for this.

## Example

This example has an example 10 Hz (0.1 s) loop. Normally, work takes 0.09 s,
but there can be significant overrun at times (e.g. 0.34s). This just shows
some implications.

```
$ python ./example_timing.py
[ CatchupMode.Nothing ]
t = 0.1
  dt: 0.1
  overrun: 0
t = 0.2
  dt: 0.1
  overrun: 0
t = 0.3
  dt: 0.1
  overrun: 0
t = 0.64
  dt: 0.34
  overrun: 0.24
t = 0.73
  dt: 0.09
  overrun: 0.23
t = 0.82
  dt: 0.09
  overrun: 0.22
t = 0.91
  dt: 0.09
  overrun: 0.21

[ CatchupMode.Grid ]
t = 0.1
  dt: 0.1
  overrun: 0
t = 0.2
  dt: 0.1
  overrun: 0
t = 0.3
  dt: 0.1
  overrun: 0
t = 0.64
  dt: 0.34
  overrun: 0.24
t = 0.73
  dt: 0.09
  overrun: 0.03
t = 0.82
  dt: 0.09
  overrun: 0.02
t = 0.91
  dt: 0.09
  overrun: 0.01

[ CatchupMode.Reset ]
t = 0.1
  dt: 0.1
  overrun: 0
t = 0.2
  dt: 0.1
  overrun: 0
t = 0.3
  dt: 0.1
  overrun: 0
t = 0.64
  dt: 0.34
  overrun: 0.24
t = 0.74
  dt: 0.1
  overrun: 0
t = 0.84
  dt: 0.1
  overrun: 0
t = 0.94
  dt: 0.1
  overrun: 0
```
