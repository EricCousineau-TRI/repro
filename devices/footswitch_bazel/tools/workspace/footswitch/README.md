# `rgerganov/footswitch`

This repository provides command-line utilities for programming PCsensor and
Scythe foot switches, e.g., \
<https://www.amazon.com/PCsensor-Customized-Computer-Multimedia-Photoelectric/dp/B08SLX75K8>

## Configuring

Plug in device, run:

```sh
sudo ./tools/workspace/footswitch/udev_footpedal.sh
bazel build @footswitch//:footswitch
# Note: You may need to unplug and plug in the device at this point.
bazel-bin/external/footswitch/footswitch -1 -k pageup -2 -k down -3 -k pagedown
```

Then you should be able to run `pygame`-esque drivers, reconfiguring as you desire, e.g.:

```sh
bazel run //example:footswitch_demo
```
