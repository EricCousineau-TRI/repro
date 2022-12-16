# Drake Systems, State, Cache, and Initialization

**WARNING**: Not all of this is "Drake cannon". Please take these examples with
a *massive* grain of salt.

Exploring some options for making controllers, using cache, etc.

Requires `drake >= v1.11.0`.

## State, Algebraic Loops, and Prerequisites

Though a bit obtuse, see `haptic_device_example.py`, usage of
`prerequisites_of_calc` on line that calls `declare_pose_outputs()`.

## State, Cache, and Initialization

Run test to ensure it works as expected:

```sh
cd systems_controllers_rough_example
source ./setup.sh
python ./test/helpers_test.py
```

Then look at source.
