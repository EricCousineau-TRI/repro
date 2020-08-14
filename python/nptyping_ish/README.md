# This is *not* `nptyping`

# (Experimental) Python Typing Annotations for NumPy, Torch

This provides a means to try out providing explicit structural information
about `np.ndarray` and `torch.Tensor` instances.

It is still unstable, and has not yet been vetted in the wild.

**Note**: This is named `typing_` to avoid colliding with the builtin package
`typing`. For an example of why this is done, see:
<https://github.com/RobotLocomotion/drake/issues/8041>

See the overview in `array.py`.

For actual `nptyping` project, see here:
<https://github.com/ramonhagenaars/nptyping>

## Testing

Tested on Ubuntu 18.04, with `python3-virtualenv` installed:

```sh
cd nptyping_ish
source setup.sh
python3 ./typing_/test/typing_test.py
```
