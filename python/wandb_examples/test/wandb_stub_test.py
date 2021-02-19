from contextlib import contextmanager
import os
import unittest
import sys

import tensorboard
import wandb


@contextmanager
def mock_attr(obj, attr, replace):
    """Temporarily replaces an attribute."""
    original = getattr(obj, attr)
    setattr(obj, attr, replace)
    yield original
    setattr(obj, attr, original)


@contextmanager
def restore_dict(container):
    """Copies original values of container, then restores them afterwards."""
    original = dict(container)
    yield
    container.clear()
    container.update(original)


class TestWandb(unittest.TestCase):
    def test_wandb(self):
        with restore_dict(os.environ):
            # https://docs.wandb.com/library/advanced/environment-variables
            os.environ["WANDB_CONFIG_DIR"] = os.environ["TEST_TMPDIR"]
            os.environ["WANDB_DIR"] = os.environ["TEST_TMPDIR"]
            os.environ["WANDB_MODE"] = "dryrun"
            wandb.init(project="test_project", sync_tensorboard=True)


@contextmanager
def count_calls(obj, attr):
    original = None
    num_calls = [0]

    def _replace(*args, **kwargs):
        num_calls[0] += 1
        return original(*args, **kwargs)

    with mock_attr(obj, attr, _replace) as original:
        yield num_calls


if __name__ == "__main__":
    with count_calls(wandb, "init") as num_calls:
        try:
            unittest.main()
        finally:
            print(f"Num calls: {num_calls[0]}")
